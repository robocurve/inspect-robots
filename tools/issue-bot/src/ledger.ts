import { DurableObject } from "cloudflare:workers";
import {
  artifactDigest,
  digest,
  SHA,
  StageOutput,
  validateFiles,
  type StageRequest,
  type IssueSnapshot,
  type Publication,
  type FixPublication,
  type PublishedFix,
  type FileChange,
  type StageKind,
} from "./contracts";
import { POLICY } from "./policy";

export interface Job {
  id: string;
  issue: IssueSnapshot;
  created: number;
  state: "queued" | "running" | "fix" | "waiting_ci" | "held" | "done";
  context: string;
  duplicate: boolean;
  stage: string | null;
  next: StageKind | null;
  plan: string;
  approvedPlan: string;
  feedback: string;
  files: FileChange[];
  approvedArtifact: string;
  planRounds: number;
  codeRounds: number;
  summary: string;
  checks: string[];
  published: PublishedFix | null;
}
export interface Stage {
  request: StageRequest;
  started: number;
  launched: boolean;
  cleaned: boolean;
  closed: boolean;
  output: StageOutput | null;
  consumed: boolean;
  failure: string | null;
}
interface Outbox {
  publication: Publication;
  delivered: boolean;
}
export class IssueLedger extends DurableObject<IssueEnv> {
  constructor(ctx: DurableObjectState, env: IssueEnv) {
    super(ctx, env);
    ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,issue INTEGER NOT NULL,created INTEGER NOT NULL,data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS stages(id TEXT PRIMARY KEY,token TEXT UNIQUE,receipt TEXT UNIQUE,data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY,data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,issue INTEGER,month TEXT,reserved INTEGER,actual INTEGER); CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);`,
    );
  }
  private get<T>(table: string, id: string): T | null {
    return (
      this.ctx.storage.sql
        .exec<{ data: string }>(`SELECT data FROM ${table} WHERE id=?`, id)
        .toArray()
        .map((r) => JSON.parse(r.data) as T)[0] ?? null
    );
  }
  private save(job: Job) {
    this.ctx.storage.sql.exec(
      "UPDATE jobs SET data=? WHERE id=?",
      JSON.stringify(job),
      job.id,
    );
  }
  private saveStage(stage: Stage) {
    this.ctx.storage.sql.exec(
      "UPDATE stages SET data=? WHERE id=?",
      JSON.stringify(stage),
      stage.request.id,
    );
  }
  async register(
    issue: IssueSnapshot,
    context: string,
    duplicate: boolean,
    requestId = "",
  ) {
    const active = (await this.pending()).find(
      (j) => j.issue.number === issue.number,
    );
    if (active) return active.id;
    const id = (
      await digest(
        `${issue.number}:${issue.revision}:${issue.base}:issue-policy-1:${requestId}`,
      )
    ).slice(0, 48);
    const concurrent = this.ctx.storage.sql
      .exec<{ data: string }>(
        "SELECT data FROM jobs WHERE issue=?",
        issue.number,
      )
      .toArray()
      .map((r) => JSON.parse(r.data) as Job)
      .find((j) => !["done", "held"].includes(j.state));
    if (concurrent) return concurrent.id;
    const job: Job = {
      id,
      issue,
      created: Date.now(),
      state: "queued",
      context,
      duplicate,
      stage: null,
      next: "triage",
      plan: "",
      approvedPlan: "",
      feedback: "",
      files: [],
      approvedArtifact: "",
      planRounds: 0,
      codeRounds: 0,
      summary: "",
      checks: [],
      published: null,
    };
    this.ctx.storage.sql.exec(
      "INSERT OR IGNORE INTO jobs VALUES(?,?,?,?)",
      id,
      issue.number,
      job.created,
      JSON.stringify(job),
    );
    return id;
  }
  async job(id: string) {
    return this.get<Job>("jobs", id);
  }
  async stage(id: string) {
    return this.get<Stage>("stages", id);
  }
  async pending() {
    return this.ctx.storage.sql
      .exec<{ data: string }>("SELECT data FROM jobs ORDER BY created,id")
      .toArray()
      .map((r) => JSON.parse(r.data) as Job)
      .filter((j) => !["done", "held"].includes(j.state));
  }
  async queueState() {
    return {
      owner: this.owner(),
      jobs: await this.pending(),
      outbox: await this.outbox(),
    };
  }
  private owner() {
    return (
      this.ctx.storage.sql
        .exec<{ value: string }>("SELECT value FROM meta WHERE key='slot'")
        .toArray()[0]?.value ?? null
    );
  }
  async claim(id: string) {
    if (this.owner()) return this.owner() === id;
    const first = (await this.pending()).find((j) =>
      ["queued", "running"].includes(j.state),
    );
    // No await after reading the slot and before acquiring it: pending() above can yield.
    if (this.owner()) return this.owner() === id;
    if (first?.id !== id) return false;
    this.ctx.storage.sql.exec("INSERT INTO meta VALUES('slot',?)", id);
    return true;
  }
  async release(id: string) {
    const job = this.get<Job>("jobs", id);
    if (this.owner() !== id || !job) return false;
    const stage = job.stage ? this.get<Stage>("stages", job.stage) : null;
    if (stage && !stage.cleaned) return false;
    this.ctx.storage.sql.exec("DELETE FROM meta WHERE key='slot'");
    return true;
  }
  async refreshUnstarted(id: string, current: IssueSnapshot) {
    const job = this.get<Job>("jobs", id);
    if (
      !job ||
      this.owner() !== id ||
      job.state !== "queued" ||
      job.stage !== null ||
      job.next !== "triage" ||
      current.number !== job.issue.number ||
      current.state !== "open" ||
      current.revision !== job.issue.revision ||
      !SHA.test(current.base)
    )
      return false;
    // Pin code only when admitted to the sandbox, before any paid evidence exists.
    // Keep the durable job identity and lifetime charges unchanged.
    job.issue.base = current.base;
    this.save(job);
    return true;
  }
  async tickClaim(id: string) {
    const now = Date.now();
    const key = `tick:${id}`;
    const row = this.ctx.storage.sql
      .exec<{ value: string }>("SELECT value FROM meta WHERE key=?", key)
      .toArray()[0];
    if (row && JSON.parse(row.value).expires > now) return null;
    const token = crypto.randomUUID();
    this.ctx.storage.sql.exec(
      "INSERT OR REPLACE INTO meta VALUES(?,?)",
      key,
      JSON.stringify({ token, expires: now + 120000 }),
    );
    return token;
  }
  async tickRelease(id: string, token: string) {
    const key = `tick:${id}`;
    const row = this.ctx.storage.sql
      .exec<{ value: string }>("SELECT value FROM meta WHERE key=?", key)
      .toArray()[0];
    if (row && JSON.parse(row.value).token === token)
      this.ctx.storage.sql.exec("DELETE FROM meta WHERE key=?", key);
  }
  async prepare(id: string) {
    const job = this.get<Job>("jobs", id);
    if (!job || this.owner() !== id || !job.next) return null;
    if (job.stage) {
      const old = this.get<Stage>("stages", job.stage);
      if (old && !old.consumed) return old;
    }
    if (
      job.next === "implement" &&
      job.approvedPlan !== (await digest(job.plan))
    )
      throw new Error("plan_approval_mismatch");
    const request: StageRequest = {
      id: crypto.randomUUID(),
      jobId: id,
      kind: job.next,
      base: job.issue.base,
      issue: job.issue,
      context: POLICY + "\n" + job.context,
      plan: job.plan,
      feedback: job.feedback,
      files: job.files,
      inputDigest: "",
      token: capability(),
      checkpointToken: capability(),
      sandbox: crypto.randomUUID(),
    };
    request.inputDigest = await digest(
      JSON.stringify({
        kind: request.kind,
        context: request.context,
        base: request.base,
        issue: request.issue,
        plan: request.plan,
        feedback: request.feedback,
        files: request.files,
      }),
    );
    if (JSON.stringify(request).length > 1_400_000) {
      await this.hold(id, "stage_context_too_large");
      return null;
    }
    // Reload after hashing: another tick may have prepared the next stage.
    const live = this.get<Job>("jobs", id)!;
    if (live.stage !== job.stage)
      return live.stage ? this.get<Stage>("stages", live.stage) : null;
    const stage: Stage = {
      request,
      started: Date.now(),
      launched: false,
      cleaned: false,
      closed: false,
      output: null,
      consumed: false,
      failure: null,
    };
    this.ctx.storage.sql.exec(
      "INSERT INTO stages VALUES(?,?,?,?)",
      request.id,
      request.token,
      request.checkpointToken,
      JSON.stringify(stage),
    );
    job.stage = request.id;
    job.state = "running";
    this.save(job);
    return stage;
  }
  async claimLaunch(id: string) {
    const s = this.get<Stage>("stages", id);
    if (!s || s.launched || s.closed) return false;
    const reservation = 100000;
    const issue = s.request.issue.number,
      month = new Date().toISOString().slice(0, 7);
    const issueUsed = this.ctx.storage.sql
      .exec<{ n: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) n FROM charges WHERE issue=?",
        issue,
      )
      .one().n;
    const monthUsed = this.ctx.storage.sql
      .exec<{ n: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) n FROM charges WHERE month=?",
        month,
      )
      .one().n;
    if (
      issueUsed + reservation > Number(this.env.ISSUE_LIMIT_MICROS) ||
      monthUsed + reservation > Number(this.env.MONTH_LIMIT_MICROS)
    )
      return false;
    this.ctx.storage.sql.exec(
      "INSERT INTO charges VALUES(?,?,?,?,NULL)",
      `sandbox:${id}`,
      issue,
      month,
      reservation,
    );
    s.launched = true;
    this.saveStage(s);
    return true;
  }
  async session(token: string) {
    const row = this.ctx.storage.sql
      .exec<{ data: string }>("SELECT data FROM stages WHERE token=?", token)
      .toArray()[0];
    if (!row) return null;
    const s = JSON.parse(row.data) as Stage;
    return s.closed || s.output || Date.now() - s.started > 45 * 60000
      ? null
      : s;
  }
  async completed(token: string) {
    const row = this.ctx.storage.sql
      .exec<{ data: string }>(
        "SELECT data FROM stages WHERE token=? OR receipt=?",
        token,
        token,
      )
      .toArray()[0];
    if (!row) throw new Error("invalid_stage");
    return (JSON.parse(row.data) as Stage).output !== null;
  }
  async checkpoint(token: string, body: string) {
    if (body.length > 2_000_000) throw new Error("output_too_large");
    const row = this.ctx.storage.sql
      .exec<{ data: string }>("SELECT data FROM stages WHERE receipt=?", token)
      .toArray()[0];
    if (!row) throw new Error("invalid_receipt");
    const s = JSON.parse(row.data) as Stage;
    if (s.output || s.closed) return;
    const output = StageOutput.parse(JSON.parse(body));
    output.files = validateFiles(output.files);
    s.output = output;
    s.closed = true;
    this.saveStage(s);
  }
  async close(id: string, reason?: string) {
    const s = this.get<Stage>("stages", id);
    if (!s) return;
    s.closed = true;
    s.failure = reason ?? s.failure;
    this.saveStage(s);
  }
  async cleaned(id: string) {
    const s = this.get<Stage>("stages", id);
    if (!s) return;
    s.cleaned = true;
    this.saveStage(s);
  }
  async failSession(token: string, reason: string) {
    const s = await this.session(token);
    if (s) {
      s.failure = reason;
      this.saveStage(s);
    }
  }
  async costs(issue: number) {
    const rows = this.ctx.storage.sql
      .exec<{ amount: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) AS amount FROM charges WHERE issue=?",
        issue,
      )
      .toArray();
    return rows[0].amount;
  }
  async remaining(issue: number) {
    const a = this.ctx.storage.sql
      .exec<{ amount: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) AS amount FROM charges WHERE issue=?",
        issue,
      )
      .toArray()[0].amount;
    const b = this.ctx.storage.sql
      .exec<{ amount: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) AS amount FROM charges WHERE month=?",
        new Date().toISOString().slice(0, 7),
      )
      .toArray()[0].amount;
    return Math.max(
      0,
      Math.min(
        Number(this.env.ISSUE_LIMIT_MICROS) - a,
        Number(this.env.MONTH_LIMIT_MICROS) - b,
      ),
    );
  }
  async reserve(token: string, id: string, amount: number) {
    const s = await this.session(token);
    if (!s || !Number.isSafeInteger(amount) || amount < 0) return false;
    const allowance = await this.remaining(s.request.issue.number); // all related writes below are synchronous
    const live = await this.session(token);
    if (!live) return false;
    const issue = live.request.issue.number;
    const month = new Date().toISOString().slice(0, 7);
    const usedIssue = this.ctx.storage.sql
      .exec<{ n: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) n FROM charges WHERE issue=?",
        issue,
      )
      .one().n;
    const usedMonth = this.ctx.storage.sql
      .exec<{ n: number }>(
        "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) n FROM charges WHERE month=?",
        month,
      )
      .one().n;
    if (
      amount > allowance ||
      usedIssue + amount > Number(this.env.ISSUE_LIMIT_MICROS) ||
      usedMonth + amount > Number(this.env.MONTH_LIMIT_MICROS)
    )
      return false;
    if (
      this.ctx.storage.sql
        .exec("SELECT id FROM charges WHERE id=?", id)
        .toArray().length
    )
      return false;
    this.ctx.storage.sql.exec(
      "INSERT INTO charges VALUES(?,?,?,?,NULL)",
      id,
      issue,
      month,
      amount,
    );
    return true;
  }
  async settle(id: string, amount: number) {
    if (!Number.isSafeInteger(amount) || amount < 0)
      throw new Error("invalid_usage");
    const charge = this.ctx.storage.sql
      .exec<{ reserved: number; actual: number | null }>(
        "SELECT reserved,actual FROM charges WHERE id=?",
        id,
      )
      .toArray()[0];
    if (!charge || charge.actual !== null) return;
    this.ctx.storage.sql.exec(
      "UPDATE charges SET actual=? WHERE id=?",
      Math.max(0, amount),
      id,
    );
    if (amount > charge.reserved)
      this.ctx.storage.sql.exec(
        "INSERT OR REPLACE INTO meta VALUES('billing_hold','true')",
      );
  }
  async billingHold() {
    return (
      this.ctx.storage.sql
        .exec("SELECT value FROM meta WHERE key='billing_hold'")
        .toArray().length > 0
    );
  }
  private notice(job: Job, status: string, summary: string, details: string[]) {
    const publication: Publication = {
      jobId: job.id,
      issue: job.issue,
      status,
      summary,
      details: boundedDetails(details),
      costMicros: 0,
      ...(job.published ? { pr: job.published.number } : {}),
    };
    this.ctx.storage.sql.exec(
      "INSERT OR IGNORE INTO outbox VALUES(?,?)",
      `${job.id}:${status}`,
      JSON.stringify({ publication, delivered: false }),
    );
  }
  async hold(id: string, reason: string) {
    const job = this.get<Job>("jobs", id);
    if (!job || job.state === "done") return;
    job.state = "held";
    job.next = null;
    this.save(job);
    const result = job.stage
      ? this.get<Stage>("stages", job.stage)?.output?.result
      : null;
    const currentEvidence = result
      ? [
          result.summary,
          ...result.findings,
          ...result.limitations,
          ...result.checks.map(
            (c) => `Recorded check or next validation: ${c}`,
          ),
        ]
      : [];
    this.notice(job, "REQUIRE_REVIEWER", `Automation paused: ${reason}.`, [
      ...currentEvidence,
      ...(job.feedback ? [`Earlier revision feedback: ${job.feedback}`] : []),
      "Resolve the specific blocker and request /triage to resume with a fresh assessment. Saved evidence and previous spending are retained.",
    ]);
  }
  async consume(id: string) {
    const job = this.get<Job>("jobs", id);
    if (!job || !job.stage) return;
    const s = this.get<Stage>("stages", job.stage);
    if (!s || s.consumed || !s.cleaned) return;
    const out = s.output;
    const r = out?.result;
    if (s.failure || !out || out.exitCode !== 0 || out.failure || !r) {
      s.consumed = true;
      this.saveStage(s);
      await this.hold(
        id,
        s.failure ?? out?.failure ?? "stage_execution_incomplete",
      );
      return;
    }
    const approve =
      r.status === "APPROVE" &&
      r.findings.length === 0 &&
      r.limitations.length === 0;
    const kind = s.request.kind;
    if (kind === "triage") {
      if (
        ![
          "CONFIRMED",
          "NEEDS_INFO",
          "NOT_REPRODUCED",
          "DUPLICATE",
          "FIX_PROPOSED",
          "REQUIRE_REVIEWER",
        ].includes(r.status)
      )
        return this.hold(id, "invalid_triage_status");
      const serious =
        r.status === "CONFIRMED" &&
        r.serious &&
        r.evidence.length > 0 &&
        r.limitations.length === 0 &&
        !job.duplicate;
      // The persisted legacy flag records an existing fix PR, not a duplicate issue.
      const status = job.duplicate ? "FIX_PROPOSED" : r.status;
      this.notice(job, status, r.summary, [...r.evidence, ...r.limitations]);
      job.next = serious ? "plan" : null;
      if (!serious) job.state = "done";
    } else if (kind === "plan") {
      if (r.status !== "PLAN" || !r.plan.trim())
        return this.hold(id, "invalid_plan");
      job.plan = r.plan;
      job.next = "plan_review";
    } else if (kind === "plan_review") {
      if (approve) {
        job.approvedPlan = await digest(s.request.plan);
        job.next = "implement";
      } else if (r.status === "REQUEST_CHANGES" && ++job.planRounds < 3) {
        job.feedback = [...r.findings, ...r.limitations].join("\n");
        job.next = "plan";
      } else return this.hold(id, "plan_review_not_approved");
    } else if (kind === "implement") {
      if (r.status !== "IMPLEMENTED" || out.files.length === 0)
        return this.hold(id, "implementation_missing");
      job.files = out.files;
      job.next = "code_review";
    } else {
      if (
        approve &&
        out.executions.some((e) => isValidationCommand(e.command)) &&
        out.executions
          .filter((e) => isValidationCommand(e.command))
          .every((e) => e.exitCode === 0) &&
        r.checks.length > 0
      ) {
        job.approvedArtifact = await artifactDigest(s.request.files);
        job.summary = r.summary;
        job.checks = out.executions
          .filter((e) => isValidationCommand(e.command))
          .slice(0, 30)
          .map((e) => `${e.command.slice(0, 1000)}: exit ${e.exitCode}`);
        job.next = null;
        job.state = "fix";
      } else if (r.status === "REQUEST_CHANGES" && ++job.codeRounds < 3) {
        job.feedback = [...r.findings, ...r.limitations].join("\n");
        job.next = "implement";
      } else return this.hold(id, "code_review_missing_successful_validation");
    }
    s.consumed = true;
    this.saveStage(s);
    this.save(job);
  }
  async fix(id: string): Promise<FixPublication> {
    const j = this.get<Job>("jobs", id);
    if (
      !j ||
      !["fix", "waiting_ci"].includes(j.state) ||
      !j.approvedArtifact ||
      j.approvedArtifact !== (await artifactDigest(j.files))
    )
      throw new Error("artifact_approval_mismatch");
    return {
      jobId: j.id,
      issue: j.issue,
      files: j.files,
      artifactDigest: j.approvedArtifact,
      approvedDigest: j.approvedArtifact,
      plan: j.plan,
      summary: j.summary,
      checks: j.checks,
    };
  }
  async published(id: string, fix: PublishedFix) {
    const j = this.get<Job>("jobs", id);
    if (!j) return;
    j.published = fix;
    j.state = "waiting_ci";
    this.save(j);
  }
  async ready(id: string) {
    const j = this.get<Job>("jobs", id);
    if (!j) return;
    j.state = "done";
    this.save(j);
    this.notice(
      j,
      "PR_READY",
      "The independently reviewed fix is ready for the PR reviewer.",
      j.checks,
    );
  }
  async outbox() {
    return this.ctx.storage.sql
      .exec<{ id: string; data: string }>("SELECT id,data FROM outbox")
      .toArray()
      .map((r) => ({ id: r.id, ...(JSON.parse(r.data) as Outbox) }))
      .filter((r) => !r.delivered);
  }
  async delivered(id: string) {
    const row = this.get<Outbox>("outbox", id);
    if (row) {
      row.delivered = true;
      this.ctx.storage.sql.exec(
        "UPDATE outbox SET data=? WHERE id=?",
        JSON.stringify(row),
        id,
      );
    }
  }
}

/** Classify executable words, never strings merely printed by echo/printf. */
export function isValidationCommand(command: string, depth = 0): boolean {
  if (depth > 4 || command.length > 12000) return false;
  const segments = shellSegments(command);
  if (segments === null) return false;
  return segments.some((words) => {
    let offset = 0;
    if (basename(words[0] ?? "") === "env") {
      offset++;
      if (words[offset] === "-i") offset++;
    }
    while (/^[A-Za-z_][A-Za-z0-9_]*=/.test(words[offset] ?? "")) offset++;
    const executable = basename(words[offset] ?? "");
    const args = words.slice(offset + 1);
    if (["sh", "bash", "dash", "zsh", "ksh"].includes(executable)) {
      while (["--noprofile", "--norc"].includes(args[0])) args.shift();
      return (
        args.length === 2 &&
        /^-[a-zA-Z]*c[a-zA-Z]*$/.test(args[0]) &&
        isValidationCommand(args[1], depth + 1)
      );
    }
    if (["pytest", "ruff", "mypy", "pyright"].includes(executable)) return true;
    if (["python", "python3"].includes(executable)) {
      return args[0] === "-m" && ["pytest", "unittest"].includes(args[1]);
    }
    if (["cargo", "go"].includes(executable)) return args[0] === "test";
    if (executable === "npm")
      return (
        args[0] === "test" ||
        (args[0] === "run" &&
          ["test", "check", "lint", "typecheck", "build"].includes(args[1]))
      );
    if (executable === "pnpm")
      return ["test", "check", "lint", "typecheck", "build"].includes(args[0]);
    if (executable === "uv" && args[0] === "run") {
      return (
        ["pytest", "ruff", "mypy", "pyright"].includes(
          basename(args[1] ?? ""),
        ) ||
        (["python", "python3"].includes(basename(args[1] ?? "")) &&
          args[2] === "-m" &&
          ["pytest", "unittest"].includes(args[3]))
      );
    }
    return false;
  });
}

function basename(word: string): string {
  return word.split("/").at(-1) ?? "";
}

/** Small conservative lexer for ordinary shell argv and sequential commands.
 * Quotes protect embedded separators; unsupported substitutions are not parsed.
 */
function shellSegments(source: string): string[][] | null {
  const segments: string[][] = [];
  let words: string[] = [],
    word = "";
  let quote: "'" | '"' | null = null,
    started = false;
  const finishWord = () => {
    if (started) words.push(word);
    word = "";
    started = false;
  };
  const finishSegment = () => {
    finishWord();
    if (words.length) segments.push(words);
    words = [];
  };
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (char === "\\" && quote !== "'") {
      if (i + 1 >= source.length) return null;
      const next = source[++i];
      if (next !== "\n") {
        word += next;
        started = true;
      }
      continue;
    }
    if (quote) {
      if (char === quote) quote = null;
      else word += char;
      started = true;
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      started = true;
      continue;
    }
    if (char === "#" && !started) {
      while (i < source.length && source[i] !== "\n") i++;
      finishSegment();
      continue;
    }
    if (char === "\n" || char === ";" || char === "&" || char === "|") {
      finishSegment();
      continue;
    }
    if (/\s/.test(char)) {
      finishWord();
      continue;
    }
    // Heredoc bodies and substitutions need a full shell parser. Never treat
    // text embedded there as an independently successful validation command.
    if (
      (char === "<" && source[i + 1] === "<") ||
      char === "`" ||
      (char === "$" && source[i + 1] === "(")
    )
      return null;
    if (char === ">" || char === "<") {
      word += char;
      started = true;
      if ([">", "&"].includes(source[i + 1])) word += source[++i];
      continue;
    }
    word += char;
    started = true;
  }
  if (quote) return null;
  finishSegment();
  return segments;
}

function capability(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(32)), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

function boundedDetails(details: string[]): string[] {
  let remaining = 30000;
  return details.flatMap((detail) => {
    if (remaining <= 0) return [];
    const value = detail.slice(0, Math.min(3000, remaining));
    remaining -= value.length;
    return [value];
  });
}
