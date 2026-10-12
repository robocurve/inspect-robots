import {
  WorkflowEntrypoint,
  type WorkflowEvent,
  type WorkflowStep,
} from "cloudflare:workers";
import {
  boundedText,
  linksIssue,
  MAINTAINER_ID,
  REPO,
  SHA,
  semanticRevision,
  type IssueSnapshot,
} from "./contracts";
export { IssueLedger } from "./ledger";
export { ModelGateway } from "./model-gateway";

export async function verifySignature(
  body: string,
  signature: string | null,
  secret: string,
) {
  if (!secret || !signature || !/^sha256=[0-9a-f]{64}$/.test(signature))
    return false;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"],
  );
  return crypto.subtle.verify(
    "HMAC",
    key,
    Uint8Array.from(signature.slice(7).match(/../g)!, (x) => parseInt(x, 16)),
    new TextEncoder().encode(body),
  );
}
async function read(
  env: Pick<IssueEnv, "PUBLISHER">,
  path: string,
): Promise<Record<string, any>> {
  return JSON.parse(await env.PUBLISHER.read(path));
}
export async function snapshot(
  env: Pick<IssueEnv, "PUBLISHER">,
  number: number,
): Promise<IssueSnapshot> {
  const issue = await read(env, `/issues/${number}`),
    base = await read(env, "/commits/main");
  if (
    issue.pull_request ||
    issue.number !== number ||
    !SHA.test(base.sha) ||
    !Number.isSafeInteger(issue.user?.id) ||
    typeof issue.user?.login !== "string" ||
    typeof issue.title !== "string" ||
    typeof issue.state !== "string"
  )
    throw new Error("invalid_issue_snapshot");
  return {
    number,
    title: issue.title,
    body: issue.body ?? "",
    author: issue.user.login,
    authorId: issue.user.id,
    state: issue.state,
    revision: await semanticRevision(
      issue as { title: string; body: string; state: string },
    ),
    base: base.sha,
  };
}
export async function enqueue(
  env: IssueEnv,
  number: number,
  maintainerOnly = false,
  requestId = "",
) {
  if (env.ENABLED !== "true") throw new Error("issue_bot_disabled");
  if (!Number.isSafeInteger(number) || number < 1)
    throw new Error("invalid_issue");
  const issue = await snapshot(env, number);
  if (issue.state !== "open") throw new Error("issue_not_open");
  if (maintainerOnly && issue.authorId !== MAINTAINER_ID)
    throw new Error("trial_issue_not_maintainer_authored");
  const comments = JSON.parse(
    await env.PUBLISHER.read(`/issues/${number}/comments?per_page=100`),
  ) as Array<Record<string, any>>;
  const pulls = JSON.parse(
    await env.PUBLISHER.read("/pulls?state=open&per_page=100"),
  ) as Array<Record<string, any>>;
  let lastPageSize = pulls.length;
  for (let page = 2; lastPageSize === 100 && page <= 5; page++) {
    const next = JSON.parse(
      await env.PUBLISHER.read(`/pulls?state=open&per_page=100&page=${page}`),
    ) as Array<Record<string, any>>;
    lastPageSize = next.length;
    pulls.push(...next);
  }
  const searchIncomplete = lastPageSize === 100;
  const linked = pulls.filter((p) => linksIssue(String(p.body ?? ""), number));
  const context = JSON.stringify({
    comments: comments
      .filter(
        (c) =>
          c.user?.type !== "Bot" &&
          !(c.body ?? "").includes("<!-- robocurve-issue"),
      )
      .map((c) => ({
        author: c.user?.login,
        body: String(c.body ?? "").slice(0, 6000),
      }))
      .slice(-30),
    openPullRequests: pulls.map((p) => ({
      number: p.number,
      title: p.title,
      body: String(p.body ?? "").slice(0, 3000),
    })),
    existingFixes: linked.map((p) => p.number),
    pullListMayBeTruncated: searchIncomplete,
  });
  // Conservatively avoid automatic fixes when the duplicate search is incomplete.
  const id = await env.LEDGER.getByName("coordinator").register(
    issue,
    context,
    linked.length > 0,
    requestId,
  );
  if (searchIncomplete)
    await env.LEDGER.getByName("coordinator").hold(
      id,
      "duplicate_search_incomplete",
    );
  try {
    await env.ISSUE.create({
      id: `${id}-${Math.floor(Date.now() / 300000)}`,
      params: { id },
    });
  } catch {
    /* saved job is recovered by cron */
  }
  return id;
}
export async function handleWebhook(request: Request, env: IssueEnv) {
  let body: string;
  try {
    body = await boundedText(request, 1_000_000);
  } catch {
    return new Response("Too large", { status: 413 });
  }
  if (
    !(await verifySignature(
      body,
      request.headers.get("x-hub-signature-256"),
      env.GITHUB_WEBHOOK_SECRET,
    ))
  )
    return new Response("Unauthorized", { status: 401 });
  const event = request.headers.get("x-github-event");
  if (event === "ping") return new Response("pong");
  if (env.ENABLED !== "true") return new Response("Disabled", { status: 503 });
  let data;
  try {
    data = JSON.parse(body);
  } catch {
    return new Response("Invalid JSON", { status: 400 });
  }
  if (
    data.repository?.full_name !== REPO ||
    Number(env.INSTALLATION_ID) <= 0 ||
    data.installation?.id !== Number(env.INSTALLATION_ID)
  )
    return new Response("Ignored", { status: 202 });
  if (
    event === "issues" &&
    data.action === "opened" &&
    !data.issue?.pull_request
  )
    await enqueue(env, data.issue.number);
  else if (
    event === "issue_comment" &&
    data.action === "created" &&
    !data.issue?.pull_request &&
    data.comment?.user?.id === MAINTAINER_ID &&
    /^\/triage\s*$/.test(data.comment?.body ?? "")
  )
    await enqueue(env, data.issue.number, false, String(data.comment.id));
  return new Response("Accepted", { status: 202 });
}
// Startup waits up to 60 s for the container and 60 s for its port; allow generous slack.
const LAUNCH_GRACE_MS = 5 * 60000;
// The sandbox destroys itself at 42 min, so a stopped container after that is the normal timeout.
const STAGE_EXPIRY_MS = 41 * 60000;
/** Public hold reasons carry only fixed codes; prose error text goes to Worker logs. */
function launchFailure(e: unknown) {
  const message = e instanceof Error ? e.message : String(e);
  console.error(JSON.stringify({ event: "issue_stage_launch_error", message }));
  return /^[a-z_]{3,60}$/.test(message)
    ? `stage_launch_failed (${message})`
    : "stage_launch_failed";
}
export async function tick(env: IssueEnv, id: string) {
  const ledger = env.LEDGER.getByName("coordinator");
  const tickToken = await ledger.tickClaim(id);
  if (!tickToken) return false;
  try {
    let job = await ledger.job(id);
    if (!job) return true;
    // A terminal job may still own a container after a failed cleanup; never skip it.
    if (job.stage) {
      const s = await ledger.stage(job.stage);
      if (
        s &&
        !s.cleaned &&
        (s.output || s.closed || Date.now() - s.started > 45 * 60000)
      ) {
        const reason = s.output
          ? undefined
          : (s.failure ??
            (s.modelUsed === false ? s.launchError : undefined) ??
            "stage_timeout");
        await ledger.close(s.request.id, reason);
        await env.RUNNER.cleanup(s.request.sandbox);
        await ledger.cleaned(s.request.id);
        if (reason) await ledger.hold(id, reason);
      }
    }
    if (["done", "held"].includes(job.state)) {
      await ledger.release(id);
      return true;
    }
    if (env.ENABLED !== "true") return false;
    if (["fix", "waiting_ci"].includes(job.state)) {
      const fix = await ledger.fix(id);
      let published = job.published;
      try {
        if (!published) {
          published = await env.PUBLISHER.createFix(fix);
          await ledger.published(id, published);
        }
        if (await env.PUBLISHER.ready(fix, published)) await ledger.ready(id);
      } catch (e) {
        const reason = e instanceof Error ? e.message : "";
        if (
          /^(ci_failed|issue_changed|base_changed|unapproved_artifact|branch_changed|ambiguous_pull_request|pull_request_changed|unrecognized_pull_request|competing_fix|duplicate_search_incomplete|stale_|.*mismatch|.*unsafe|.*conflict)/.test(
            reason,
          )
        )
          await ledger.hold(id, reason);
        else throw e;
      }
      await ledger.release(id);
      return (await ledger.job(id))?.state === "done";
    }
    if (!(await ledger.claim(id))) return false;
    const current = await snapshot(env, job.issue.number);
    if (!job.stage && job.next === "triage") {
      await ledger.refreshUnstarted(id, current);
      job = (await ledger.job(id))!;
    }
    // Every stage runs on the base pinned at admission, even after main advances:
    // a busy main would otherwise stop every multi-stage fix. The publisher opens
    // the PR on that base; GitHub mergeability and CI judge it against newer main.
    if (current.state !== "open" || current.revision !== job.issue.revision) {
      await ledger.hold(id, "issue_changed");
      if (job.stage) {
        const s = await ledger.stage(job.stage);
        if (s && !s.cleaned) {
          await ledger.close(s.request.id);
          await env.RUNNER.cleanup(s.request.sandbox);
          await ledger.cleaned(s.request.id);
        }
      }
      await ledger.release(id);
      return true;
    }
    if (
      (!job.stage || (await ledger.stage(job.stage))?.consumed) &&
      ((await ledger.billingHold()) ||
        (await ledger.remaining(job.issue.number)) < 100000)
    ) {
      await ledger.hold(id, "budget_exhausted");
      if (job.stage) {
        const s = await ledger.stage(job.stage);
        if (s && !s.cleaned) {
          await ledger.close(s.request.id);
          await env.RUNNER.cleanup(s.request.sandbox);
          await ledger.cleaned(s.request.id);
        }
      }
      await ledger.release(id);
      return true;
    }
    const stage = await ledger.prepare(id);
    if (!stage) return false;
    if (!stage.launched && (await ledger.claimLaunch(stage.request.id))) {
      try {
        await env.RUNNER.start(stage.request);
      } catch (e) {
        /* Launch acknowledgement ambiguous: only poll the saved sandbox. */
        await ledger.launchFailed(stage.request.id, launchFailure(e));
      }
      return false;
    }
    if (!stage.output && !stage.closed) {
      const phase = await env.RUNNER.poll(
        stage.request.sandbox,
        stage.request.checkpointToken,
      );
      const age = Date.now() - stage.started;
      // A runner deployed before phases existed returns a boolean; never treat it as stopped.
      if (
        phase === "not_started" &&
        age > LAUNCH_GRACE_MS &&
        age < STAGE_EXPIRY_MS
      ) {
        const reason = await ledger.closeNeverStarted(stage.request.id);
        if (reason) {
          await env.RUNNER.cleanup(stage.request.sandbox);
          await ledger.cleaned(stage.request.id);
          if (await ledger.retryLaunch(id)) return false;
          await ledger.hold(id, reason);
          await ledger.release(id);
          return true;
        }
      }
    }
    const latest = await ledger.stage(stage.request.id);
    if (latest?.output) {
      await ledger.close(stage.request.id);
      if (!latest.cleaned) {
        await env.RUNNER.cleanup(stage.request.sandbox);
        await ledger.cleaned(stage.request.id);
      }
      await ledger.consume(id);
      job = (await ledger.job(id))!;
      if (!job.next) await ledger.release(id);
    }
    return ["done", "held"].includes((await ledger.job(id))?.state ?? "");
  } finally {
    await ledger.tickRelease(id, tickToken);
  }
}
export async function deliver(env: IssueEnv) {
  const ledger = env.LEDGER.getByName("coordinator");
  for (const item of await ledger.outbox()) {
    try {
      const publication = {
        ...item.publication,
        costMicros: await ledger.costs(item.publication.issue.number),
      };
      if (await env.PUBLISHER.publish(publication))
        await ledger.delivered(item.id);
    } catch {
      console.error(
        JSON.stringify({
          event: "issue_publication_deferred",
          job: item.publication.jobId,
        }),
      );
    }
  }
}
export class IssueWorkflow extends WorkflowEntrypoint<
  IssueEnv,
  {
    id?: string;
    issue?: number;
    inspect?: boolean;
    export?: number;
    requestId?: string;
  }
> {
  async run(
    event: WorkflowEvent<{
      id?: string;
      issue?: number;
      inspect?: boolean;
      export?: number;
      requestId?: string;
    }>,
    step: WorkflowStep,
  ) {
    const ledger = this.env.LEDGER.getByName("coordinator");
    if (event.payload.inspect) return ledger.queueState();
    if (event.payload.export !== undefined) {
      if (!Number.isSafeInteger(event.payload.export))
        throw new Error("invalid_issue");
      return ledger.export(event.payload.export);
    }
    const id =
      event.payload.id ??
      (await step.do("enqueue management trial", () =>
        enqueue(
          this.env,
          event.payload.issue!,
          true,
          event.payload.requestId ?? "",
        ),
      ));
    for (let n = 0; n < 100; n++) {
      let done = false;
      try {
        done = await step.do(`advance ${n}`, () => tick(this.env, id));
      } catch {
        console.error(
          JSON.stringify({ event: "issue_tick_deferred", job: id }),
        );
      }
      await step.do(`outbox ${n}`, () => deliver(this.env));
      if (done) return;
      await step.sleep(`wait ${n}`, "30 seconds");
    }
  }
}
export async function scheduled(env: IssueEnv) {
  const ledger = env.LEDGER.getByName("coordinator");
  await deliver(env);
  const queue = await ledger.queueState();
  if (queue.owner) {
    try {
      await tick(env, queue.owner);
    } catch {
      console.error(
        JSON.stringify({ event: "issue_cleanup_deferred", job: queue.owner }),
      );
    }
  }
  if (env.ENABLED !== "true") return;
  for (const job of await ledger.pending()) {
    try {
      await env.ISSUE.create({
        id: `${job.id}-${Math.floor(Date.now() / 300000)}`,
        params: { id: job.id },
      });
    } catch {
      /* same bucket instance already exists */
    }
  }
}
export default {
  fetch(request: Request, env: IssueEnv) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health")
      return Response.json({ ok: true, enabled: env.ENABLED === "true" });
    if (request.method === "POST" && url.pathname === "/webhook")
      return handleWebhook(request, env);
    return new Response("Not found", { status: 404 });
  },
  scheduled(
    _controller: ScheduledController,
    env: IssueEnv,
    ctx: ExecutionContext,
  ) {
    ctx.waitUntil(scheduled(env));
  },
};

export { linksIssue } from "./contracts";
