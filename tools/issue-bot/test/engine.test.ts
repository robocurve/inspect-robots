import { validateRequest } from "../src/sandbox";
import { env, runInDurableObject } from "cloudflare:test";
import { describe, it, expect, vi } from "vitest";
import {
  artifactDigest,
  semanticRevision,
  type IssueSnapshot,
  type StageOutput,
  type StageResult,
} from "../src/contracts";
import {
  linksIssue,
  verifySignature,
  handleWebhook,
  tick,
  deliver,
} from "../src/worker";
import { isValidationCommand, type IssueLedger } from "../src/ledger";
const testEnv = env as unknown as IssueEnv;
const snapshot: IssueSnapshot = {
  number: 401,
  title: "A serious defect",
  body: "reproduction",
  author: "jeqcho",
  authorId: 42904912,
  state: "open",
  revision: "a".repeat(64),
  base: "b".repeat(40),
};
function ledger() {
  return testEnv.LEDGER.getByName(crypto.randomUUID());
}
function result(
  status: StageResult["status"],
  extra: Partial<StageResult> = {},
): StageResult {
  return {
    status,
    serious: true,
    summary: "Observed defect",
    evidence: ["reproduction"],
    plan: "Implement regression test then fix",
    findings: [],
    checks: ["pytest tests/test_bug.py"],
    limitations: [],
    ...extra,
  };
}
function output(
  status: StageResult["status"],
  extra: Partial<StageOutput> = {},
): StageOutput {
  return {
    exitCode: 0,
    failure: null,
    result: result(status),
    files: [],
    executions: [{ command: "pytest tests/test_bug.py", exitCode: 0 }],
    ...extra,
  };
}
async function run(
  l: DurableObjectStub<IssueLedger>,
  id: string,
  out: StageOutput,
) {
  const s = await l.prepare(id);
  expect(s).not.toBeNull();
  expect(validateRequest(s!.request)).toEqual(s!.request);
  await l.claimLaunch(s!.request.id);
  await l.checkpoint(s!.request.checkpointToken, JSON.stringify(out));
  await l.cleaned(s!.request.id);
  await l.consume(id);
  return s!;
}
describe("durable issue coordinator", () => {
  it("deduplicates revisions and serializes FIFO before any launch", async () => {
    const l = ledger();
    const a = await l.register(snapshot, "", false),
      same = await l.register(snapshot, "", false);
    expect(same).toBe(a);
    const b = await l.register({ ...snapshot, number: 402 }, "", false);
    const claims = await Promise.all([l.claim(a), l.claim(b)]);
    expect(claims).toEqual([true, false]);
    const s = await l.prepare(a);
    expect(await l.claimLaunch(s!.request.id)).toBe(true);
    expect(await l.claimLaunch(s!.request.id)).toBe(false);
    expect(await l.release(a)).toBe(false);
    await l.close(s!.request.id);
    await l.cleaned(s!.request.id);
    await l.hold(a, "test_hold");
    expect(await l.release(a)).toBe(true);
    expect(await l.claim(b)).toBe(true);
  });
  it("approval is bound to exact plan and artifact and requires real successful executions", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    await run(l, id, output("CONFIRMED"));
    await run(l, id, output("PLAN"));
    await run(l, id, output("APPROVE"));
    const files = [
      { path: "src/test.py", content: "fixed\n", mode: "100644" as const },
    ];
    await run(l, id, output("IMPLEMENTED", { files }));
    const review = await run(l, id, output("APPROVE"));
    expect(review.request.files).toEqual(files);
    const fix = await l.fix(id);
    expect(fix.approvedDigest).toBe(await artifactDigest(files));
    expect(fix.checks).toEqual(["pytest tests/test_bug.py: exit 0"]);
  });
  it("never approves a review with no executed checks despite prose claiming success", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    for (const status of ["CONFIRMED", "PLAN", "APPROVE"] as const)
      await run(l, id, output(status));
    await run(
      l,
      id,
      output("IMPLEMENTED", {
        files: [{ path: "src/test.py", content: "fixed", mode: "100644" }],
      }),
    );
    await run(l, id, output("APPROVE", { executions: [] }));
    expect((await l.job(id))?.state).toBe("held");
    expect(
      (await l.outbox()).some(
        (x) => x.publication.status === "REQUIRE_REVIEWER",
      ),
    ).toBe(true);
    expect(
      await runInDurableObject(l, async (obj) => {
        try {
          await obj.fix(id);
          return "";
        } catch (e) {
          return (e as Error).message;
        }
      }),
    ).toBe("artifact_approval_mismatch");
  });
  it("ignores observational exit codes but refuses any failed validation command", async () => {
    for (const failing of [false, true]) {
      const l = ledger(),
        id = await l.register(snapshot, "", false);
      await l.claim(id);
      for (const status of ["CONFIRMED", "PLAN", "APPROVE"] as const)
        await run(l, id, output(status));
      await run(
        l,
        id,
        output("IMPLEMENTED", {
          files: [{ path: "src/test.py", content: "fixed", mode: "100644" }],
        }),
      );
      await run(
        l,
        id,
        output("APPROVE", {
          executions: [
            { command: "pytest tests/test_bug.py", exitCode: 0 },
            {
              command:
                'pytest tests/base_regression.py; observed=$?; test "$observed" -eq 1',
              exitCode: 0,
            },
            {
              command: failing ? "pytest tests/test_other.py" : "rg absent src",
              exitCode: 1,
            },
          ],
        }),
      );
      expect((await l.job(id))?.state).toBe(failing ? "held" : "fix");
    }
  });
  it("existing fix and nonserious reports stay triage-only", async () => {
    for (const duplicate of [true, false]) {
      const l = ledger(),
        id = await l.register(snapshot, "", duplicate);
      await l.claim(id);
      await run(
        l,
        id,
        output("CONFIRMED", {
          result: result("CONFIRMED", { serious: duplicate }),
        }),
      );
      expect((await l.job(id))?.state).toBe("done");
      expect((await l.job(id))?.next).toBeNull();
      expect((await l.outbox())[0].publication.status).toBe(
        duplicate ? "FIX_PROPOSED" : "CONFIRMED",
      );
    }
  });
  it("deduplicates concurrent explicit retries of one issue", async () => {
    const l = ledger();
    const ids = await Promise.all([
      l.register(snapshot, "", false, "retry-a"),
      l.register(snapshot, "", false, "retry-b"),
    ]);
    expect(new Set(ids).size).toBe(1);
    expect(await l.pending()).toHaveLength(1);
  });
  it("caps reviewer revision loops", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    await run(l, id, output("CONFIRMED"));
    for (let i = 0; i < 3; i++) {
      await run(l, id, output("PLAN"));
      await run(
        l,
        id,
        output("REQUEST_CHANGES", {
          result: result("REQUEST_CHANGES", { findings: ["fix this"] }),
        }),
      );
    }
    expect((await l.job(id))?.state).toBe("held");
  });
  it("scopes checkpoint receipts separately from model capabilities and saves first terminal result", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    const s = (await l.prepare(id))!;
    expect(
      await runInDurableObject(l, async (obj) => {
        try {
          await obj.checkpoint(
            s.request.token,
            JSON.stringify(output("CONFIRMED")),
          );
          return "";
        } catch (e) {
          return (e as Error).message;
        }
      }),
    ).toBe("invalid_receipt");
    await l.checkpoint(
      s.request.checkpointToken,
      JSON.stringify(output("NEEDS_INFO")),
    );
    await l.checkpoint(
      s.request.checkpointToken,
      JSON.stringify(output("CONFIRMED")),
    );
    expect((await l.stage(s.request.id))?.output?.result?.status).toBe(
      "NEEDS_INFO",
    );
    expect(await l.session(s.request.token)).toBeNull();
  });
  it("atomically reserves lifetime budget across concurrent requests and semantic revisions", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    const s = (await l.prepare(id))!;
    const reservations = await Promise.all([
      l.reserve(s.request.token, "a", 11000000),
      l.reserve(s.request.token, "b", 11000000),
    ]);
    expect(reservations.filter(Boolean)).toHaveLength(1);
    expect(await l.costs(401)).toBe(11000000);
    await l.settle("a", 1000000);
    expect(await l.costs(401)).toBe(1000000);
    expect(await l.reserve(s.request.token, "a", 1000000)).toBe(false);
  });
  it("retains ambiguous spend and holds globally on an exceeded reservation", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    const s = (await l.prepare(id))!;
    await l.reserve(s.request.token, "ambiguous", 1000000);
    expect(await l.costs(401)).toBe(1000000);
    await l.settle("ambiguous", 1000001);
    expect(await l.billingHold()).toBe(true);
  });
  it("retries held failure publications durably", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.hold(id, "cleanup_failed");
    const publish = vi
      .fn()
      .mockRejectedValueOnce(new Error("github_http_503"))
      .mockResolvedValue(true);
    const e = {
      LEDGER: { getByName: () => l },
      PUBLISHER: { publish },
    } as unknown as IssueEnv;
    await deliver(e);
    expect(await l.outbox()).toHaveLength(1);
    await deliver(e);
    expect(await l.outbox()).toHaveLength(0);
    expect(publish).toHaveBeenCalledTimes(2);
  });
  it("does not retry an ambiguously acknowledged start", async () => {
    const l = ledger();
    const issue = { ...snapshot, revision: await semanticRevision(snapshot) };
    const id = await l.register(issue, "", false);
    const start = vi.fn().mockRejectedValue(new Error("connection_lost"));
    const poll = vi.fn().mockResolvedValue(false);
    const e = {
      ...testEnv,
      LEDGER: { getByName: () => l },
      PUBLISHER: {
        read: vi.fn(async (path: string) =>
          JSON.stringify(
            path === "/commits/main"
              ? { sha: issue.base }
              : { ...issue, user: { id: issue.authorId, login: issue.author } },
          ),
        ),
      },
      RUNNER: { start, poll, cleanup: vi.fn() },
    } as unknown as IssueEnv;
    await tick(e, id);
    await tick(e, id);
    expect(start).toHaveBeenCalledTimes(1);
    expect(poll).toHaveBeenCalledTimes(1);
  });
});
describe("queue base freshness", () => {
  async function fixture() {
    const l = ledger();
    const issue = { ...snapshot, revision: await semanticRevision(snapshot) };
    const id = await l.register(issue, "", false);
    const current = { ...issue };
    const runner = { start: vi.fn(), poll: vi.fn(), cleanup: vi.fn() };
    const e = {
      ...testEnv,
      LEDGER: { getByName: () => l },
      PUBLISHER: {
        read: async (path: string) =>
          JSON.stringify(
            path === "/commits/main"
              ? { sha: current.base }
              : {
                  ...current,
                  user: { id: current.authorId, login: current.author },
                },
          ),
      },
      RUNNER: runner,
    } as unknown as IssueEnv;
    return { l, issue, id, current, runner, e };
  }
  it("pins the latest base only after FIFO admission and preserves charges", async () => {
    const { l, issue, id, current, runner, e } = await fixture();
    const second = await l.register({ ...issue, number: 402 }, "", false);
    current.base = "c".repeat(40);
    expect(await l.refreshUnstarted(id, current)).toBe(false);
    expect(await tick(e, second)).toBe(false);
    expect(runner.start).not.toHaveBeenCalled();
    await tick(e, id);
    const job = (await l.job(id))!;
    expect(job.issue.base).toBe(current.base);
    expect((await l.job(second))?.issue.base).toBe(issue.base);
    expect(runner.start.mock.calls[0][0].base).toBe(current.base);
    expect(await l.costs(issue.number)).toBe(100000);
    expect(
      await l.refreshUnstarted(id, { ...current, base: "d".repeat(40) }),
    ).toBe(false);
    expect((await l.job(id))?.issue.base).toBe(current.base);
  });
  it("lets active triage finish on its immutable base after a merge", async () => {
    const { l, issue, id, current, runner, e } = await fixture();
    await tick(e, id);
    const stage = (await l.stage((await l.job(id))!.stage!))!;
    current.base = "c".repeat(40);
    await tick(e, id);
    expect(runner.poll).toHaveBeenCalledOnce();
    expect(runner.cleanup).not.toHaveBeenCalled();
    await l.checkpoint(
      stage.request.checkpointToken,
      JSON.stringify(output("NEEDS_INFO")),
    );
    expect(await tick(e, id)).toBe(true);
    expect((await l.job(id))?.state).toBe("done");
    expect((await l.outbox())[0].publication.issue.base).toBe(issue.base);
    expect(runner.start).toHaveBeenCalledOnce();
  });
  it("cannot plan a serious fix using triage from an outdated base", async () => {
    const { l, id, current, runner, e } = await fixture();
    await tick(e, id);
    const stage = (await l.stage((await l.job(id))!.stage!))!;
    current.base = "c".repeat(40);
    await l.checkpoint(
      stage.request.checkpointToken,
      JSON.stringify(output("CONFIRMED")),
    );
    await tick(e, id);
    expect((await l.job(id))?.next).toBe("plan");
    await tick(e, id);
    expect((await l.job(id))?.state).toBe("held");
    expect(runner.start).toHaveBeenCalledOnce();
    expect((await l.outbox()).map((x) => x.publication.status)).toContain(
      "REQUIRE_REVIEWER",
    );
  });
  it.each([false, true])(
    "still holds edited issues (already started: %s)",
    async (started) => {
      const { l, id, current, runner, e } = await fixture();
      if (started) await tick(e, id);
      current.title = "Changed reproduction";
      current.base = "c".repeat(40);
      await tick(e, id);
      expect((await l.job(id))?.state).toBe("held");
      expect(runner.start).toHaveBeenCalledTimes(started ? 1 : 0);
      expect(runner.cleanup).toHaveBeenCalledTimes(started ? 1 : 0);
      expect((await l.queueState()).owner).toBeNull();
    },
  );
});
describe("actionable holds", () => {
  it("retains the latest reviewer findings and limitations in the durable notice", async () => {
    const l = ledger(),
      id = await l.register(snapshot, "", false);
    await l.claim(id);
    await run(l, id, output("CONFIRMED"));
    await run(l, id, output("PLAN"));
    await run(
      l,
      id,
      output("REQUIRE_REVIEWER", {
        result: result("REQUIRE_REVIEWER", {
          findings: ["Plan does not preserve the actuator safety invariant"],
          limitations: ["Hardware fixture missing; run the safety simulation"],
          checks: ["Safety simulation remains unchecked"],
        }),
      }),
    );
    const notice = (await l.outbox()).find(
      (x) => x.publication.status === "REQUIRE_REVIEWER",
    )!;
    expect(notice.publication.details.join("\n")).toContain(
      "actuator safety invariant",
    );
    expect(notice.publication.details.join("\n")).toContain(
      "Hardware fixture missing",
    );
  });
});
describe("signed intake", () => {
  it("detects existing fixes using both closing-reference formats without number-prefix matches", () => {
    expect(linksIssue("Fixes #401", 401)).toBe(true);
    expect(
      linksIssue(
        "Resolves https://github.com/robocurve/inspect-robots/issues/401",
        401,
      ),
    ).toBe(true);
    expect(linksIssue("Fixes #4010", 401)).toBe(false);
  });
  it("rejects forged signatures without touching external services", async () => {
    expect(
      await verifySignature("{}", "sha256=" + "0".repeat(64), "secret"),
    ).toBe(false);
    const r = await handleWebhook(
      new Request("https://test/webhook", { method: "POST", body: "{}" }),
      testEnv,
    );
    expect(r.status).toBe(401);
  });
  it("bot comments and updated_at do not change semantic issue revision", async () => {
    expect(
      await semanticRevision({
        ...snapshot,
        updated_at: "new",
      } as IssueSnapshot),
    ).toBe(await semanticRevision(snapshot));
    expect(await semanticRevision({ ...snapshot, body: "changed" })).not.toBe(
      await semanticRevision(snapshot),
    );
  });
});

describe("native validation command classification", () => {
  it.each([
    "/bin/bash -c 'pytest tests/test_bug.py -q'",
    "/bin/bash -lc 'cd /workspace/issue/candidate\nuv run pytest -q'",
    "cd /workspace/issue/candidate\npytest tests/test_bug.py",
    "/bin/bash --noprofile --norc -ec 'npm run typecheck'",
    "PYTHONPATH=src /opt/issue-env/bin/python -m pytest -q",
    'pytest tests/base.py; observed=$?; test "$observed" -eq 1',
    "pytest -q > /tmp/test.log 2>&1",
  ])("recognizes the actual executable in %s", (command) =>
    expect(isValidationCommand(command)).toBe(true),
  );

  it.each([
    "echo 'pytest tests/test_bug.py'",
    "printf '%s\\n' 'pytest; npm test'",
    "/bin/bash -c 'echo pytest'",
    "echo 'log\npytest test_bug.py'",
    "cat <<EOF\npytest test_bug.py\nEOF",
    "echo $(pytest)",
    "true",
    "rg pytest tests",
  ])("does not count printed or embedded prose in %s", (command) =>
    expect(isValidationCommand(command)).toBe(false),
  );

  it("requires successful wrapped validation results before approval", async () => {
    for (const exitCode of [0, 1]) {
      const l = ledger(),
        id = await l.register(snapshot, "", false);
      await l.claim(id);
      for (const status of ["CONFIRMED", "PLAN", "APPROVE"] as const)
        await run(l, id, output(status));
      await run(
        l,
        id,
        output("IMPLEMENTED", {
          files: [{ path: "src/test.py", content: "fixed", mode: "100644" }],
        }),
      );
      await run(
        l,
        id,
        output("APPROVE", {
          executions: [
            {
              command:
                "/bin/bash -lc 'cd /workspace/issue/candidate\npytest tests/test_bug.py'",
              exitCode,
            },
          ],
        }),
      );
      expect((await l.job(id))?.state).toBe(exitCode === 0 ? "fix" : "held");
    }
  });
});
