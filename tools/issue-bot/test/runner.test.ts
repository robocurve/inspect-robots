import { describe, expect, it, vi } from "vitest";
import { modelOutbound, validateRequest } from "../src/sandbox";
import type { StageRequest } from "../src/contracts";

function request(): StageRequest {
  return {
    id: "stage-1",
    jobId: "job-1",
    kind: "triage",
    base: "a".repeat(40),
    issue: {
      number: 401,
      title: "Failure",
      body: "Reproduce",
      author: "jeqcho",
      authorId: 42904912,
      state: "open",
      revision: "d".repeat(64),
      base: "a".repeat(40),
    },
    context: "{}",
    plan: "",
    feedback: "",
    files: [],
    inputDigest: "d".repeat(64),
    token: "b".repeat(64),
    checkpointToken: "c".repeat(64),
    sandbox: "12345678-1234-1234-1234-123456789012",
  };
}

describe("isolated stage request", () => {
  it("accepts each supported role with distinct capabilities", () => {
    for (const kind of [
      "triage",
      "plan",
      "plan_review",
      "implement",
      "code_review",
    ] as const) {
      expect(validateRequest({ ...request(), kind }).kind).toBe(kind);
    }
  });
  it("rejects a mismatched pinned base", () => {
    expect(() =>
      validateRequest({ ...request(), base: "e".repeat(40) }),
    ).toThrow("invalid_stage_request");
  });
  it("does not let model capabilities double as checkpoint capabilities", () => {
    const value = request();
    value.checkpointToken = value.token;
    expect(() => validateRequest(value)).toThrow("invalid_stage_request");
  });
  it("rejects unsupported actions and unsafe files before starting a container", () => {
    expect(() =>
      validateRequest({
        ...request(),
        kind: "publish" as StageRequest["kind"],
      }),
    ).toThrow();
    expect(() =>
      validateRequest({
        ...request(),
        files: [
          {
            path: ".github/workflows/change.yml",
            content: "bad",
            mode: "100644",
          },
        ],
      }),
    ).toThrow("unsafe_artifact");
    expect(() =>
      validateRequest({ ...request(), context: "x".repeat(1_400_000) }),
    ).toThrow("invalid_stage_request");
  });
});

describe("fixed Container supervisor controller", () => {
  it("polling a stopped container does not start or replace it", async () => {
    const { IssueSandbox } = await import("../src/sandbox");
    const box = Object.create(IssueSandbox.prototype) as InstanceType<
      typeof IssueSandbox
    >;
    let attemptedConnection = false;
    Object.defineProperty(box, "ctx", {
      value: {
        storage: {
          get: async () => ({
            control: "a".repeat(64),
            digest: "b".repeat(64),
          }),
        },
        container: {
          running: false,
          getTcpPort: () => {
            attemptedConnection = true;
            throw new Error("must_not_connect");
          },
        },
      },
    });
    expect(await box.status()).toBeNull();
    expect(attemptedConnection).toBe(false);
  });

  it("polls only the authenticated fixed status route and validates output", async () => {
    const { IssueSandbox } = await import("../src/sandbox");
    const box = Object.create(IssueSandbox.prototype) as InstanceType<
      typeof IssueSandbox
    >;
    let request: Request | undefined;
    const output = {
      exitCode: 1,
      failure: "fixture",
      result: null,
      files: [],
      executions: [],
    };
    Object.defineProperty(box, "ctx", {
      value: {
        storage: {
          get: async () => ({
            control: "a".repeat(64),
            digest: "b".repeat(64),
          }),
        },
        container: {
          running: true,
          getTcpPort: (port: number) => {
            expect(port).toBe(8080);
            return {
              fetch: async (input: Request) => {
                request = input;
                return Response.json({ state: "complete", output });
              },
            };
          },
        },
      },
    });
    expect(JSON.parse((await box.status())!)).toEqual(output);
    expect(new URL(request!.url).pathname).toBe("/status");
    expect(request!.method).toBe("GET");
    expect(request!.headers.get("Authorization")).toBe(
      `Bearer ${"a".repeat(64)}`,
    );
  });

  it("does not offer the generic Container HTTP forwarder", async () => {
    const { IssueSandbox } = await import("../src/sandbox");
    const box = Object.create(IssueSandbox.prototype) as InstanceType<
      typeof IssueSandbox
    >;
    expect((await box.fetch()).status).toBe(404);
  });
});

describe("client-only gateway bearer capability", () => {
  const capability = "a".repeat(64);
  function gateway() {
    return {
      MODEL: {
        respond: vi.fn(async () => new Response("response")),
        deliverCheckpoint: vi.fn(async () => undefined),
        checkpoint: vi.fn(async () => undefined),
        completed: vi.fn(async () => false),
      },
    };
  }
  it("accepts only exact bearer capability on the fixed responses route", async () => {
    const env = gateway();
    const response = await modelOutbound(
      new Request("http://issue-model.local/responses", {
        method: "POST",
        headers: { Authorization: `Bearer ${capability}` },
        body: "{}",
      }),
      env,
    );
    expect(response.status).toBe(200);
    expect(env.MODEL.respond).toHaveBeenCalledWith(capability, "{}");
  });
  it.each([
    "",
    "Bearer wrong",
    `Bearer ${capability} extra`,
    `Basic ${capability}`,
    `Bearer ${capability}, Bearer ${capability}`,
  ])("rejects malformed authentication %s", async (authorization) => {
    const env = gateway();
    const response = await modelOutbound(
      new Request("http://issue-model.local/responses", {
        method: "POST",
        headers: { Authorization: authorization },
        body: "{}",
      }),
      env,
    );
    expect(response.status).toBe(403);
    expect(env.MODEL.respond).not.toHaveBeenCalled();
  });
  it("rejects the old capability-in-URL model route", async () => {
    const env = gateway();
    const response = await modelOutbound(
      new Request(`http://issue-model.local/${capability}/responses`, {
        method: "POST",
        headers: { Authorization: `Bearer ${capability}` },
        body: "{}",
      }),
      env,
    );
    expect(response.status).toBe(403);
    expect(env.MODEL.respond).not.toHaveBeenCalled();
  });
  it("retains the separate trusted checkpoint receipt route", async () => {
    const env = gateway();
    const response = await modelOutbound(
      new Request(`http://issue-model.local/${capability}/checkpoint`, {
        method: "POST",
        body: "{}",
      }),
      env,
    );
    expect(response.status).toBe(200);
    expect(env.MODEL.deliverCheckpoint).toHaveBeenCalledWith(capability, "{}");
    expect(env.MODEL.respond).not.toHaveBeenCalled();
  });
});
