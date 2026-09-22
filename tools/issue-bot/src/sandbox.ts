import { WorkerEntrypoint } from "cloudflare:workers";
import { Container, ContainerProxy } from "@cloudflare/containers";
import { Buffer } from "node:buffer";
import { zodTextFormat } from "openai/helpers/zod";
import {
  boundedText,
  digest,
  HASH,
  SHA,
  StageKind,
  StageResult,
  StageOutput,
  validateFiles,
  type StageRequest,
} from "./contracts";
export { ContainerProxy };

export interface IssueRunnerEnv {
  SANDBOX: DurableObjectNamespace<IssueSandbox>;
  MODEL: {
    respond(token: string, body: string): Promise<Response>;
    deliverCheckpoint(token: string, body: string): Promise<unknown>;
    checkpoint(token: string, body: string): Promise<unknown>;
    completed(token: string): Promise<boolean>;
  };
}

export function validateRequest(input: StageRequest): StageRequest {
  if (
    !SHA.test(input.base) ||
    !HASH.test(input.token) ||
    !HASH.test(input.checkpointToken) ||
    input.token === input.checkpointToken ||
    !HASH.test(input.inputDigest) ||
    !/^[a-f0-9-]{36}$/.test(input.sandbox) ||
    input.base !== input.issue.base ||
    !Number.isSafeInteger(input.issue.number) ||
    input.issue.number < 1 ||
    JSON.stringify(input).length > 1_400_000
  )
    throw new Error("invalid_stage_request");
  StageKind.parse(input.kind);
  validateFiles(input.files);
  return input;
}

interface LaunchClaim {
  control: string;
  digest: string;
}
export class IssueSandbox extends Container<IssueRunnerEnv> {
  defaultPort = 8080;
  sleepAfter = "45m";
  enableInternet = false;
  allowedHosts = ["issue-model.local"];
  private launching?: Promise<void>;

  // Deliberately do not expose Container's default HTTP forwarder.
  override async fetch(): Promise<Response> {
    return new Response("Not found", { status: 404 });
  }
  async launch(request: string): Promise<void> {
    if (!this.launching)
      this.launching = this.launchOnce(request).finally(() => {
        this.launching = undefined;
      });
    await this.launching;
  }
  private async launchOnce(request: string): Promise<void> {
    const input = validateRequest(JSON.parse(request));
    const requestDigest = await digest(request);
    const existing = await this.ctx.storage.get<LaunchClaim>("launch-claimed");
    if (existing) {
      if (existing.digest !== requestDigest)
        throw new Error("stage_request_changed");
      return; // Ambiguous acknowledgement never replays startup or paid execution.
    }
    const source = await archive(input.base);
    const control = Array.from(
      crypto.getRandomValues(new Uint8Array(32)),
      (b) => b.toString(16).padStart(2, "0"),
    ).join("");
    await this.ctx.storage.put("launch-claimed", {
      control,
      digest: requestDigest,
    });
    await this.schedule(42 * 60, "expireStage");
    await this.startAndWaitForPorts({
      ports: [8080],
      startOptions: {
        envVars: { ISSUE_CONTROL_SECRET: control },
        enableInternet: false,
        entrypoint: [
          "/opt/issue-env/bin/python",
          "-I",
          "/opt/issue-supervisor.py",
        ],
      },
      cancellationOptions: {
        instanceGetTimeoutMS: 60_000,
        portReadyTimeoutMS: 60_000,
      },
    });
    const response = await this.controlFetch(
      "/start",
      control,
      JSON.stringify({
        request: {
          ...input,
          schema: zodTextFormat(StageResult, "stage_result").schema,
        },
        archive: source,
      }),
    );
    if (!response.ok) throw new Error("stage_launch_uncertain");
  }
  private async controlFetch(
    path: "/start" | "/status",
    secret: string,
    body?: string,
  ): Promise<Response> {
    // containerFetch auto-starts stopped containers. Direct TCP-port fetch cannot
    // silently replace a crashed execution or revive a deliberately cleaned one.
    if (!this.ctx.container?.running)
      throw new Error("stage_container_stopped");
    return this.ctx.container.getTcpPort(8080).fetch(
      new Request(`http://container${path}`, {
        method: body === undefined ? "GET" : "POST",
        headers: {
          Authorization: `Bearer ${secret}`,
          "Content-Type": "application/json",
        },
        body,
        signal: AbortSignal.timeout(30_000),
      }),
    );
  }
  async status(): Promise<string | null> {
    const claim = await this.ctx.storage.get<LaunchClaim>("launch-claimed");
    if (!claim || !this.ctx.container?.running) return null;
    const response = await this.controlFetch("/status", claim.control);
    if (!response.ok) throw new Error("stage_status_unavailable");
    const status = JSON.parse(await boundedText(response, 1_600_000));
    if (status.state !== "complete") return null;
    return JSON.stringify(StageOutput.parse(status.output));
  }
  async expireStage(): Promise<void> {
    await this.destroy();
  }
}
/** Fixed model route uses a client-only bearer capability, never a URL token. */
export async function modelOutbound(
  request: Request,
  env: Pick<IssueRunnerEnv, "MODEL">,
): Promise<Response> {
  const path = new URL(request.url).pathname;
  if (request.method !== "POST")
    return new Response("Not allowed", { status: 403 });
  const checkpoint = /^\/([a-f0-9]{64})\/checkpoint$/.exec(path);
  if (checkpoint) {
    await env.MODEL.deliverCheckpoint(
      checkpoint[1],
      await boundedText(request, 1_500_000),
    );
    return new Response("Saved");
  }
  const authorization = /^Bearer ([a-f0-9]{64})$/.exec(
    request.headers.get("Authorization") ?? "",
  );
  if (path !== "/responses" || !authorization)
    return new Response("Not allowed", { status: 403 });
  return env.MODEL.respond(
    authorization[1],
    await boundedText(request, 3_000_000),
  );
}
IssueSandbox.outboundByHost = { "issue-model.local": modelOutbound };

async function archive(sha: string): Promise<string> {
  const response = await fetch(
    new Request(
      `https://codeload.github.com/robocurve/inspect-robots/tar.gz/${sha}`,
      { redirect: "manual", signal: AbortSignal.timeout(30000) },
    ),
  );
  if (!response.ok || !response.body) throw new Error("archive_unavailable");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    size += part.value.length;
    if (size > 10_000_000) {
      await reader.cancel();
      throw new Error("archive_too_large");
    }
    chunks.push(part.value);
  }
  return Buffer.concat(chunks).toString("base64");
}

export class CodeRunner extends WorkerEntrypoint<IssueRunnerEnv> {
  private box(id: string) {
    if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error("invalid_sandbox");
    return this.env.SANDBOX.getByName(id);
  }
  async start(request: StageRequest): Promise<void> {
    validateRequest(request);
    if (await this.env.MODEL.completed(request.token)) return;
    await this.box(request.sandbox).launch(JSON.stringify(request));
  }
  async poll(sandbox: string, token: string): Promise<boolean> {
    if (!HASH.test(token)) throw new Error("invalid_stage_token");
    if (await this.env.MODEL.completed(token)) return true;
    const output = await this.box(sandbox).status();
    if (output === null) return false;
    await this.env.MODEL.checkpoint(token, output);
    return true;
  }
  async cleanup(sandbox: string): Promise<void> {
    await this.box(sandbox).destroy();
  }
}
export default {
  fetch() {
    return new Response("Not found", { status: 404 });
  },
};
