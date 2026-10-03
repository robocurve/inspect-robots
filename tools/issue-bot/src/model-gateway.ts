import { WorkerEntrypoint } from "cloudflare:workers";
import OpenAI from "openai";
import { digest, MODEL } from "./contracts";

function localTools(tools: unknown[]): boolean {
  return tools.every((t) => {
    if (!t || typeof t !== "object") return false;
    const tool = t as { type?: string; tools?: unknown[] };
    return (
      ["function", "custom", "local_shell", "apply_patch"].includes(
        tool.type ?? "",
      ) ||
      (tool.type === "namespace" &&
        Array.isArray(tool.tools) &&
        localTools(tool.tools))
    );
  });
}
export class ModelGateway extends WorkerEntrypoint<IssueEnv> {
  async deliverCheckpoint(token: string, body: string) {
    await this.env.LEDGER.getByName("coordinator").checkpoint(token, body);
  }
  async checkpoint(token: string, body: string) {
    await this.deliverCheckpoint(token, body);
  }
  async completed(token: string) {
    return this.env.LEDGER.getByName("coordinator").completed(token);
  }
  async respond(token: string, body: string): Promise<Response> {
    const ledger = this.env.LEDGER.getByName("coordinator"),
      stage = await ledger.session(token);
    if (this.env.ENABLED !== "true" || !stage || (await ledger.billingHold()))
      return new Response("Session unavailable", { status: 403 });
    if (body.length > 3_000_000)
      return new Response("Too large", { status: 413 });
    let request;
    try {
      request = JSON.parse(body);
    } catch {
      return new Response("Invalid request", { status: 400 });
    }
    if (
      !Array.isArray(request.input) ||
      !Array.isArray(request.tools ?? []) ||
      !localTools(request.tools ?? []) ||
      request.previous_response_id ||
      request.input.some(
        (i: { type?: string; tools?: unknown[] }) =>
          i.type === "additional_tools" &&
          (!Array.isArray(i.tools) || !localTools(i.tools)),
      )
    )
      return new Response("Unsupported request", { status: 400 });
    const allowance = await ledger.remaining(stage.request.issue.number);
    // An allowlist prevents caller-controlled provider storage, service tier, model or remote tools.
    const params = {
      model: MODEL,
      instructions: request.instructions,
      input: [
        ...request.input,
        {
          role: "developer",
          content: `This issue workflow has $${(allowance / 1e6).toFixed(2)} remaining. Preserve budget for independent review and report limitations honestly.`,
        },
      ],
      tools: request.tools ?? [],
      text: request.text,
      reasoning: { effort: "high" as const },
      stream: true,
      store: false,
      background: false,
      service_tier: "default",
    };
    const client = new OpenAI({
      apiKey: this.env.OPENAI_API_KEY,
      maxRetries: 0,
      timeout: 60000,
    });
    let count;
    try {
      count = await client.responses.inputTokens.count({
        model: MODEL,
        instructions: params.instructions,
        input: params.input,
        tools: params.tools,
        text: params.text,
        reasoning: params.reasoning,
      });
    } catch {
      return new Response("Token counting failed", { status: 502 });
    }
    const input = (count.input_tokens + 128) * 13,
      maxOutput = Math.min(16000, Math.floor((allowance - input) / 50));
    if (count.input_tokens > 200000 || maxOutput < 2000) {
      await ledger.failSession(token, "budget_or_context_exhausted");
      return new Response("Budget or context exhausted", { status: 429 });
    }
    const charge = await digest(token + body);
    if (!(await ledger.reserve(token, charge, input + maxOutput * 50)))
      return new Response("Budget or replay guard", { status: 409 });
    let response: Response;
    try {
      response = await fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        redirect: "manual",
        headers: {
          Authorization: `Bearer ${this.env.OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ...params, max_output_tokens: maxOutput }),
        signal: AbortSignal.timeout(15 * 60000),
      });
    } catch {
      return new Response("Upstream failed; reservation retained", {
        status: 502,
      });
    }
    if (!response.ok || !response.body)
      return new Response("Upstream failed; reservation retained", {
        status: 502,
      });
    let pending = "";
    const decoder = new TextDecoder();
    const transform = new TransformStream<Uint8Array, Uint8Array>({
      async transform(chunk, controller) {
        pending += decoder.decode(chunk, { stream: true });
        if (pending.length > 3_000_000)
          throw new Error("model_event_too_large");
        let newline;
        while ((newline = pending.indexOf("\n")) >= 0) {
          const line = pending.slice(0, newline).trimEnd();
          pending = pending.slice(newline + 1);
          if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
          const event = JSON.parse(line.slice(6));
          if (
            [
              "response.completed",
              "response.incomplete",
              "response.failed",
            ].includes(event.type) &&
            event.response?.usage
          ) {
            const u = event.response.usage;
            if (
              !Number.isSafeInteger(u.input_tokens) ||
              u.input_tokens < 0 ||
              !Number.isSafeInteger(u.output_tokens) ||
              u.output_tokens < 0
            )
              throw new Error("invalid_usage");
            const c = u.input_tokens_details?.cached_tokens;
            const cached =
              Number.isSafeInteger(c) && c >= 0 && c <= u.input_tokens ? c : 0;
            await ledger.settle(
              charge,
              (u.input_tokens - cached) * 13 + cached + u.output_tokens * 50,
            );
          }
        }
        controller.enqueue(chunk);
      },
    });
    return new Response(response.body.pipeThrough(transform), {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }
}
