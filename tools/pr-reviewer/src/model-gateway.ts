import { WorkerEntrypoint } from 'cloudflare:workers';
import OpenAI from 'openai';
import { digest, MODEL } from './common';
import { retryLedger } from './ledger-connection';

function localTools(tools: any[]): boolean {
  return tools.every(tool => ['function', 'custom', 'local_shell', 'apply_patch'].includes(tool.type) || (tool.type === 'namespace' && Array.isArray(tool.tools) && localTools(tool.tools)));
}

// Private binding. Codex receives a short-lived budget capability, never an API key.
export type GatewayEnvironment = Pick<ReviewerEnv, 'OPENAI_API_KEY'> & { LEDGER: Pick<ReviewerEnv['LEDGER'], 'getByName'> };
export class ModelGateway extends WorkerEntrypoint<GatewayEnvironment> {
  async deliverCheckpoint(receipt: string, body: string): Promise<void> {
    await retryLedger(this.env, ledger => ledger.deliverCheckpoint(receipt, body));
  }
  async checkpoint(token: string, body: string): Promise<void> {
    await retryLedger(this.env, ledger => ledger.checkpoint(token, body));
  }
  async completed(token: string): Promise<boolean> {
    // Fetch a fresh stub for each RPC, including retries after Workflow sleeps.
    const ledger = () => this.env.LEDGER.getByName('budget');
    const job = await ledger().session(token);
    if (!job) throw new Error('invalid_review_session');
    return await ledger().runOutput(job.id) !== null;
  }
  async respond(token: string, body: string): Promise<Response> {
    const ledger = () => this.env.LEDGER.getByName('budget');
    if (await ledger().reviewQueuePaused()) return new Response('Review service paused', { status: 503 });
    const job = await ledger().session(token);
    if (!job || await ledger().runOutput(job.id) !== null) return new Response('Review session unavailable', { status: 403 });
    const stop = async (reason: string, message: string, status: number) => {
      await ledger().sessionFailure(token, reason);
      return new Response(message, { status });
    };
    if (await ledger().billingHold()) {
      console.error(JSON.stringify({ event: 'model_stopped', job: job.id, reason: 'billing_hold' }));
      return stop('billing_hold', 'Review billing hold', 403);
    }
    if (body.length > 3_000_000) return new Response('Context too large', { status: 413 });
    let request;
    try { request = JSON.parse(body); } catch { return new Response('Invalid request', { status: 400 }); }
    if (!Array.isArray(request.input) || !Array.isArray(request.tools ?? []) || !localTools(request.tools ?? []) || request.input.some((item: any) => item.type === 'additional_tools' && (!Array.isArray(item.tools) || !localTools(item.tools))) || request.previous_response_id) return new Response('Unsupported review request', { status: 400 });
    const client = new OpenAI({ apiKey: this.env.OPENAI_API_KEY, maxRetries: 0, timeout: 60000 });
    const params = { ...request, model: MODEL, reasoning: { ...request.reasoning, effort: 'high' }, service_tier: 'default', stream: true, background: false, store: false };
    delete params.max_output_tokens;
    const allowance = await ledger().remaining(`${job.pr}-${job.head}`, job.pr);
    params.input = [...params.input, { role: 'developer', content: `The review has $${(allowance / 1_000_000).toFixed(2)} remaining, including this call. Preserve enough budget to write the final review; report REQUIRE_REVIEWER for material evidence gaps instead of claiming a complete review.` }];
    let count;
    try { count = await client.responses.inputTokens.count({ model: MODEL, instructions: params.instructions, input: params.input, tools: params.tools, text: params.text, reasoning: params.reasoning }); }
    catch (error) {
      console.error(JSON.stringify({ event: 'model_stopped', job: job.id, reason: 'token_count_failed', status: error instanceof OpenAI.APIError ? error.status : null }));
      return new Response('Review token counting failed', { status: 502 });
    }
    if (count.input_tokens > 200_000) return stop('context_too_large', 'Review context limit reached', 413);
    const inputReservation = (count.input_tokens + 128) * 13;
    const maxOutput = Math.min(16000, Math.floor((allowance - inputReservation) / 50));
    console.log(JSON.stringify({ event: 'model_allowance', job: job.id, inputTokens: count.input_tokens, remainingMicros: allowance, maxOutput }));
    if (maxOutput < 2000) return stop('budget_exhausted', 'Review budget exhausted', 429);
    if (allowance < inputReservation + 300_000) {
      params.tool_choice = 'none';
      params.input[params.input.length - 1].content = 'Final review turn: budget cannot safely cover more investigation. Return review JSON now. Name exact unchecked files or behavior, the budget limit, and the next check. Use REQUIRE_REVIEWER with COMPLETE_REVIEW for unfinished inspection, empty decision_needed, and specific limitations. ESCALATE only for an actual product decision. Do not invent findings.';
    }
    const charge = `${job.id}-codex-${await digest(token + body)}`;
    if (!await ledger().reserve(charge, `${job.pr}-${job.head}`, job.pr, inputReservation + maxOutput * 50)) return new Response('Review budget or duplicate request guard', { status: 409 });
    const response = await fetch(new Request('https://api.openai.com/v1/responses', {
      method: 'POST', redirect: 'manual', headers: { Authorization: `Bearer ${this.env.OPENAI_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...params, max_output_tokens: maxOutput }), signal: AbortSignal.timeout(15 * 60_000),
    }));
    if (!response.ok || !response.body) {
      console.error(JSON.stringify({ event: 'model_stopped', job: job.id, reason: 'model_request_failed', status: response.status }));
      return new Response('Model request did not complete; reservation retained', { status: 502 });
    }
    const env = this.env;
    let pending = '';
    const decoder = new TextDecoder();
    const stream = new TransformStream<Uint8Array, Uint8Array>({
      async transform(chunk, controller) {
        pending += decoder.decode(chunk, { stream: true });
        if (pending.length > 3_000_000) throw new Error('model_event_too_large');
        let newline;
        while ((newline = pending.indexOf('\n')) >= 0) {
          const line = pending.slice(0, newline).trimEnd(); pending = pending.slice(newline + 1);
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
          const event = JSON.parse(line.slice(6));
          if (['response.completed', 'response.incomplete', 'response.failed'].includes(event.type) && event.response?.usage) {
            const usage = event.response.usage;
            const reportedCached = usage.input_tokens_details?.cached_tokens;
            const cached = Number.isSafeInteger(reportedCached) && reportedCached >= 0 && reportedCached <= usage.input_tokens ? reportedCached : 0;
            // Keep the cache-write ceiling for other input, but credit confirmed
            // cache reads at the published $1/M rate. Never assume a cache hit.
            const amount = (usage.input_tokens - cached) * 13 + cached + usage.output_tokens * 50;
            const settled = await retryLedger(env, ledger => ledger.settle(charge, amount));
            console.log(JSON.stringify({ event: 'model_usage', job: job.id, inputTokens: usage.input_tokens, cachedTokens: cached, outputTokens: usage.output_tokens, chargedMicros: amount, settled }));
          }
        }
        controller.enqueue(chunk);
      },
    });
    return new Response(response.body.pipeThrough(stream), { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' } });
  }
}
