import { env, createExecutionContext } from 'cloudflare:test';
import { describe, expect, it, vi } from 'vitest';
import { ModelGateway } from '../src/model-gateway';
import type { Job } from '../src/common';

const job: Job = { id: 'codex-gateway', pr: 99, head: 'a'.repeat(40), base: 'b'.repeat(40), scope: '', status: 'running', result: null, notified: 0, created: 1 };
const request = JSON.stringify({ model: 'untrusted-model', input: [{ role: 'user', content: 'review' }], tools: [{ type: 'function', name: 'shell', parameters: { type: 'object', properties: {} } }], reasoning: { effort: 'low' } });
async function setup() {
  const ledger = env.LEDGER.getByName(crypto.randomUUID());
  await ledger.register(job);
  const token = await ledger.openSession(job.id);
  const gateway = new ModelGateway(createExecutionContext(), { LEDGER: { getByName: () => ledger }, OPENAI_API_KEY: 'sk-test-only' });
  return { ledger, token, gateway };
}
describe('Codex private budget gateway', () => {
  it('refuses model calls while the persistent safety pause is active', async () => {
    const s = await setup(); const send = vi.fn(); vi.stubGlobal('fetch', send);
    await s.ledger.pauseReviewQueue(true);
    expect((await s.gateway.respond(s.token, request)).status).toBe(503);
    expect(send).not.toHaveBeenCalled();
    expect((await s.ledger.costs(job.id)).modelCalls).toBe(0);
  });
  it('pins Astra/high, settles streamed usage and refuses duplicate submissions', async () => {
    const s = await setup(); let creates = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => {
      const r = new Request(url, init);
      if (r.url.endsWith('/input_tokens')) return Response.json({ input_tokens: 100 });
      creates++;
      expect(await r.json()).toMatchObject({ model: 'gpt-6-astra', reasoning: { effort: 'high' }, service_tier: 'default', max_output_tokens: 16000, stream: true });
      return new Response('data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":10}}}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
    }));
    expect(await (await s.gateway.respond(s.token, request)).text()).toContain('response.completed');
    expect(await s.ledger.remaining(`99-${job.head}`, 99)).toBe(5_000_000 - 1800);
    expect((await s.gateway.respond(s.token, request)).status).toBe(409);
    expect(creates).toBe(1);
  });
  it('denies expired capabilities and paid provider tools before inference', async () => {
    const s = await setup(); const send = vi.fn(); vi.stubGlobal('fetch', send);
    expect((await s.gateway.respond(s.token, JSON.stringify({ input: [], tools: [{ type: 'web_search' }] }))).status).toBe(400);
    await s.ledger.closeSession(s.token);
    expect((await s.gateway.respond(s.token, request)).status).toBe(403);
    expect(send).not.toHaveBeenCalled();
  });
  it('retains reservations when a stream never reports usage', async () => {
    const s = await setup();
    vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => new Request(url, init).url.endsWith('/input_tokens') ? Response.json({ input_tokens: 100 }) : new Response('data: {"type":"response.created"}\n\n')));
    await (await s.gateway.respond(s.token, request)).text();
    expect(await s.ledger.remaining(`99-${job.head}`, 99)).toBe(5_000_000 - 228 * 13 - 16000 * 50);
  });
  it('credits confirmed cache reads without assuming future cache hits', async () => {
    const s = await setup();
    vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => new Request(url, init).url.endsWith('/input_tokens') ? Response.json({ input_tokens: 100 }) : new Response('data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":80},"output_tokens":10}}}\n\n')));
    await (await s.gateway.respond(s.token, request)).text();
    expect(await s.ledger.remaining(`99-${job.head}`, 99)).toBe(5_000_000 - 840);
  });
  it('continues investigation when the current call is funded despite a large uncached history', async () => {
    const s = await setup();
    await s.ledger.reserve('earlier-runs', `99-${job.head}`, 99, 3_000_000);
    vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => {
      const r = new Request(url, init);
      if (r.url.endsWith('/input_tokens')) return Response.json({ input_tokens: 80000 });
      const p: any = await r.json();
      expect(p.tool_choice).not.toBe('none');
      expect(p.max_output_tokens).toBe(16000);
      return new Response('data: {"type":"response.completed","response":{"usage":{"input_tokens":80000,"output_tokens":10}}}\n\n');
    }));
    await (await s.gateway.respond(s.token, request)).text();
    expect(await s.ledger.remaining(`99-${job.head}`, 99)).toBe(959500);
  });
  it('reserves a final answer turn near the budget boundary and records a hard stop', async () => {
    const s = await setup();
    await s.ledger.reserve('prior-spending', `99-${job.head}`, 99, 4_700_000);
    vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => {
      const r = new Request(url, init);
      if (r.url.endsWith('/input_tokens')) return Response.json({ input_tokens: 100 });
      const p: any = await r.json();
      expect(p.tool_choice).toBe('none');
      expect(p.input.at(-1)).toMatchObject({ role: 'developer', content: expect.stringContaining('Final review turn') });
      return new Response('data: {"type":"response.created"}\n\n');
    }));
    await (await s.gateway.respond(s.token, request)).text();
    expect((await s.gateway.respond(s.token, request.replace('review', 'finish'))).status).toBe(429);
    expect(await s.ledger.sessionFailure(s.token)).toBe('budget_exhausted');
  });
});

it('reconnects during streamed settlement and retries a lost commit acknowledgement without repeating inference', async () => {
  const s = await setup();
  let generation = 0, settlements = 0, creates = 0;
  const getByName = () => {
    const connected = generation;
    return new Proxy(s.ledger, { get(target, key) {
      return async (...args: any[]) => {
        if (connected !== generation) throw new Error('Connection closed: this Durable Object instance is no longer active. Reconnect or retry the request.');
        const result = await (target as any)[key](...args);
        if (key === 'settle' && ++settlements === 1) {
          generation++;
          throw Object.assign(new Error('Lost acknowledgement after commit'), { retryable: true });
        }
        return result;
      };
    } });
  };
  const gateway = new ModelGateway(createExecutionContext(), { LEDGER: { getByName }, OPENAI_API_KEY: 'sk-test-only' });
  vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => {
    if (new Request(url, init).url.endsWith('/input_tokens')) return Response.json({ input_tokens: 100 });
    creates++; generation++;
    return new Response('data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":10}}}\n\n');
  }));
  expect(await (await gateway.respond(s.token, request)).text()).toContain('response.completed');
  expect(settlements).toBe(2);
  expect(creates).toBe(1);
  expect(await s.ledger.remaining(`99-${job.head}`, 99)).toBe(5_000_000 - 1800);
  expect((await s.ledger.costs(job.id)).modelCalls).toBe(1);
});

it('does not replay an ambiguous reservation or submit paid inference after its acknowledgement is lost', async () => {
  const s = await setup(); let reservations = 0;
  const gateway = new ModelGateway(createExecutionContext(), { LEDGER: { getByName: () => new Proxy(s.ledger, { get(target, key) {
    return async (...args: any[]) => {
      const result = await (target as any)[key](...args);
      if (key === 'reserve') { reservations++; throw Object.assign(new Error('Lost reservation acknowledgement'), { retryable: true }); }
      return result;
    };
  } }) }, OPENAI_API_KEY: 'sk-test-only' });
  const send = vi.fn(async () => Response.json({ input_tokens: 100 })); vi.stubGlobal('fetch', send);
  await expect(gateway.respond(s.token, request)).rejects.toThrow('Lost reservation acknowledgement');
  expect(reservations).toBe(1);
  expect(send).toHaveBeenCalledTimes(1); // Token counting only, no paid response.
  expect(await s.ledger.remaining(`99-${job.head}`, 99)).toBeLessThan(5_000_000);
});
