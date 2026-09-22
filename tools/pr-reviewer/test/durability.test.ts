import { env, evictDurableObject, runInDurableObject, createExecutionContext } from 'cloudflare:test';
import { describe, expect, it, vi } from 'vitest';
import { ReviewSandbox } from '../src/sandbox';
import { ModelGateway } from '../src/model-gateway';

const job = { id: 'durability-test', pr: 77, head: 'a'.repeat(40), base: 'b'.repeat(40), scope: '' };
const raw = JSON.stringify({ exitCode: 1, failure: 'codex_review_incomplete', review: null, executions: [] });
describe('execution records in SQLite', () => {
  it('tracks the recovery workflow rather than the failed original instance', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID()); await ledger.register(job); await ledger.claimReviewSlot(job.id);
    expect(await ledger.workflowInstance(job.id)).toBe(job.id);
    await ledger.workflowInstance(job.id, 'recovered-instance');
    await evictDurableObject(ledger);
    expect(await ledger.workflowInstance(job.id)).toBe('recovered-instance');
  });
  it('atomically prepares one execution and charge across concurrent calls and eviction', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID()); await ledger.register(job); await ledger.claimReviewSlot(job.id);
    const prepared = await Promise.all(Array.from({ length: 8 }, () => ledger.prepareExecution(job.id, job.base)));
    expect(new Set(prepared.map(v => v.token)).size).toBe(1);
    expect(new Set(prepared.map(v => v.sandbox)).size).toBe(1);
    await evictDurableObject(ledger);
    expect(await ledger.execution(job.id)).toEqual(prepared[0]);
    expect(await ledger.remaining(`77-${job.head}`, 77)).toBe(4_900_000);
  });
  it('persists an immutable checkpoint, rejects other capabilities and blocks further model spending', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID()); await ledger.register(job); await ledger.claimReviewSlot(job.id);
    const run = await ledger.prepareExecution(job.id, job.base);
    const gateway = new ModelGateway(createExecutionContext(), { LEDGER: { getByName: () => ledger }, OPENAI_API_KEY: 'sk-test' });
    const rejected = await runInDurableObject(ledger, async instance => {
      try { await instance.deliverCheckpoint(run.token, raw); return ''; }
      catch (error) { return (error as Error).message; }
    });
    expect(rejected).toBe('invalid_checkpoint_receipt');
    await gateway.deliverCheckpoint(run.checkpointToken, raw);
    await gateway.deliverCheckpoint(run.checkpointToken, raw); // Lost acknowledgement: same content is safe.
    await evictDurableObject(ledger);
    expect(await ledger.runOutput(job.id)).toBe(raw);
    await expect(gateway.checkpoint(run.token, raw.replace('exitCode":1', 'exitCode":2'))).rejects.toThrow('conflicting_review_output');
    await expect(gateway.checkpoint('f'.repeat(64), raw)).rejects.toThrow('invalid_review_session');
    vi.stubGlobal('fetch', vi.fn());
    expect((await gateway.respond(run.token, JSON.stringify({ input: [] }))).status).toBe(403);
    expect(fetch).not.toHaveBeenCalled();
  });
});

function sandboxDouble() {
  const store = new Map<string, unknown>();
  let process: object | null = null;
  const start = vi.fn(async () => { process = { id: 'codex-review' }; return process; });
  const dependencies = {
    ctx: { storage: { get: async (key: string) => store.get(key), put: async (key: string, value: unknown) => { store.set(key, value); } } },
    armDeadline: vi.fn(async () => {}), exec: vi.fn(async () => ({ success: true })), writeFile: vi.fn(async () => {}),
    getProcess: vi.fn(async () => process), startProcess: start,
  };
  const fresh = () => Object.assign(Object.create(ReviewSandbox.prototype), dependencies) as ReviewSandbox;
  const request = JSON.stringify({ head: job.head, base: job.base, token: 'c'.repeat(64) });
  vi.stubGlobal('fetch', vi.fn(async () => new Response('archive')));
  return { fresh, start, request, setProcess: () => { process = { id: 'codex-review' }; } };
}
describe('sandbox launch idempotency', () => {
  it('coalesces concurrent starts and recovers a lost acknowledgement after eviction', async () => {
    const s = sandboxDouble(); const box = s.fresh();
    s.start.mockImplementation(async () => { s.setProcess(); throw new Error('lost acknowledgement'); });
    const results = await Promise.allSettled([box.launch(s.request), box.launch(s.request)]);
    expect(results.every(v => v.status === 'rejected')).toBe(true);
    await s.fresh().launch(s.request);
    expect(s.start).toHaveBeenCalledTimes(1);
  });
  it('does not relaunch when a committed claim has no observable process', async () => {
    const s = sandboxDouble();
    s.start.mockRejectedValue(new Error('ambiguous start'));
    await expect(s.fresh().launch(s.request)).rejects.toThrow('ambiguous start');
    await expect(s.fresh().launch(s.request)).rejects.toThrow('review_launch_uncertain');
    expect(s.start).toHaveBeenCalledTimes(1);
  });
});
