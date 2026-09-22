import { env, evictDurableObject, runInDurableObject } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';
import type { Job } from '../src/common';

const head = 'a'.repeat(40), base = 'b'.repeat(40);
const job = (id: string, pr: number): Job => ({ id, pr, head, base, scope: '', status: 'queued', result: null, notified: 0, created: 0 });

describe('durable global review queue', () => {
  it('persists the safety pause across eviction and blocks admission, execution and charges', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    await ledger.register(job('paused', 1));
    await ledger.claimReviewSlot('paused');
    await ledger.pauseReviewQueue(true);
    await evictDurableObject(ledger);
    expect((await ledger.queueState()).paused).toBe(true);
    expect(await ledger.claimReviewSlot('paused')).toBe('waiting');
    expect(await ledger.reserve('paused-charge', `1-${head}`, 1, 100)).toBe(false);
    const error = await runInDurableObject(ledger, async instance => {
      try { await instance.prepareExecution('paused', base); return ''; }
      catch (e) { return (e as Error).message; }
    });
    expect(error).toBe('review_service_paused');
    expect((await ledger.costs('paused')).sandboxMicros).toBe(0);
    await ledger.pauseReviewQueue(false);
    expect(await ledger.claimReviewSlot('paused')).toBe('acquired');
    await ledger.prepareExecution('paused', base);
  });
  it('admits only the oldest request under concurrent claims, without spending for waiters', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    for (let i = 1; i <= 24; i++) await ledger.register(job(`run-${i}`, i));
    const claims = await Promise.all(Array.from({ length: 24 }, (_, i) => ledger.claimReviewSlot(`run-${24-i}`)));
    expect(claims.filter(v => v === 'acquired')).toHaveLength(1);
    expect(claims[23]).toBe('acquired');
    const execution = await ledger.prepareExecution('run-1', base);
    const rejected = await runInDurableObject(ledger, async instance => {
      try { await instance.prepareExecution('run-2', base); return ''; }
      catch (error) { return (error as Error).message; }
    });
    expect(rejected).toBe('review_slot_required');
    expect((await ledger.costs('run-2')).sandboxMicros).toBe(0);
    expect((await ledger.costs('run-2')).modelCalls).toBe(0);
    await evictDurableObject(ledger);
    expect(await ledger.claimReviewSlot('run-2')).toBe('waiting');
    expect(await ledger.claimReviewSlot('run-1')).toBe('acquired');
    expect(await ledger.prepareExecution('run-1', base)).toEqual(execution);
    expect((await ledger.costs('run-1')).sandboxMicros).toBe(100_000);
  });
  it('keeps ownership through failures until confirmed cleanup, then advances in FIFO order', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    await ledger.register(job('first', 1)); await ledger.register(job('second', 2));
    await ledger.claimReviewSlot('first'); await ledger.prepareExecution('first', base);
    await ledger.finish('first', 'held');
    expect(await ledger.releaseReviewSlot('first')).toBe(false);
    await ledger.releaseReviewSlot('second'); // A different workflow cannot unlock it.
    await evictDurableObject(ledger);
    expect(await ledger.claimReviewSlot('second')).toBe('waiting');
    await ledger.sandboxCleaned('first');
    expect(await ledger.releaseReviewSlot('first')).toBe(true);
    expect(await ledger.claimReviewSlot('second')).toBe('acquired');
    await ledger.releaseReviewSlot('first'); // A late acknowledgement cannot free the successor.
    expect((await ledger.queueState()).owner).toBe('second');
  });
  it('adopts an existing pre-deployment run and skips stale requests', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    await ledger.register(job('legacy', 1)); await ledger.finish('legacy', 'running');
    await ledger.register(job('stale', 2)); await ledger.register(job('fresh', 3));
    expect(await ledger.claimReviewSlot('fresh')).toBe('waiting');
    expect((await ledger.queueState()).owner).toBe('legacy');
    await ledger.finish('legacy', 'held');
    expect(await ledger.releaseReviewSlot('legacy')).toBe(true);
    await ledger.finish('stale', 'stale');
    expect(await ledger.claimReviewSlot('stale')).toBe('obsolete');
    expect(await ledger.claimReviewSlot('fresh')).toBe('acquired');
  });
});
