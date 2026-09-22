import { env } from 'cloudflare:test';
import type { WorkflowStep } from 'cloudflare:workers';
import { describe, expect, it, vi } from 'vitest';
import worker, { handleWebhook, ReviewWorkflow } from '../src/worker';
import { runReview } from '../src/review';
import type { Job } from '../src/common';

const head = 'a'.repeat(40), base = 'b'.repeat(40);
const job: Job = { id: 'run-1', pr: 9, head, base, scope: '', status: 'running', result: null, notified: 0, created: 1 };
const pr = { number: 9, head: { sha: head }, base: { sha: base }, state: 'open', draft: false, changed_files: 1, title: 'Fix boundary', body: '', user: { login: 'author' } };
function setup() {
  const ledger = env.LEDGER.getByName(crypto.randomUUID());
  const read = vi.fn(async (p: string) => {
    if (p === '/pulls/9') return JSON.stringify(pr);
    if (p.startsWith('/compare/')) return JSON.stringify({ merge_base_commit: { sha: base } });
    if (p.includes('/files?')) return JSON.stringify([{ filename: 'x.py', status: 'modified', additions: 1, deletions: 1, patch: '@@ -1 +1 @@\n-old\n+new' }]);
    if (p.startsWith('/contents/')) return JSON.stringify({ type: 'file', encoding: 'base64', size: 4, content: btoa('new\n') });
    if (p.startsWith('/git/trees/')) return JSON.stringify({ truncated: false, tree: [] });
    return '[]';
  });
  const create = vi.fn<ReviewerEnv['REVIEW']['create']>();
  const config = { ENABLED: 'true', GITHUB_WEBHOOK_SECRET: 'test-hook', OPENAI_API_KEY: 'sk-test', LEDGER: { getByName: () => ledger }, PUBLISHER: { read }, RUNNER: { start: vi.fn(async (...args: any[]) => { await ledger.checkpoint(args[5], JSON.stringify({ exitCode: 1, review: null, executions: [] })); }), poll: vi.fn(async () => false), cleanup: vi.fn(async () => {}) }, REVIEW: { create } };
  const steps: { name: string; options: any }[] = [];
  const step = { do: async (name: string, optionsOrFn: any, callback?: () => Promise<any>) => { steps.push({ name, options: callback ? optionsOrFn : {} }); return (callback ?? optionsOrFn)(); }, sleep: async () => {} } as Pick<WorkflowStep, 'do' | 'sleep'>;
  return { ledger, read, create, config, step, steps };
}
async function webhook(body: any, event: string) {
  const text = JSON.stringify(body);
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode('test-hook'), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const signature = [...new Uint8Array(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(text)))].map(x => x.toString(16).padStart(2, '0')).join('');
  return new Request('https://reviewer.test/webhook', { method: 'POST', body: text, headers: { 'x-github-event': event, 'x-hub-signature-256': 'sha256=' + signature } });
}
const repo = { repository: { full_name: 'robocurve/inspect-robots' }, installation: { id: 163290338 } };

describe('webhook dispatch', () => {
  it('requeues the latest revision when a waiting snapshot becomes stale before admission', async () => {
    const s = setup(); await s.ledger.register(job);
    const read = s.read.getMockImplementation()!;
    const newHead = 'c'.repeat(40);
    s.read.mockImplementation(async p => p === '/pulls/9' ? JSON.stringify({ ...pr, head: { sha: newHead } }) : read(p));
    const publish = vi.fn(async () => false);
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } } });
    await workflow.run({ instanceId: job.id, payload: { id: job.id } } as any, s.step as WorkflowStep);
    expect((await s.ledger.job(job.id))?.status).toBe('stale');
    expect(await s.ledger.pending()).toEqual([expect.objectContaining({ pr: 9, head: newHead, status: 'queued' })]);
    expect(s.create).toHaveBeenCalledTimes(1);
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    expect((await s.ledger.costs(job.id)).sandboxMicros).toBe(0);
  });
  it('rejects unsigned input before any GitHub read', async () => {
    const s = setup();
    const response = await handleWebhook(new Request('https://x/', { method: 'POST', body: '{}' }), s.config);
    expect(response.status).toBe(401); expect(s.read).not.toHaveBeenCalled();
  });
  it('ignores other repositories and installations and deduplicates delivery', async () => {
    const s = setup();
    const payload = { ...repo, action: 'opened', number: 9 };
    await handleWebhook(await webhook({ ...payload, installation: { id: 2 } }, 'pull_request'), s.config);
    expect(s.read).not.toHaveBeenCalled();
    await handleWebhook(await webhook(payload, 'pull_request'), s.config);
    await handleWebhook(await webhook(payload, 'pull_request'), s.config);
    expect(s.create).toHaveBeenCalledTimes(1);
  });
  it('allows only Jay to request reruns and rejects a scope decision for another head', async () => {
    const s = setup();
    const payload = { ...repo, action: 'created', issue: { number: 9, pull_request: {} }, comment: { id: 3, user: { id: 1 }, body: '/review' } };
    await handleWebhook(await webhook(payload, 'issue_comment'), s.config);
    expect(s.read).not.toHaveBeenCalled();
    payload.comment.user.id = 42904912;
    payload.comment.body = `/review scope ${base} Approved feature`;
    await handleWebhook(await webhook(payload, 'issue_comment'), s.config);
    expect(s.create).not.toHaveBeenCalled();
    payload.comment.body = `/review scope ${head} Approved feature`;
    await handleWebhook(await webhook(payload, 'issue_comment'), s.config);
    expect(s.create).toHaveBeenCalledTimes(1);
  });
});

describe('Codex review lifecycle', () => {
  it('runs a fresh CLI session with the policy and closes its scoped gateway access', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    let capability = '';
    const result = { worthwhile: 'YES', scope: 'ESTABLISHED', verdict: 'APPROVE', recommended_action: 'MERGE', rationale: 'Concrete boundary fix.', blockers: [], contract_and_test_review: 'Preserved.', checks: [], limitations: [], sufficient_review: true, decision_needed: '', body: 'Verified.' };
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => {
      capability = args[5];
      expect(await s.ledger.session(capability)).toMatchObject({ id: job.id });
      expect(args[0]).toBe(head); expect(args[1]).toBe(base);
      expect(args[3]).toContain('Authority and scope');
      await s.ledger.checkpoint(capability, JSON.stringify({ exitCode: 0, review: result, executions: [] }));
    });
    expect((await runReview(s.config as any, job, s.step)).verdict).toBe('APPROVE');
    expect(await s.ledger.session(capability)).toBeNull();
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(4_900_000);
  });
  it('declines an underfunded rerun without starting a sandbox or spending more', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    await s.ledger.reserve('earlier-run', `9-${head}`, 9, 3_100_000);
    await expect(runReview(s.config as any, job, s.step)).rejects.toThrow('insufficient_run_budget');
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(1_900_000);
  });
  it('never accepts a partial verdict or retries a failed Codex session', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    await expect(runReview(s.config as any, job, s.step)).rejects.toThrow('codex_review_incomplete');
    expect(s.config.RUNNER.start).toHaveBeenCalledTimes(1);
    expect(s.config.RUNNER.poll).not.toHaveBeenCalled();
  });
  it('preserves the trusted gateway stop even if CLI diagnostics omit its reason', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => {
      await s.ledger.sessionFailure(args[5], 'budget_exhausted');
      await s.ledger.checkpoint(args[5], JSON.stringify({ exitCode: 0, review: { verdict: 'APPROVE' }, failure: null, executions: [] }));
    });
    await expect(runReview(s.config as any, job, s.step)).rejects.toThrow('budget_exhausted');
  });
});

const savedApproval = { worthwhile: 'YES', scope: 'ESTABLISHED', verdict: 'APPROVE', recommended_action: 'MERGE', rationale: 'Concrete boundary fix.', blockers: [], contract_and_test_review: 'Preserved.', checks: [], limitations: [], sufficient_review: true, decision_needed: '', body: 'Verified.' };
const completeOutput = JSON.stringify({ exitCode: 0, failure: null, review: savedApproval, executions: [] });
describe('durable result recovery', () => {
  it('recovers after the workflow loses the poll acknowledgement without new inference', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    let token = '';
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { token = args[5]; });
    s.config.RUNNER.poll.mockImplementation(async () => { await s.ledger.checkpoint(token, completeOutput); throw new Error('injected lost acknowledgement'); });
    expect((await runReview(s.config as any, job, s.step)).verdict).toBe('APPROVE');
    expect(s.config.RUNNER.start).toHaveBeenCalledTimes(1);
    expect(s.config.RUNNER.cleanup).toHaveBeenCalledTimes(1);
    expect(await s.ledger.session(token)).toBeNull();
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(4_900_000);
  });
  it('reuses the completed result on another workflow even if no budget remains', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    const execution = await s.ledger.prepareExecution(job.id, base);
    await s.ledger.checkpoint(execution.token, completeOutput);
    await s.ledger.closeSession(execution.token);
    await s.ledger.reserve('other-spending', `9-${head}`, 9, 4_900_000);
    expect((await runReview(s.config as any, job, s.step)).verdict).toBe('APPROVE');
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    expect(s.config.RUNNER.poll).not.toHaveBeenCalled();
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(0);
  });
  it('does not let cleanup failure overwrite a completed review', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { await s.ledger.checkpoint(args[5], completeOutput); });
    s.config.RUNNER.cleanup.mockRejectedValue(new Error('injected destroy failure'));
    expect((await runReview(s.config as any, job, s.step)).verdict).toBe('APPROVE');
    expect(await s.ledger.runOutput(job.id)).toBe(completeOutput);
  });
  it('keeps a pending process accessible after a transient workflow failure', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    let token = '';
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { token = args[5]; });
    s.config.RUNNER.poll.mockRejectedValue(new Error('injected platform failure'));
    await expect(runReview(s.config as any, job, s.step)).rejects.toThrow('injected platform failure');
    expect(await s.ledger.session(token)).not.toBeNull();
    expect(s.config.RUNNER.cleanup).not.toHaveBeenCalled();
    await s.ledger.checkpoint(token, completeOutput);
    expect((await runReview(s.config as any, job, s.step)).verdict).toBe('APPROVE');
    expect(s.config.RUNNER.start).toHaveBeenCalledTimes(1);
  });
});

describe('publication durability', () => {
  it('keeps a failed workflow hold pending and publishes it after GitHub recovers', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    await s.ledger.reserve('earlier-run', `9-${head}`, 9, 3_100_000);
    const publish = vi.fn(async (_job: unknown, _result: unknown, notice?: string) => {
      if (notice === 'held') throw new Error('publisher outage');
      return true;
    });
    const config = { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } };
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: config });
    await expect(workflow.run({ instanceId: job.id, payload: { id: job.id } } as any, s.step as WorkflowStep)).rejects.toThrow('publisher outage');
    expect((await s.ledger.job(job.id))?.status).toBe('holding');
    expect((await s.ledger.pending()).map(j => j.id)).toContain(job.id);
    expect((await s.ledger.queueState()).owner).toBeNull();
    publish.mockResolvedValue(true);
    await worker.scheduled({} as ScheduledController, config as any);
    expect((await s.ledger.job(job.id))?.status).toBe('held');
    expect(publish).toHaveBeenLastCalledWith(expect.objectContaining({ id: job.id }), expect.objectContaining({ code: 'insufficient_run_budget' }), 'held');
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    expect((await s.ledger.costs(job.id)).modelCalls).toBe(0);
  });
  it('retries a hold created by scheduled interruption recovery', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    await s.ledger.finish(job.id, 'running');
    const publish = vi.fn(async (): Promise<boolean> => { throw new Error('publisher outage'); });
    const get = vi.fn(async () => ({ status: async () => ({ status: 'errored' }) }));
    const config = { ...s.config, REVIEW: { ...s.config.REVIEW, get }, PUBLISHER: { ...s.config.PUBLISHER, publish } };
    await worker.scheduled({} as ScheduledController, config as any);
    expect((await s.ledger.job(job.id))?.status).toBe('holding');
    publish.mockResolvedValue(true);
    await worker.scheduled({} as ScheduledController, config as any);
    expect((await s.ledger.job(job.id))?.status).toBe('held');
    expect(publish).toHaveBeenCalledTimes(2);
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
  it('keeps a validated verdict pending when GitHub publication fails', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { await s.ledger.checkpoint(args[5], completeOutput); });
    const publish = vi.fn(async (_job: unknown, _result: unknown, notice?: string) => { if (notice === 'started' || notice === 'queued') return true; throw new Error('injected publisher outage'); });
    const config = { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } };
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: config });
    await workflow.run({ instanceId: job.id, payload: { id: job.id } } as any, s.step as WorkflowStep);
    const saved = await s.ledger.job(job.id);
    expect(saved?.status).toBe('publishing');
    expect(JSON.parse(saved!.result!).verdict).toBe('APPROVE');
    expect(publish).toHaveBeenCalledTimes(3);
    expect((await s.ledger.pending()).map(v => v.id)).toContain(job.id);
  });
});


describe('scheduled recovery', () => {
  it('retries uncertain cleanup for terminal jobs before admitting another review', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    const execution = await s.ledger.prepareExecution(job.id, base);
    await s.ledger.checkpoint(execution.token, completeOutput);
    await s.ledger.finish(job.id, 'done');
    await s.ledger.register({ ...job, id: 'next', pr: 10 });
    s.config.RUNNER.cleanup.mockRejectedValueOnce(new Error('temporary cleanup failure'));
    await worker.scheduled({} as ScheduledController, s.config as any);
    expect(await s.ledger.claimReviewSlot('next')).toBe('waiting');
    await worker.scheduled({} as ScheduledController, s.config as any);
    expect(await s.ledger.claimReviewSlot('next')).toBe('acquired');
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
  it('resumes a failed execution using the original job and latest workflow instance', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    const execution = await s.ledger.prepareExecution(job.id, base);
    await s.ledger.finish(job.id, 'running');
    await s.ledger.workflowInstance(job.id, 'previous-recovery');
    const get = vi.fn(async () => ({ status: async () => ({ status: 'errored' }) }));
    const config = { ...s.config, REVIEW: { ...s.config.REVIEW, get } };
    await worker.scheduled({} as ScheduledController, config as any);
    expect(get).toHaveBeenCalledWith('previous-recovery');
    expect((await s.ledger.job(job.id))?.status).toBe('recovering');
    await worker.scheduled({} as ScheduledController, config as any);
    expect(s.create).toHaveBeenCalledWith(expect.objectContaining({ params: { id: job.id } }));
    expect(await s.ledger.execution(job.id)).toEqual(execution);
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(4_900_000);
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
  it('publishes a saved result after an outage without launching another review', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    await s.ledger.finish(job.id, 'publishing', JSON.stringify(savedApproval));
    const publish = vi.fn(async () => true);
    await worker.scheduled({} as ScheduledController, { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } } as any);
    expect(publish).toHaveBeenCalledTimes(1);
    expect((await s.ledger.job(job.id))?.status).toBe('approved');
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    expect(s.create).not.toHaveBeenCalled();
  });
});

it('queues simultaneous ready events and runs their workflows one at a time', async () => {
  const s = setup();
  const read = s.read.getMockImplementation()!;
  s.read.mockImplementation(async p => p === '/pulls/10' ? JSON.stringify({ ...pr, number: 10 }) : read(p));
  await Promise.all([9, 10].map(async number => handleWebhook(await webhook({ ...repo, action: 'ready_for_review', number }, 'pull_request'), s.config)));
  expect(s.create).toHaveBeenCalledTimes(2);
  const jobs = await s.ledger.pending();
  let active = 0, peak = 0;
  let unblockFirst!: () => void, firstStarted!: () => void, unblockWaiter!: () => void, waiterSleeping!: () => void;
  const blocked = new Promise<void>(resolve => { unblockFirst = resolve; });
  const started = new Promise<void>(resolve => { firstStarted = resolve; });
  const waiting = new Promise<void>(resolve => { waiterSleeping = resolve; });
  const resume = new Promise<void>(resolve => { unblockWaiter = resolve; });
  s.config.RUNNER.start.mockImplementation(async (...args: any[]) => {
    peak = Math.max(peak, ++active);
    if (s.config.RUNNER.start.mock.calls.length === 1) { firstStarted(); await blocked; }
    await s.ledger.checkpoint(args[5], completeOutput);
  });
  s.config.RUNNER.cleanup.mockImplementation(async () => { active--; });
  const publish = vi.fn(async () => true);
  const config = { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } };
  function execute(id: string, step: typeof s.step) {
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: config });
    return workflow.run({ instanceId: id, payload: { id } } as any, step as WorkflowStep);
  }
  const first = execute(jobs[0].id, s.step);
  await started;
  const second = execute(jobs[1].id, { ...s.step, sleep: async () => { waiterSleeping(); await resume; } });
  await waiting;
  expect(s.config.RUNNER.start).toHaveBeenCalledTimes(1);
  expect((await s.ledger.costs(jobs[1].id)).sandboxMicros).toBe(0);
  expect(publish.mock.calls.some((c: any[]) => c[0].id === jobs[1].id && c[2] === 'queued')).toBe(true);
  unblockFirst(); await first;
  unblockWaiter(); await second;
  expect(s.config.RUNNER.start).toHaveBeenCalledTimes(2);
  expect(peak).toBe(1);
  expect((await s.ledger.queueState()).owner).toBeNull();
});


it('allows an operator to inspect saved output without inference or publication', async () => {
  const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
  const execution = await s.ledger.prepareExecution(job.id, base);
  await s.ledger.checkpoint(execution.token, completeOutput);
  const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
  Object.defineProperty(workflow, 'env', { value: s.config });
  const output = await workflow.run({ payload: { id: job.id, inspectOutput: true } } as any, s.step as WorkflowStep);
  expect(JSON.parse(output as string).output.review).toEqual(savedApproval);
  expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  expect(s.read).not.toHaveBeenCalled();
  expect((await s.ledger.job(job.id))?.status).toBe('queued');
  expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(4_900_000);
});

const disconnected = () => new Error('Connection closed: this Durable Object instance is no longer active. Reconnect or retry the request.');
describe('Durable Object connection recovery', () => {
  it('reconnects after queue sleep instead of keeping a permanently broken stub', async () => {
    const s = setup();
    await s.ledger.register({ ...job, id: 'first' });
    await s.ledger.claimReviewSlot('first');
    await s.ledger.register(job);
    let generation = 0;
    const getByName = () => {
      const connected = generation;
      return new Proxy(s.ledger, { get(target, key) {
        return async (...args: any[]) => {
          if (connected !== generation) throw disconnected();
          return (target as any)[key](...args);
        };
      } });
    };
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { await s.ledger.checkpoint(args[5], completeOutput); });
    const publish = vi.fn(async () => true);
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: { ...s.config, LEDGER: { getByName }, PUBLISHER: { ...s.config.PUBLISHER, publish } } });
    const sleep = vi.fn(async () => { generation++; await s.ledger.finish('first', 'done'); await s.ledger.releaseReviewSlot('first'); });
    await workflow.run({ instanceId: job.id, payload: { id: job.id } } as any, { ...s.step, sleep } as unknown as WorkflowStep);
    expect(sleep).toHaveBeenCalledTimes(1);
    expect(s.config.RUNNER.start).toHaveBeenCalledTimes(1);
    expect((await s.ledger.job(job.id))?.status).toBe('approved');
    expect((await s.ledger.costs(job.id)).sandboxMicros).toBe(100_000);
  });
  it('leaves a transient admission failure queued for reconciliation without charges or a terminal notice', async () => {
    const s = setup(); await s.ledger.register(job);
    const getByName = () => new Proxy(s.ledger, { get(target, key) {
      if (key === 'claimReviewSlot') return async () => { throw disconnected(); };
      return (...args: any[]) => (target as any)[key](...args);
    } });
    const publish = vi.fn(async () => true);
    const config = { ...s.config, LEDGER: { getByName }, REVIEW: { ...s.config.REVIEW, get: async () => ({ status: async () => ({ status: 'complete' }) }) }, PUBLISHER: { ...s.config.PUBLISHER, publish } };
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: config });
    await workflow.run({ instanceId: job.id, payload: { id: job.id } } as any, s.step as WorkflowStep);
    expect((await s.ledger.job(job.id))?.status).toBe('queued');
    expect(publish.mock.calls.map((c: any[]) => c[2])).toEqual(['queued']);
    expect((await s.ledger.costs(job.id)).sandboxMicros).toBe(0);
    await worker.scheduled({} as ScheduledController, config as any);
    expect(s.create).toHaveBeenCalledWith(expect.objectContaining({ params: { id: job.id } }));
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
  it('does not replace a published verdict when a later budget warning fails', async () => {
    const s = setup(); await s.ledger.register(job);
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { await s.ledger.checkpoint(args[5], completeOutput); });
    const getByName = () => new Proxy(s.ledger, { get(target, key) {
      if (key === 'warningNeeded') return async () => true;
      return (...args: any[]) => (target as any)[key](...args);
    } });
    const publish = vi.fn(async (_job: unknown, _result: unknown, notice?: string) => { if (notice === 'budget-warning') throw disconnected(); return true; });
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: { ...s.config, LEDGER: { getByName }, PUBLISHER: { ...s.config.PUBLISHER, publish } } });
    await workflow.run({ instanceId: job.id, payload: { id: job.id } } as any, s.step as WorkflowStep);
    const saved = await s.ledger.job(job.id);
    expect(saved?.status).toBe('approved');
    expect(JSON.parse(saved!.result!).verdict).toBe('APPROVE');
    expect(publish.mock.calls.some(c => c[2] === 'held')).toBe(false);
  });
});

// Exercise signed webhook -> SQLite registration -> reconciliation -> Workflow,
// including platform IDs that cannot be reused once a Workflow already exists.
describe('ready and reopened PR lifecycle', () => {
  it.each(['draft', 'closed'] as const)('reviews an unstarted %s PR again at the same revision, without duplicate execution', async state => {
    const s = setup(); let live = { ...pr };
    const read = s.read.getMockImplementation()!;
    s.read.mockImplementation(async p => p === '/pulls/9' ? JSON.stringify(live) : read(p));
    const instances = new Set<string>();
    s.create.mockImplementation(async options => {
      if (instances.has(options!.id!)) throw new Error('workflow_already_exists');
      instances.add(options!.id!);
      return {} as WorkflowInstance;
    });
    const event = (action: string) => webhook({ ...repo, action, number: 9 }, 'pull_request');
    await handleWebhook(await event('opened'), s.config);
    const original = (await s.ledger.pending())[0];
    // Spending from an earlier manual run must remain charged across revival.
    await s.ledger.reserve('prior-manual-run', `9-${head}`, 9, 1_000_000);
    const ids = [original.id];
    for (let cycle = 0; cycle < 2; cycle++) {
      live = { ...pr, draft: state === 'draft', state: state === 'closed' ? 'closed' : 'open' };
      await worker.scheduled({} as ScheduledController, s.config as any);
      expect(await s.ledger.pending()).toEqual([]);
      expect((await s.ledger.job(ids.at(-1)!))?.status).toBe('stale');
      live = { ...pr };
      const action = state === 'draft' ? 'ready_for_review' : 'reopened';
      await Promise.all([1, 2].map(async () => handleWebhook(await event(action), s.config)));
      const pending = await s.ledger.pending();
      expect(pending).toHaveLength(1);
      expect(pending[0]).toMatchObject({ head, base, status: 'queued' });
      expect(ids).not.toContain(pending[0].id);
      ids.push(pending[0].id);
      expect(s.create).toHaveBeenCalledTimes(cycle + 2);
      expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(4_000_000);
    }
    s.config.RUNNER.start.mockImplementation(async (...args: any[]) => { await s.ledger.checkpoint(args[5], completeOutput); });
    const publish = vi.fn(async () => true);
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } } });
    // An old sleeping workflow wakes after replacement. It must stay obsolete.
    await workflow.run({ instanceId: original.id, payload: { id: original.id } } as any, s.step as WorkflowStep);
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    const id = ids.at(-1)!;
    await workflow.run({ instanceId: id, payload: { id } } as any, s.step as WorkflowStep);
    expect((await s.ledger.job(id))?.status).toBe('approved');
    expect(s.config.RUNNER.start).toHaveBeenCalledTimes(1);
    expect((await s.ledger.job(original.id))?.status).toBe('stale');
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(3_900_000);
  });
  it('does not duplicate a still-queued review when ready/reopened notifications repeat', async () => {
    const s = setup();
    for (const action of ['opened', 'ready_for_review', 'reopened', 'edited']) {
      await handleWebhook(await webhook({ ...repo, action, number: 9 }, 'pull_request'), s.config);
    }
    expect(s.create).toHaveBeenCalledTimes(1);
    expect(await s.ledger.pending()).toHaveLength(1);
  });
  it('recovers a replacement after Workflow creation fails without losing registration', async () => {
    const s = setup();
    const request = () => webhook({ ...repo, action: 'reopened', number: 9 }, 'pull_request');
    await handleWebhook(await request(), s.config);
    const original = (await s.ledger.pending())[0];
    await s.ledger.finish(original.id, 'stale');
    s.create.mockRejectedValueOnce(new Error('workflow service unavailable'));
    await expect(handleWebhook(await request(), s.config)).rejects.toThrow('workflow service unavailable');
    const replacement = (await s.ledger.pending())[0];
    expect(replacement.id).not.toBe(original.id);
    await handleWebhook(await request(), s.config);
    expect(s.create).toHaveBeenCalledTimes(2);
    const get = vi.fn(async () => { throw new Error('workflow not created'); });
    await worker.scheduled({} as ScheduledController, { ...s.config, REVIEW: { ...s.config.REVIEW, get } } as any);
    expect(s.create).toHaveBeenLastCalledWith({ id: replacement.id, params: { id: replacement.id } });
    expect(s.create).toHaveBeenCalledTimes(3);
    expect((await s.ledger.costs(replacement.id)).sandboxMicros).toBe(0);
  });
  it('does not automatically replace started or charged stale jobs', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    const execution = await s.ledger.prepareExecution(job.id, base);
    await s.ledger.finish(job.id, 'stale');
    expect(await s.ledger.enqueueJob(job)).toBeNull();
    expect(await s.ledger.execution(job.id)).toEqual(execution);
    const charged = { ...job, id: 'legacy-charged' };
    await s.ledger.register(charged);
    await s.ledger.reserve(`${charged.id}-codex-reservation`, `9-${head}`, 9, 500_000);
    await s.ledger.finish(charged.id, 'stale');
    expect(await s.ledger.enqueueJob(charged)).toBeNull();
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(4_400_000);
  });
  it('prevents an admitted stale workflow from starting or charging after replacement', async () => {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    expect(await s.ledger.startReview(job.id)).toBe(true);
    await s.ledger.finish(job.id, 'stale');
    const replacement = await s.ledger.enqueueJob(job);
    expect(replacement).not.toBeNull();
    expect(await s.ledger.startReview(job.id)).toBe(false);
    await expect((async () => await s.ledger.prepareExecution(job.id, base))()).rejects.toThrow('stale_revision');
    expect((await s.ledger.job(job.id))?.status).toBe('stale');
    expect(await s.ledger.execution(job.id)).toBeNull();
    expect(await s.ledger.remaining(`9-${head}`, 9)).toBe(5_000_000);
    await s.ledger.releaseReviewSlot(job.id);
    expect(await s.ledger.claimReviewSlot(replacement!)).toBe('acquired');
  });
});

describe('operator recovery of held saved results', () => {
  async function held(withOutput = true) {
    const s = setup(); await s.ledger.register(job); await s.ledger.claimReviewSlot(job.id);
    const execution = await s.ledger.prepareExecution(job.id, base);
    if (withOutput) await s.ledger.checkpoint(execution.token, completeOutput);
    await s.ledger.closeSession(execution.token);
    await s.ledger.sandboxCleaned(job.id);
    await s.ledger.finish(job.id, 'held', JSON.stringify({ code: 'invalid_review_result' }));
    await s.ledger.releaseReviewSlot(job.id);
    const publish = vi.fn(async () => true);
    const workflow = Object.create(ReviewWorkflow.prototype) as ReviewWorkflow;
    Object.defineProperty(workflow, 'env', { value: { ...s.config, PUBLISHER: { ...s.config.PUBLISHER, publish } } });
    const run = () => workflow.run({ instanceId: 'operator-recovery', payload: { id: job.id, recoverSaved: true } } as any, s.step as WorkflowStep);
    return { ...s, execution, publish, run };
  }
  it('revalidates and publishes through the complete workflow without new inference or spending', async () => {
    const s = await held();
    await s.ledger.reserve('prior-spending', `9-${head}`, 9, 4_900_000);
    const costs = await s.ledger.costs(job.id);
    await s.run();
    expect((await s.ledger.job(job.id))?.status).toBe('approved');
    expect(s.publish).toHaveBeenCalledWith(expect.objectContaining({ id: job.id }), expect.objectContaining({ verdict: 'APPROVE' }));
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
    expect(s.config.RUNNER.poll).not.toHaveBeenCalled();
    expect(await s.ledger.runOutput(job.id)).toBe(completeOutput);
    expect(await s.ledger.execution(job.id)).toEqual(s.execution);
    expect(await s.ledger.costs(job.id)).toEqual(costs);
    expect((await s.ledger.queueState()).owner).toBeNull();
    await s.run(); // A repeated operator request must not reopen the verdict.
    expect(s.publish).toHaveBeenCalledTimes(3); // Queued, started, final verdict.
  });
  it.each(['stale', 'security_stopped', 'done'])('does not revive a %s job with saved output', async status => {
    const s = await held(); await s.ledger.finish(job.id, status);
    await s.run();
    expect((await s.ledger.job(job.id))?.status).toBe(status);
    expect(s.publish).not.toHaveBeenCalled();
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
  it('does not recover a held job without a saved output', async () => {
    const s = await held(false); await s.run();
    expect((await s.ledger.job(job.id))?.status).toBe('held');
    expect(s.publish).not.toHaveBeenCalled();
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
  it('respects the persistent safety pause', async () => {
    const s = await held(); await s.ledger.pauseReviewQueue(true); await s.run();
    expect((await s.ledger.job(job.id))?.status).toBe('held');
    expect(s.publish).not.toHaveBeenCalled();
  });
  it('refuses a saved result when the live PR revision has changed', async () => {
    const s = await held();
    s.read.mockResolvedValue(JSON.stringify({ ...pr, head: { sha: 'c'.repeat(40) } }));
    await s.run();
    expect((await s.ledger.job(job.id))?.status).toBe('held');
    expect(s.publish).not.toHaveBeenCalled();
    expect(s.config.RUNNER.start).not.toHaveBeenCalled();
  });
});
