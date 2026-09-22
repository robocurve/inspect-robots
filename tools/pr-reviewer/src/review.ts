import { zodTextFormat } from 'openai/helpers/zod';
import type { WorkflowStep } from 'cloudflare:workers';
import policy from './policy.md';
import { ReviewSchema, RunOutput, validateReview, type Execution, type Job } from './common';
import { collectContext, readFile } from './context';

const retryRead = { retries: { limit: 3, delay: '2 seconds' }, timeout: '2 minutes' } as const;
export type ReviewEnvironment = { LEDGER: Pick<ReviewerEnv['LEDGER'], 'getByName'>; PUBLISHER: Pick<ReviewerEnv['PUBLISHER'], 'read'>; RUNNER: Pick<ReviewerEnv['RUNNER'], 'start' | 'poll' | 'cleanup'> };
export async function runReview(env: ReviewEnvironment, job: Job, step: Pick<WorkflowStep, 'do' | 'sleep'>) {
  // Fetch a fresh stub for each RPC, including retries after Workflow sleeps.
  const ledger = () => env.LEDGER.getByName('budget');
  const read = async (p: string): Promise<any> => JSON.parse(await env.PUBLISHER.read(p));
  const context = JSON.parse(await step.do('gather review discussion', retryRead, async () => JSON.stringify(await collectContext(read, job))));
  const execution: Execution = JSON.parse(await step.do('prepare reviewer execution', retryRead, async () => JSON.stringify(await ledger().prepareExecution(job.id, context.merge_base))));
  let timedOut = false;
  let raw = await ledger().runOutput(job.id);
  try {
    if (raw === null) {
      if (await ledger().isSandboxCleaned(job.id)) throw new Error('codex_review_incomplete');
      if (Date.now() - execution.started >= 22 * 60_000) { timedOut = true; throw new Error('model_timeout'); }
      await step.do('launch Codex process', retryRead, async () => {
        if (await ledger().runOutput(job.id) !== null) return;
        await env.RUNNER.start(job.head, execution.mergeBase, JSON.stringify(context), policy, JSON.stringify(zodTextFormat(ReviewSchema, 'review').schema), execution.token, execution.sandbox, execution.checkpointToken);
      });
      for (let poll = 0; poll < 100; poll++) {
        const state = await step.do(`inspect reviewer process ${poll}`, retryRead, async () => {
          if (await ledger().runOutput(job.id) !== null) return 'complete';
          if (Date.now() - execution.started >= 22 * 60_000) return 'timeout';
          return await env.RUNNER.poll(execution.sandbox, execution.token) ? 'complete' : 'running';
        });
        if (state === 'complete') break;
        if (state === 'timeout') { timedOut = true; throw new Error('model_timeout'); }
        await step.sleep(`wait for reviewer ${poll}`, '15 seconds');
      }
      raw = await ledger().runOutput(job.id);
      if (raw === null) { timedOut = true; throw new Error('model_timeout'); }
    }
  } catch (error) {
    // A failed step/acknowledgement cannot discard a separately committed result.
    raw = await ledger().runOutput(job.id);
    if (raw === null) throw error;
    console.log(JSON.stringify({ event: 'review_result_recovered', job: job.id }));
  } finally {
    if (raw !== null || timedOut) {
      try { await step.do('revoke reviewer access', retryRead, () => ledger().closeSession(execution.token)); }
      catch { console.error(JSON.stringify({ event: 'review_revocation_deferred', job: job.id })); }
      // Cleanup is independently retryable and cannot replace a saved verdict.
      try {
        await step.do('cleanup reviewer sandbox', retryRead, () => env.RUNNER.cleanup(execution.sandbox));
        await step.do('record sandbox cleanup', () => ledger().sandboxCleaned(job.id));
      }
      catch { console.error(JSON.stringify({ event: 'review_cleanup_deferred', job: job.id })); }
    }
  }
  const output = RunOutput.parse(JSON.parse(raw));
  if (output.exitCode !== 0 || !output.review) throw new Error(['model_timeout', 'budget_exhausted', 'billing_hold', 'context_too_large'].includes(output.failure ?? '') ? output.failure! : 'codex_review_incomplete');
  const result = validateReview(output.review);
  for (const blocker of result.blockers) {
    try { await readFile(read, job, blocker.file, 'head', blocker.line, 1); }
    catch { await readFile(read, { ...job, base: execution.mergeBase }, blocker.file, 'base', blocker.line, 1); }
  }
  return { ...result, execution_records: output.executions, cost_summary: await ledger().costs(job.id) };
}
