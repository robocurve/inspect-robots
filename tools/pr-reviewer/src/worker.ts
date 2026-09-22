import { WorkflowEntrypoint, type WorkflowEvent, type WorkflowStep } from 'cloudflare:workers';
import { boundedText, current, digest, INSTALLATION_ID, JAY_ID, POLICY_VERSION, REPO, snapshot, validateReview, verifySignature, type Job } from './common';
import { ciGreen } from './github';
import { runReview } from './review';
import { holdReason } from './holds';
import { transientLedgerError } from './ledger-connection';
export { ReviewLedger } from './ledger';
export { ModelGateway } from './model-gateway';

export type WebhookEnvironment = Pick<ReviewerEnv, 'ENABLED' | 'GITHUB_WEBHOOK_SECRET'> & {
  LEDGER: Pick<ReviewerEnv['LEDGER'], 'getByName'>;
  PUBLISHER: Pick<ReviewerEnv['PUBLISHER'], 'read'>;
  REVIEW: Pick<ReviewerEnv['REVIEW'], 'create'>;
};
async function read(env: Pick<WebhookEnvironment, 'PUBLISHER'>, path: string): Promise<any> { return JSON.parse(await env.PUBLISHER.read(path)); }

async function enqueue(env: WebhookEnvironment, pr: number, scope = '', requestId = '') {
  const info = snapshot(await read(env, `/pulls/${pr}`));
  if (info.state !== 'open' || info.draft) return;
  const id = (await digest(`${pr}:${info.head}:${info.base}:${POLICY_VERSION}:${scope}:${requestId}`)).slice(0, 48);
  const ledger = () => env.LEDGER.getByName('budget');
  const queuedId = await ledger().enqueueJob({ id, pr, head: info.head, base: info.base, scope });
  if (queuedId) await env.REVIEW.create({ id: queuedId, params: { id: queuedId } });
}

export async function handleWebhook(request: Request, env: WebhookEnvironment): Promise<Response> {
  const body = await boundedText(request, 1_000_000);
  if (!await verifySignature(body, request.headers.get('x-hub-signature-256'), env.GITHUB_WEBHOOK_SECRET)) return new Response('Unauthorized', { status: 401 });
  const event = request.headers.get('x-github-event');
  if (event === 'ping') return new Response('pong');
  if (env.ENABLED !== 'true') return new Response('Reviewer disabled', { status: 503 });
  const data = JSON.parse(body);
  if (data.repository?.full_name !== REPO || data.installation?.id !== INSTALLATION_ID) return new Response('Ignored', { status: 202 });
  if (event === 'pull_request' && ['opened', 'synchronize', 'reopened', 'ready_for_review', 'edited'].includes(data.action)) {
    if (!Number.isSafeInteger(data.number) || data.number < 1) return new Response('Invalid PR', { status: 400 });
    await enqueue(env, data.number);
  } else if (event === 'issue_comment' && data.action === 'created' && data.issue?.pull_request && data.comment?.user?.id === JAY_ID) {
    const command = /^\/review(?:\s+scope\s+([a-f0-9]{40})\s+([^\n]{1,2000}))?\s*$/.exec(data.comment.body);
    if (command) {
      const pr = snapshot(await read(env, `/pulls/${data.issue.number}`));
      if (!command[1] || command[1] === pr.head) await enqueue(env, pr.number, command[2] ? `jeqcho explicitly decided scope for ${pr.head}: ${command[2]}` : '', String(data.comment.id));
    }
  }
  return new Response('Accepted', { status: 202 });
}

export class ReviewWorkflow extends WorkflowEntrypoint<ReviewerEnv, { id: string; inspectOnly?: boolean; inspectOutput?: boolean; inspectQueue?: boolean; recoverSaved?: boolean; pauseReviews?: boolean }> {
  async run(event: WorkflowEvent<{ id: string; inspectOnly?: boolean; inspectOutput?: boolean; inspectQueue?: boolean; recoverSaved?: boolean; pauseReviews?: boolean }>, step: WorkflowStep) {
    const ledger = () => this.env.LEDGER.getByName('budget');
    if (event.payload.pauseReviews !== undefined) {
      // Operator-only management invocation. Never taken from a GitHub webhook.
      await ledger().pauseReviewQueue(event.payload.pauseReviews);
      const owner = (await ledger().queueState()).owner;
      if (event.payload.pauseReviews && owner) {
        const execution = await ledger().execution(owner);
        if (execution) {
          await ledger().closeSession(execution.token);
          await this.env.RUNNER.cleanup(execution.sandbox);
          await ledger().sandboxCleaned(owner);
        }
        await ledger().finish(owner, 'security_stopped');
        await ledger().releaseReviewSlot(owner);
      }
      return ledger().queueState();
    }
    if (event.payload.inspectQueue === true) return ledger().queueState();
    const job: Job | null = JSON.parse(await step.do('load job', async () => JSON.stringify(await ledger().job(event.payload.id))));
    if (!job) throw new Error('unknown_job');
    // Cloudflare management API only; never accepted from a webhook or PR text.
    if (event.payload.inspectOnly === true) return ledger().costs(job.id);
    if (event.payload.inspectOutput === true) {
      const raw = await ledger().runOutput(job.id);
      return JSON.stringify({ output: raw === null ? null : JSON.parse(raw), cost_summary: await ledger().costs(job.id) });
    }
    if (event.payload.recoverSaved === true) {
      // Management API only. GitHub /review always requests a fresh model run.
      const recover = await step.do('recover held saved output', async () => {
        if (this.env.ENABLED !== 'true') return false;
        if (!current(job, snapshot(await read(this.env, `/pulls/${job.pr}`)))) return false;
        return ledger().recoverSavedReview(job.id);
      });
      if (!recover) return;
    }
    try {
      if (this.env.ENABLED !== 'true') throw new Error('reviewer_disabled');
      await step.do('record workflow instance', () => ledger().workflowInstance(job.id, event.instanceId));
      if (!await step.do('publish queued check', () => this.env.PUBLISHER.publish(job, null, 'queued'))) {
        await step.do('mark stale before admission', async () => {
          await ledger().finish(job.id, 'stale');
          await enqueue(this.env, job.pr);
        });
        return;
      }
      // Waiting is durable Workflow sleep, not a running container or model call.
      // The SQLite ledger is shared by webhooks, manual commands and recovery.
      let admitted = false;
      for (let turn = 0; turn < 300; turn++) {
        const admission = await step.do(`queue admission ${turn}`, async () => {
          const live = snapshot(await read(this.env, `/pulls/${job.pr}`));
          if (!current(job, live)) {
            await ledger().finish(job.id, 'stale');
            await enqueue(this.env, job.pr);
            return 'obsolete';
          }
          return ledger().claimReviewSlot(job.id);
        });
        if (admission === 'obsolete') return;
        if (admission === 'acquired') { admitted = true; break; }
        await step.sleep(`wait for review slot ${turn}`, '1 minute');
      }
      // Bound Workflow history for arbitrarily long queues. Reconciliation starts
      // another waiting instance with the same job and FIFO position, without spend.
      if (!admitted) return;
      const started = await step.do('start', async () => {
        if (!await ledger().startReview(job.id)) return false;
        await ledger().workflowInstance(job.id, event.instanceId);
        if (!await this.env.PUBLISHER.publish(job, null, 'started')) throw new Error('stale_revision');
        return true;
      });
      if (started === false) return;
      const result = await runReview(this.env, job, step);
      await step.do('save verdict', () => ledger().finish(job.id, 'publishing', JSON.stringify(result)));
      const published = await step.do('publish verdict', () => this.env.PUBLISHER.publish(job, result));
      if (!published) await step.do('mark stale', () => ledger().finish(job.id, 'stale'));
      else if (result.verdict === 'APPROVE') await step.do('await required CI', () => ledger().finish(job.id, 'approved', JSON.stringify(result)));
      else await step.do('mark delivered', () => ledger().notified(job.id));
      if (await ledger().warningNeeded()) {
        if (await step.do('publish budget warning', () => this.env.PUBLISHER.publish(job, null, 'budget-warning'))) await step.do('record budget warning', () => ledger().warningSent());
      }
    } catch (error) {
      const saved = await ledger().job(job.id);
      if (saved?.status === 'stale') return;
      if (saved?.result && ['publishing', 'approved', 'done'].includes(saved.status)) {
        console.error(JSON.stringify({ event: 'review_publication_deferred', job: job.id }));
        return; // Keep the validated verdict for the reconciler; do not overwrite it with a hold.
      }
      if (transientLedgerError(error) && !await ledger().execution(job.id)) {
        // No model process exists. Preserve FIFO position for scheduled recovery.
        await step.do('defer interrupted admission', () => ledger().finish(job.id, 'queued'));
        console.error(JSON.stringify({ event: 'review_admission_deferred', job: job.id }));
        return;
      }
      // Never log external response bodies, prompts, code, headers or credentials.
      console.error(JSON.stringify({ job: job.id, status: 'held', error: error instanceof Error && /^[a-z_]+$/.test(error.message) ? error.message : 'review_failed' }));
      const execution = await ledger().execution(job.id);
      if (execution && await ledger().runOutput(job.id) === null && Date.now() - execution.started < 22 * 60_000) {
        // Leave the independent process/callback alive. The reconciler resumes
        // polling this exact execution; prepareExecution never charges it twice.
        await step.do('record recovery pending', () => ledger().finish(job.id, 'recovering'));
        return;
      }
      const details = { ...holdReason(error), cost_summary: await ledger().costs(job.id) };
      await step.do('record pending hold', () => ledger().finish(job.id, 'holding', JSON.stringify(details)));
      const delivered = await step.do('publish hold', () => this.env.PUBLISHER.publish(job, details, 'held'));
      await step.do('record hold delivery', () => ledger().finish(job.id, delivered ? 'held' : 'stale', JSON.stringify(details)));
    } finally {
      // releaseReviewSlot refuses to release an execution whose cleanup is uncertain.
      await step.do('release review slot', () => ledger().releaseReviewSlot(job.id));
    }
  }
}

async function scheduled(env: ReviewerEnv) {
  if (env.ENABLED !== 'true') return;
  const ledger = () => env.LEDGER.getByName('budget');
  // Also recover cleanup after a completed/held workflow, which is no longer in
  // pending(). Release only after destroy succeeds; a crash retains ownership.
  const owner = (await ledger().queueState()).owner;
  if (owner) {
    try {
      const job = await ledger().job(owner);
      const execution = await ledger().execution(owner);
      if (execution && (await ledger().runOutput(owner) !== null || Date.now() - execution.started >= 22 * 60_000)) {
        await ledger().closeSession(execution.token);
        await env.RUNNER.cleanup(execution.sandbox);
        await ledger().sandboxCleaned(owner);
        await ledger().releaseReviewSlot(owner);
      } else if (!execution && job && !['queued', 'running', 'recovering'].includes(job.status)) {
        await ledger().releaseReviewSlot(owner);
      }
    } catch { console.error(JSON.stringify({ job: owner, status: 'queue_cleanup_deferred' })); }
  }
  for (const job of await ledger().pending()) {
    try {
      const info = snapshot(await read(env, `/pulls/${job.pr}`));
      if (info.state !== 'open' || info.draft) { await ledger().finish(job.id, 'stale'); await ledger().releaseReviewSlot(job.id); continue; }
      if (!current(job, info)) {
        await ledger().finish(job.id, 'stale');
        await ledger().releaseReviewSlot(job.id);
        await enqueue(env, job.pr);
        continue;
      }
      if (job.status === 'holding' && job.result) {
        const delivered = await env.PUBLISHER.publish(job, JSON.parse(job.result), 'held');
        await ledger().finish(job.id, delivered ? 'held' : 'stale', job.result);
      } else if (job.status === 'recovering') {
        await env.REVIEW.create({ id: `${job.id}-recover-${Math.floor(Date.now() / 600000)}`, params: { id: job.id } });
      } else if (job.status === 'publishing' && job.result) {
        const result = JSON.parse(job.result);
        if (await env.PUBLISHER.publish(job, result)) {
          if (validateReview(result).verdict === 'APPROVE') await ledger().finish(job.id, 'approved', job.result);
          else await ledger().notified(job.id);
        }
      } else if (job.status === 'approved' && job.result && await ciGreen(p => read(env, p), job.head)) {
        if (await env.PUBLISHER.publish(job, JSON.parse(job.result))) await ledger().notified(job.id);
      } else if (job.status === 'queued' || job.status === 'running') {
        let instance;
        const instanceId = await ledger().workflowInstance(job.id);
        try { instance = await env.REVIEW.get(instanceId); await instance.status(); }
        catch { await env.REVIEW.create({ id: instanceId, params: { id: job.id } }); continue; }
        const status = await instance.status();
        if (['errored', 'terminated', 'complete'].includes(status.status)) {
          if (await ledger().execution(job.id)) { await ledger().finish(job.id, 'recovering'); continue; }
          if (job.status === 'queued') {
            // A waiting Workflow can be interrupted without ever using the slot.
            // Preserve its FIFO position and resume with no spending.
            await env.REVIEW.create({ id: `${job.id}-queue-${Math.floor(Date.now() / 600000)}`, params: { id: job.id } });
            continue;
          }
          const details = { code: 'codex_review_incomplete', cost_summary: await ledger().costs(job.id) };
          await ledger().finish(job.id, 'holding', JSON.stringify(details));
          await ledger().releaseReviewSlot(job.id);
          const delivered = await env.PUBLISHER.publish(job, details, 'held');
          await ledger().finish(job.id, delivered ? 'held' : 'stale', JSON.stringify(details));
        }
      }
    } catch { console.error(JSON.stringify({ job: job.id, status: 'reconcile_failed' })); }
  }
}

export default {
  async fetch(request: Request, env: ReviewerEnv): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/health' && request.method === 'GET') return Response.json({ service: 'inspect-robots-reviewer', mode: env.MODE, enabled: env.ENABLED === 'true', policy: POLICY_VERSION });
    if (url.pathname !== '/webhook' || request.method !== 'POST') return new Response('Not found', { status: 404 });
    try { return await handleWebhook(request, env); }
    catch { return new Response('Webhook processing failed; delivery may be retried', { status: 500 }); }
  },
  async scheduled(_controller: ScheduledController, env: ReviewerEnv): Promise<void> { await scheduled(env); },
} satisfies ExportedHandler<ReviewerEnv>;
