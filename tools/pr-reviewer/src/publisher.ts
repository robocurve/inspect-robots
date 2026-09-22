import { WorkerEntrypoint } from 'cloudflare:workers';
import { APP_ID, CHECK_NAME, ExecutionRecords, REPO, SHA, current, renderCost, renderReview, snapshot, validateReview, type Job } from './common';
import { allowedRead, ciGreen, github, installationToken } from './github';
import { renderHold } from './holds';

export class GithubPublisher extends WorkerEntrypoint<PublisherEnv> {
  async read(path: string): Promise<string> {
    if (!allowedRead(path)) throw new Error('read_not_allowed');
    try {
      return JSON.stringify(await github(await installationToken(this.env, false), `/repos/${REPO}${path}`));
    } catch (error) {
      console.error(JSON.stringify({ event: 'github_read_failed', name: error instanceof Error ? error.name : 'unknown', code: error instanceof Error && /^github_http_\d+$/.test(error.message) ? error.message : 'reader_failed' }));
      throw error;
    }
  }
  async publish(job: Job, result: unknown, notice?: 'queued' | 'started' | 'held' | 'budget-warning'): Promise<boolean> {
    if (!Number.isSafeInteger(job.pr) || job.pr < 1 || !SHA.test(job.head) || !SHA.test(job.base) || !/^[a-z0-9-]{1,100}$/.test(job.id)) throw new Error('invalid_publication');
    const token = await installationToken(this.env, true);
    const read = (p: string) => github(token, `/repos/${REPO}${p}`);
    const pr = snapshot(await read(`/pulls/${job.pr}`));
    if (!current(job, pr)) return false;
    let body: string;
    let conclusion: 'success' | 'failure' | 'action_required' | undefined;
    if (notice === 'queued') {
      body = `Independent review is queued for ${job.head}. It will start automatically when the review sandbox is free. No sandbox or model spending occurs while waiting.`;
    } else if (notice === 'started') {
      body = `Independent review is running for ${job.head}.`;
    } else if (notice === 'held') {
      body = renderHold(job, result);
      conclusion = 'action_required';
    } else if (notice === 'budget-warning') {
      body = '@jeqcho, this month’s review spending and outstanding reservations have reached $160 of the $200 budget. The service will pause when the remaining allowance is insufficient.';
      conclusion = 'action_required';
    } else {
      const review = validateReview(result);
      const executions = ExecutionRecords.parse(result && typeof result === 'object' && 'execution_records' in result ? result.execution_records : []);
      body = renderReview(job, review, await ciGreen(read, job.head), executions, pr.author);
      conclusion = review.verdict === 'APPROVE' ? 'success' : review.verdict === 'REQUEST_CHANGES' ? 'failure' : 'action_required';
    }
    if (result && typeof result === 'object' && 'cost_summary' in result) body += renderCost(result.cost_summary);
    // Re-check after reads. Check runs always attach to the exact reviewed head.
    if (!current(job, snapshot(await read(`/pulls/${job.pr}`)))) return false;
    if (notice !== 'budget-warning') {
      const runs = await read(`/commits/${job.head}/check-runs?check_name=${encodeURIComponent(CHECK_NAME)}&per_page=100`);
      const previous = runs.check_runs?.find((r: any) => r.app?.id === APP_ID && r.external_id === job.id);
      const payload = { name: CHECK_NAME, ...(!previous ? { head_sha: job.head } : {}), external_id: job.id, status: conclusion ? 'completed' : notice === 'queued' ? 'queued' : 'in_progress', ...(conclusion ? { conclusion } : {}), output: { title: notice === 'queued' ? 'Independent review queued' : notice === 'started' ? 'Independent review running' : 'Independent review result', summary: body } };
      await github(token, `/repos/${REPO}/check-runs${previous ? `/${previous.id}` : ''}`, previous ? 'PATCH' : 'POST', payload);
    }
    if (notice === 'started' || notice === 'queued') return true;
    const marker = `<!-- inspect-robots-review:${job.id}${notice === 'budget-warning' ? ':budget' : ''} -->`;
    // One comment per snapshot. Old snapshots never overwrite newer review comments.
    let previous;
    for (let page = 1; page <= 10; page++) {
      const comments = await read(`/issues/${job.pr}/comments?per_page=100&page=${page}`);
      previous = comments.find((c: any) => c.performed_via_github_app?.id === APP_ID && c.body?.includes(marker));
      if (previous || comments.length < 100) break;
      if (page === 10) throw new Error('comment_pagination_limit');
    }
    if (!current(job, snapshot(await read(`/pulls/${job.pr}`)))) return false;
    const text = `${body}\n\n${marker}`;
    if (previous?.body === text) return true;
    await github(token, `/repos/${REPO}/${previous ? `issues/comments/${previous.id}` : `issues/${job.pr}/comments`}`, previous ? 'PATCH' : 'POST', { body: text });
    return true;
  }
}

export default { fetch() { return new Response('Not found', { status: 404 }); } };
