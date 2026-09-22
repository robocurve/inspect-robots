import { env, createExecutionContext, evictDurableObject } from 'cloudflare:test';
import { describe, expect, it, vi } from 'vitest';
import { exportPKCS8, generateKeyPair } from 'jose';
import { current, publicText, renderCost, renderReview, validateReview, verifySignature, type Job, type Review } from '../src/common';
import { allowedRead, ciGreen, github } from '../src/github';
import { collectContext, readFile, safePath } from '../src/context';
import { GithubPublisher } from '../src/publisher';
import { holdReason, renderHold } from '../src/holds';
import { handleWebhook } from '../src/worker';
import type { ReviewLedger } from '../src/ledger';

declare module 'cloudflare:test' { interface ProvidedEnv { LEDGER: DurableObjectNamespace<ReviewLedger> } }
const head = 'a'.repeat(40), base = 'b'.repeat(40);
const job: Job = { id: 'revision-1', pr: 9, head, base, scope: '', status: 'running', result: null, notified: 0, created: 1 };
const approval: Review = { worthwhile: 'YES', scope: 'ESTABLISHED', verdict: 'APPROVE', recommended_action: 'MERGE', rationale: 'Confirmed bug fix.', blockers: [], contract_and_test_review: 'Tests preserve contracts.', checks: ['Source and tests inspected'], limitations: [], sufficient_review: true, decision_needed: '', body: 'The fix preserves the documented behavior.' };
const pr = { number: 9, head: { sha: head }, base: { sha: base }, state: 'open', draft: false, changed_files: 1, user: { login: 'contributor' }, title: 'Fix error', body: '' };

describe('review gates', () => {
  it('rejects approvals with unapproved scope, missing evidence, or blockers', () => {
    expect(validateReview(approval).verdict).toBe('APPROVE');
    for (const change of [{ scope: 'NEED_REVIEWER' }, { worthwhile: 'NO' }, { sufficient_review: false }, { recommended_action: 'CLOSE' }, { decision_needed: 'Choose API' }]) expect(() => validateReview({ ...approval, ...change })).toThrow();
  });
  it('accepts empty supplemental prose without requiring repeated evidence', () => {
    const review = validateReview({ ...approval, body: '' });
    const text = renderReview(job, review, true);
    expect(text).toMatch(/^\*\*APPROVE\*\*\. Confirmed bug fix\./);
    expect(text).toContain('Contracts and tests: Tests preserve contracts.');
    expect(text).not.toContain('\n\n\n');
    expect(() => validateReview({ ...review, rationale: ' ' })).toThrow('invalid_review');
    expect(() => validateReview({ ...review, sufficient_review: false })).toThrow('inconsistent_approval');
  });
  it('requires a concrete decision for escalation and a blocker for changes', () => {
    expect(() => validateReview({ ...approval, verdict: 'ESCALATE', recommended_action: 'CLOSE' })).toThrow();
    expect(() => validateReview({ ...approval, verdict: 'REQUEST_CHANGES', recommended_action: 'REVISE' })).toThrow();
  });
  it('only asks Jay to merge when this revision has green CI', () => {
    expect(renderReview(job, approval, false)).toContain('@jeqcho, review approved. Waiting for ci-ok before requesting a merge.');
    expect(renderReview(job, approval, false)).not.toContain('Please review and merge');
    const merged = renderReview(job, approval, true);
    expect(merged).toMatch(/^\*\*APPROVE\*\*\. Confirmed bug fix\./);
    expect(merged).toContain('@jeqcho');
    expect(merged.indexOf('@jeqcho')).toBeLessThan(merged.indexOf('<details>'));
    expect(merged.split(approval.rationale)).toHaveLength(2);
    expect(merged).toContain(approval.contract_and_test_review);
    const escalated = renderReview(job, { ...approval, verdict: 'ESCALATE', recommended_action: 'CLOSE', decision_needed: 'The existing plan excludes this dependency.' }, false);
    expect(escalated).toMatch(/^\*\*ESCALATE\*\*\./);
    expect(escalated).toContain('@jeqcho, please decide whether to close');
    expect(escalated.indexOf('The existing plan excludes this dependency.')).toBeLessThan(escalated.indexOf('<details>'));
  });
  it('separates incomplete technical review from a human decision', () => {
    const partial: Review = { ...approval, verdict: 'REQUIRE_REVIEWER', recommended_action: 'COMPLETE_REVIEW', sufficient_review: false, limitations: ['Inspect scorer.py lines 40-90; session ran out of time.'] };
    expect(validateReview(partial).verdict).toBe('REQUIRE_REVIEWER');
    const text = renderReview(job, partial, false);
    expect(text).toMatch(/^\*\*REQUIRE_REVIEWER\*\*\./);
    expect(text).toContain('@jeqcho, please complete or delegate');
    expect(text.split('<details>')[0]).toContain('**Remaining checks:**\n\n- [ ] Inspect scorer.py lines 40-90; session ran out of time.');
    expect(text).not.toContain('your decision is needed');
    for (const change of [{ sufficient_review: true }, { recommended_action: 'MERGE' }, { decision_needed: 'Run pytest' }, { limitations: [] }]) expect(() => validateReview({ ...partial, ...change })).toThrow();
  });
  it('renders saved legacy unfinished reviews with the new status and the same safeguards', () => {
    const legacy = { ...approval, verdict: 'INCOMPLETE', recommended_action: 'COMPLETE_REVIEW', sufficient_review: false, limitations: ['Remaining code needs inspection.'] };
    const review = validateReview(legacy);
    expect(review.verdict).toBe('REQUIRE_REVIEWER');
    expect(renderReview(job, review, false)).toMatch(/^\*\*REQUIRE_REVIEWER\*\*\./);
    expect(() => validateReview({ ...legacy, sufficient_review: true })).toThrow('inconsistent_incomplete');
  });
  it('leads incomplete reviews with confirmed bugs and fixes above the collapsed details', () => {
    const review: Review = { ...approval, verdict: 'REQUIRE_REVIEWER', recommended_action: 'COMPLETE_REVIEW', sufficient_review: false,
      blockers: [{ file: 'runner.py', line: 144, trigger: 'PR build executes', expected: 'Immutable evidence', actual: 'Build replaces the proposed code with base.', impact: 'Changes disappear.', fix: 'Protect canonical snapshots and their parent directories.' }],
      limitations: ['Run the integration suite.'] };
    const body = renderReview(job, validateReview(review), false, [], 'contributor');
    const top = body.split('<details>')[0];
    expect(top).toMatch(/^\*\*REQUIRE_REVIEWER\*\*\. Fix the confirmed blocking defect/);
    expect(top).toContain('@jeqcho, please coordinate fixes');
    expect(top).toContain('Build replaces the proposed code with base.');
    expect(top).toContain('Protect canonical snapshots and their parent directories.');
    expect(top).toContain('then complete the remaining validation');
    expect(top).toContain('**Remaining checks:**\n\n- [ ] Run the integration suite.');
    expect(top).not.toContain('please arrange completion');
  });
  it('tags the author for edits and falls back safely when no person can be mentioned', () => {
    const changes = { ...approval, verdict: 'REQUEST_CHANGES', recommended_action: 'REVISE', blockers: [{ file: 'x.py', line: 1, trigger: 'Empty input', expected: 'Valid output', actual: 'Crash', impact: 'Task fails', fix: 'Handle empty input' }] } as Review;
    for (const author of ['Sravanthi6m', 'a', 'test-author']) {
      const text = renderReview(job, validateReview(changes), true, [], author);
      expect(text).toMatch(/^\*\*REQUEST_CHANGES\*\*\./);
      expect(text).toContain(`@${author}, please address the blocking finding`);
      expect(text.indexOf(`@${author}`)).toBeLessThan(text.indexOf('<details>'));
      expect(text).not.toContain('@jeqcho');
    }
    for (const author of ['', 'ghost', 'dependabot[bot]', 'name @victim', 'org/team', 'a'.repeat(40)]) {
      const text = renderReview(job, changes, true, [], author);
      expect(text).toContain('@jeqcho, please coordinate fixes');
      expect(text).not.toContain('@victim');
    }
  });
  it('neutralizes untrusted mentions, links and hidden markup', () => {
    const output = publicText('<!-- hidden --><img src=x> @jeqcho ![secret](https://bad.test/key) https://bad.test');
    expect(output).not.toMatch(/@jeqcho|https:|<img|<!--/);
    expect(output).toContain('&lt;!-- hidden --&gt;');
    // Removing a nested tag can reconstruct another tag. Escape every delimiter instead.
    expect(publicText('<scr<script>ipt>alert(1)</script><!<!-- -->-->&#60;img src=x>')).not.toMatch(/[<>]/);
  });
  it('rejects old heads, changed bases, drafts and closed PRs', () => {
    const s = { number: 9, head, base, title: '', body: '', draft: false, state: 'open', author: '' };
    expect(current(job, s)).toBe(true);
    for (const change of [{ head: base }, { base: head }, { draft: true }, { state: 'closed' }]) expect(current(job, { ...s, ...change })).toBe(false);
  });
  it('does not accept a spoofed ci-ok or a green status on another head', async () => {
    const check = { name: 'ci-ok', head_sha: head, app: { slug: 'github-actions' }, status: 'completed', conclusion: 'success' };
    expect(await ciGreen(async () => ({ check_runs: [check] }), head)).toBe(true);
    expect(await ciGreen(async () => ({ check_runs: [{ ...check, app: { slug: 'other' } }] }), head)).toBe(false);
    expect(await ciGreen(async () => ({ check_runs: [{ ...check, head_sha: base }] }), head)).toBe(false);
  });
});

describe('budget ledger in the Workers runtime', () => {
  it('limits the authorized trial exception to its exact PR head and preserves the PR ceiling', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    const revision = '456-696fbaa9a00d7c345a81dd179fa10934f51ade89';
    expect(await ledger.remaining(revision, 456)).toBe(25_000_000);
    expect(await ledger.remaining(revision, 457)).toBe(5_000_000);
    expect(await ledger.remaining(`456-${head}`, 456)).toBe(5_000_000);
    const accepted = await Promise.all(Array.from({ length: 27 }, (_, i) => ledger.reserve(`trial-${i}`, revision, 456, 1_000_000)));
    expect(accepted.filter(Boolean)).toHaveLength(25);
    expect(await ledger.reserve('another-head', `456-${head}`, 456, 1)).toBe(false);
    expect(await ledger.reserve('pr-ceiling', `456-${base}`, 456, 1)).toBe(false);
  });
  it('retains the default lifetime cap for other PRs and the shared monthly cap', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    for (const [i, sha] of [head, base, 'c'.repeat(40)].entries()) expect(await ledger.reserve(`default-${i}`, `457-${sha}`, 457, 5_000_000)).toBe(true);
    expect(await ledger.reserve('default-overflow', `457-${'d'.repeat(40)}`, 457, 1)).toBe(false);
    const revision = '456-696fbaa9a00d7c345a81dd179fa10934f51ade89';
    expect(await ledger.reserve('authorized', revision, 456, 20_000_000)).toBe(true);
    for (let i = 0; i < 33; i++) expect(await ledger.reserve(`monthly-${i}`, `${1000 + i}-${head}`, 1000 + i, 5_000_000)).toBe(true);
    expect(await ledger.remaining(revision, 456)).toBe(0);
    expect(await ledger.reserve('monthly-overflow', revision, 456, 1)).toBe(false);
  });
  it('reports each run separately while retaining cumulative revision spending', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    await ledger.register(job);
    await ledger.reserve(`${job.id}-sandbox`, `9-${head}`, 9, 100000);
    await ledger.reserve(`${job.id}-codex-settled`, `9-${head}`, 9, 500000);
    await ledger.settle(`${job.id}-codex-settled`, 123456);
    await ledger.reserve(`${job.id}-codex-uncertain`, `9-${head}`, 9, 300000);
    await ledger.reserve('other-run-codex-settled', `9-${head}`, 9, 1000000);
    await evictDurableObject(ledger);
    const costs = await ledger.costs(job.id);
    expect(costs).toEqual({ modelMicros: 123456, reservedMicros: 300000, sandboxMicros: 100000, headSpentMicros: 1523456, headLimitMicros: 5000000, remainingMicros: 3476544, modelCalls: 2 });
    expect(renderCost(costs)).toContain('$0.123');
    expect(renderCost(costs)).toContain('$0.300 unresolved model reservations');
    expect(() => renderCost({ ...costs, modelMicros: -1 })).toThrow();
  });
  it('atomically limits concurrent reservations and disallows duplicate charges', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    const accepted = await Promise.all(Array.from({ length: 10 }, (_, i) => ledger.reserve(`charge-${i}`, 'revision', 1, 1_000_000)));
    expect(accepted.filter(Boolean)).toHaveLength(5);
    expect(await ledger.remaining('revision', 1)).toBe(0);
    expect(await ledger.reserve('charge-0', 'revision', 1, 1)).toBe(false);
    await ledger.settle('charge-0', 100_000);
    await ledger.settle('charge-0', 0); // settlement is idempotent
    expect(await ledger.remaining('revision', 1)).toBe(900_000);
  });
  it('enforces the PR cap across revisions and the monthly cap across PRs', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    for (let i = 0; i < 3; i++) expect(await ledger.reserve(`a-${i}`, `r-${i}`, 1, 5_000_000)).toBe(true);
    expect(await ledger.reserve('a-4', 'r-4', 1, 1)).toBe(false);
    for (let i = 0; i < 37; i++) expect(await ledger.reserve(`b-${i}`, `b-${i}`, i + 2, 5_000_000)).toBe(true);
    expect(await ledger.reserve('last', 'last', 100, 1)).toBe(false);
    expect(await ledger.warningNeeded()).toBe(true);
    await ledger.warningSent(); expect(await ledger.warningNeeded()).toBe(false);
  });
  it('keeps uncertain reservations and freezes inference on unexpected usage', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    await ledger.reserve('ambiguous', 'r', 1, 5_000_000);
    expect(await ledger.remaining('r', 1)).toBe(0);
    expect(await ledger.settle('ambiguous', 5_000_001)).toBe(false);
    expect(await ledger.billingHold()).toBe(true);
  });
  it('deduplicates webhook jobs', async () => {
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    expect(await ledger.register(job)).toBe(true);
    expect(await ledger.register(job)).toBe(false);
    expect((await ledger.pending()).length).toBe(1);
  });
  it('preserves reservations after eviction and resets only the monthly allowance', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-09-30T23:00:00Z'));
    const ledger = env.LEDGER.getByName(crypto.randomUUID());
    for (let i = 0; i < 40; i++) expect(await ledger.reserve(`c-${i}`, `r-${i}`, i + 1, 5_000_000)).toBe(true);
    await evictDurableObject(ledger);
    expect(await ledger.remaining('new', 100)).toBe(0);
    vi.setSystemTime(new Date('2026-10-01T01:00:00Z'));
    expect(await ledger.remaining('new', 100)).toBe(5_000_000);
    expect(await ledger.remaining('r-0', 1)).toBe(0);
  });
});

describe('untrusted input and complete context', () => {
  it('authenticates the original webhook body and rejects mutations', async () => {
    const body = '{"action":"opened"}', secret = 'test-secret';
    const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
    const sig = 'sha256=' + [...new Uint8Array(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(body)))].map(x => x.toString(16).padStart(2, '0')).join('');
    expect(await verifySignature(body, sig, secret)).toBe(true);
    expect(await verifySignature(body + ' ', sig, secret)).toBe(false);
    expect(await verifySignature(body, null, secret)).toBe(false);
  });
  it('has no general GitHub URL or mutation tool', () => {
    for (const p of ['/pulls/9', '/contents/src/main.py?ref=' + head, '/issues/9/comments']) expect(allowedRead(p)).toBe(true);
    for (const p of ['/pulls/9/merge', '/issues/comments/1', '/../../other', 'https://bad.test/', '/actions/runs/1/approve', '/contents/../secrets']) expect(allowedRead(p)).toBe(false);
    expect(safePath('../.env')).toBe(false);
  });
  it('rejects binary data and unsupported file paths before inference', async () => {
    await expect(readFile(async () => ({ type: 'file', encoding: 'base64', size: 1, content: btoa('\0') }), job, 'x', 'head')).rejects.toThrow('binary_file');
    await expect(readFile(async () => ({}), job, '../x', 'head')).rejects.toThrow('invalid_file_path');
  });
  it('bounds individual source transfers without rejecting a PR diff', async () => {
    await expect(readFile(async () => ({ type: 'file', encoding: 'base64', size: 1_000_001, content: '' }), job, 'uv.lock', 'head')).rejects.toThrow('source_transport_limit');
  });
});

describe('actionable hold notices', () => {
  it('preserves safe file details through an Error message round-trip', async () => {
    let failure: unknown;
    failure = new Error('file_too_large:{"path":"uv.lock","size":524568}');
    const body = renderHold(job, holdReason(failure));
    expect(body).toContain('`uv.lock` is 524,568 bytes');
    expect(body).toContain('limit is 100,000 bytes');
    expect(body).toContain('large-file handling');
    expect(body).not.toContain('check the reviewer service configuration');
  });
  it('distinguishes exhausted budgets from file limits', () => {
    const body = renderHold(job, holdReason(new Error('budget_exhausted')));
    expect(body).toContain('remaining review budget');
    expect(body).not.toContain('large-file');
    expect(body).not.toContain('no model charges');
  });
  it('never publishes arbitrary errors or unsafe diagnostic fields', () => {
    for (const error of [new Error('Authorization: Bearer sk-private'), new Error('file_too_large:{"path":"@victim <img>","size":42}'), new Error('file_too_large:malformed')]) {
      const body = renderHold(job, holdReason(error));
      expect(body).not.toMatch(/sk-private|@victim|<img>|malformed/);
      expect(body).toMatch(/^\*\*REQUIRE_REVIEWER\*\*\. No verdict issued\./);
    }
    expect(renderHold(job, { code: 'review_failed', path: '@victim' })).not.toContain('@victim');
  });
});

describe('publisher authority', () => {
  it('rejects redirects without forwarding credentials', async () => {
    const send = vi.fn(async (request: Request) => {
      expect(request.redirect).toBe('manual');
      return new Response(null, { status: 302, headers: { Location: 'https://untrusted.test/' } });
    });
    vi.stubGlobal('fetch', send);
    await expect(github('test-token', '/repos/robocurve/inspect-robots/pulls/9')).rejects.toThrow('github_http_302');
    expect(send).toHaveBeenCalledTimes(1);
  });
  it('only writes check runs and a courteous comment, and stops for a stale head', async () => {
    const { privateKey } = await generateKeyPair('RS256', { extractable: true });
    const publisher = new GithubPublisher(createExecutionContext(), { GITHUB_PRIVATE_KEY: await exportPKCS8(privateKey), GITHUB_APP_ID: '5012304', GITHUB_INSTALLATION_ID: '163290338' });
    const writes: { path: string; method: string; body: any }[] = [];
    let live = pr;
    vi.stubGlobal('fetch', vi.fn(async (url: string | Request, init?: RequestInit) => {
      const request = new Request(url, init);
      const path = new URL(request.url).pathname;
      expect(request.redirect).toBe('manual');
      if (path.endsWith('/access_tokens')) return Response.json({ token: 'test-installation-token' });
      if (request.method !== 'GET') { writes.push({ path, method: request.method, body: await request.json() }); return Response.json({ id: 123 }); }
      if (path.endsWith('/pulls/9')) return Response.json(live);
      if (path.endsWith('/check-runs')) return Response.json({ check_runs: [{ name: 'ci-ok', app: { slug: 'github-actions' }, head_sha: head, status: 'completed', conclusion: 'success' }] });
      if (path.endsWith('/comments')) return Response.json([]);
      throw new Error('unexpected endpoint');
    }));
    expect(await publisher.publish(job, approval)).toBe(true);
    expect(writes.map(w => w.path)).toEqual(['/repos/robocurve/inspect-robots/check-runs', '/repos/robocurve/inspect-robots/issues/9/comments']);
    expect(writes[1].body.body).toContain('@jeqcho');
    expect(writes[1].body.body).toMatch(/^\*\*APPROVE\*\*\./);
    expect(writes[1].body.body).toMatch(/<!-- inspect-robots-review:revision-1 -->$/);
    const changes = { ...approval, verdict: 'REQUEST_CHANGES', recommended_action: 'REVISE', blockers: [{ file: 'x.py', line: 1, trigger: 'Empty input', expected: 'Output', actual: 'Crash', impact: 'Failure', fix: 'Handle input' }], author: 'attacker' };
    expect(await publisher.publish(job, changes)).toBe(true);
    expect(writes[3].body.body).toContain('@contributor, please address');
    expect(writes[3].body.body).not.toContain('@attacker');
    live = { ...pr, head: { sha: base } };
    expect(await publisher.publish(job, approval)).toBe(false);
    expect(writes).toHaveLength(4);
  });
});
