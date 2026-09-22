import { z } from 'zod';

export const REPO = 'robocurve/inspect-robots';
export const APP_ID = 5012304;
export const INSTALLATION_ID = 163290338;
export const JAY_ID = 42904912;
export const CHECK_NAME = 'Independent PR review';
export const MODEL = 'gpt-6-astra';
export const SHA = /^[a-f0-9]{40}$/;
export const LIMITS = { review: 5_000_000, pr: 15_000_000, month: 200_000_000, warn: 160_000_000 };
export const POLICY_VERSION = '3';

export const ReviewSchema = z.object({
  worthwhile: z.enum(['YES', 'NO', 'NEED_REVIEWER']),
  scope: z.enum(['ESTABLISHED', 'NEED_REVIEWER']).describe('Established project scope does not require a separate approval comment. NEED_REVIEWER requires a concrete product, maintenance or design decision, not merely absent prior approval.'),
  verdict: z.enum(['APPROVE', 'REQUEST_CHANGES', 'ESCALATE', 'REQUIRE_REVIEWER']),
  recommended_action: z.enum(['MERGE', 'REVISE', 'CLOSE', 'NEEDS_DECISION', 'COMPLETE_REVIEW']),
  rationale: z.string().max(360).describe('TL;DR in one or two short sentences: what this PR changes and the main reason for the verdict. No background narrative or repeated verdict label.'),
  blockers: z.array(z.object({ file: z.string(), line: z.number().int(), trigger: z.string(), expected: z.string(), actual: z.string(), impact: z.string(), fix: z.string() })),
  contract_and_test_review: z.string(),
  checks: z.array(z.string()),
  limitations: z.array(z.string()).describe('For each material gap, identify the exact files, behavior or checks not verified, why, and the next verification step. Separate incomplete inspection from unavailable hardware or services.'),
  sufficient_review: z.boolean(),
  decision_needed: z.string().max(500).describe('For escalation, the concrete decision and options in at most two short sentences. Otherwise empty.'),
  body: z.string().describe('Additional useful evidence only. Leave empty when the summary, findings and checks already cover the review.'),
});
export type Review = z.infer<typeof ReviewSchema>;
export const CostSummary = z.object({
  modelMicros: z.number().int().nonnegative(), reservedMicros: z.number().int().nonnegative(),
  sandboxMicros: z.number().int().nonnegative(), headSpentMicros: z.number().int().nonnegative(),
  headLimitMicros: z.number().int().positive(), remainingMicros: z.number().int().nonnegative(),
  modelCalls: z.number().int().nonnegative(),
});
export function renderCost(value: unknown): string {
  const c = CostSummary.parse(value);
  const money = (n: number) => `$${(n / 1_000_000).toFixed(3)}`;
  return `\n\nBudget: this run booked ${money(c.modelMicros)} for ${c.modelCalls} model calls, plus ${money(c.sandboxMicros)} sandbox allowance${c.reservedMicros ? ` and ${money(c.reservedMicros)} unresolved model reservations` : ''}. This revision has used ${money(c.headSpentMicros)} of ${money(c.headLimitMicros)} across all runs; ${money(c.remainingMicros)} remains under all spending caps. These are conservative ledger amounts, not an invoice.`;
}
export const ExecutionRecords = z.array(z.object({ revision: z.string().regex(/^[a-f0-9]{40}$/), command: z.string().max(12000), exitCode: z.number().int().nullable(), limit: z.string().nullable() })).max(100);
export const RunOutput = z.object({ exitCode: z.number().int(), failure: z.string().max(100).nullable().optional(), review: z.unknown(), executions: ExecutionRecords.default([]) });
export type Execution = { token: string; checkpointToken: string; sandbox: string; started: number; mergeBase: string };
export type Snapshot = { number: number; head: string; base: string; title: string; body: string; draft: boolean; state: string; author: string };
export type Job = { id: string; pr: number; head: string; base: string; scope: string; status: string; result: string | null; notified: number; created: number };

export function validateReview(value: unknown): Review {
  // Preserve saved reviews from before the public status rename.
  const candidate = value && typeof value === 'object' && 'verdict' in value && value.verdict === 'INCOMPLETE'
    ? { ...value, verdict: 'REQUIRE_REVIEWER' } : value;
  const r = ReviewSchema.parse(candidate);
  if (JSON.stringify(r).length > 24000 || !r.rationale.trim()) throw new Error('invalid_review');
  if (r.verdict === 'APPROVE' && (r.worthwhile !== 'YES' || r.scope !== 'ESTABLISHED' || !r.sufficient_review || r.blockers.length || r.recommended_action !== 'MERGE' || r.decision_needed.trim())) throw new Error('inconsistent_approval');
  if (r.verdict === 'REQUEST_CHANGES' && (!r.blockers.length || r.recommended_action !== 'REVISE' || r.scope !== 'ESTABLISHED' || r.worthwhile !== 'YES' || !r.sufficient_review)) throw new Error('inconsistent_changes');
  if (r.verdict === 'ESCALATE' && (!r.decision_needed.trim() || !['CLOSE', 'NEEDS_DECISION'].includes(r.recommended_action))) throw new Error('inconsistent_escalation');
  if (r.verdict === 'REQUIRE_REVIEWER' && (r.sufficient_review || r.recommended_action !== 'COMPLETE_REVIEW' || r.decision_needed.trim() || !r.limitations.length)) throw new Error('inconsistent_incomplete');
  for (const b of r.blockers) if (b.line < 1 || !b.file || !b.trigger || !b.expected || !b.actual || !b.fix) throw new Error('unsupported_blocker');
  return r;
}

export function snapshot(pr: Record<string, any>): Snapshot {
  const result = { number: pr.number, head: pr.head?.sha, base: pr.base?.sha, title: pr.title ?? '', body: pr.body ?? '', draft: !!pr.draft, state: pr.state, author: pr.user?.login ?? '' };
  if (!Number.isSafeInteger(result.number) || result.number < 1 || !SHA.test(result.head) || !SHA.test(result.base)) throw new Error('invalid_snapshot');
  return result;
}

export function current(a: Pick<Snapshot, 'head' | 'base'>, b: Snapshot): boolean {
  return b.state === 'open' && !b.draft && a.head === b.head && a.base === b.base;
}

export async function digest(text: string): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))), x => x.toString(16).padStart(2, '0')).join('');
}

export async function verifySignature(body: string, signature: string | null, secret: string): Promise<boolean> {
  if (!secret || !signature || !/^sha256=[a-f0-9]{64}$/.test(signature)) return false;
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['verify']);
  const bytes = Uint8Array.from(signature.slice(7).match(/../g)!, x => parseInt(x, 16));
  return crypto.subtle.verify('HMAC', key, bytes, new TextEncoder().encode(body));
}

export async function boundedText(response: Response | Request, limit = 2_000_000): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) return '';
  let size = 0;
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > limit) { await reader.cancel(); throw new Error('payload_too_large'); }
    chunks.push(value);
  }
  const all = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) { all.set(c, offset); offset += c.byteLength; }
  return new TextDecoder().decode(all);
}

// Untrusted prose cannot inject mentions, hidden markup or remote media into a bot comment.
export function publicText(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/https?:\/\/\S+/g, '[link omitted]')
    .replace(/@/g, '@\u200b').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u202a-\u202e\u2066-\u2069]/g, '').replace(/—/g, ',');
}

export function renderReview(job: Job, review: Review, ciGreen: boolean, executions: z.infer<typeof ExecutionRecords> = [], author = ''): string {
  const incompleteWithBugs = review.verdict === 'REQUIRE_REVIEWER' && review.blockers.length > 0;
  let body = `**${review.verdict}**. ${incompleteWithBugs ? `Fix ${review.blockers.length === 1 ? 'the confirmed blocking defect' : `the ${review.blockers.length} confirmed blocking defects`} below; review is also incomplete. ` : ''}${publicText(review.rationale).replace(/\s+/g, ' ').trim()}`;
  if (review.verdict === 'APPROVE') body += ciGreen ? '\n\n@jeqcho, review approved and ci-ok is green for this revision. Please review and merge if you agree.' : '\n\n@jeqcho, review approved. Waiting for ci-ok before requesting a merge.';
  else if (review.verdict === 'ESCALATE') body += `\n\n@jeqcho, ${review.recommended_action === 'CLOSE' ? 'please decide whether to close this PR. ' : 'your decision is needed. '}${publicText(review.decision_needed).replace(/\s+/g, ' ').trim()}`;
  else if (review.verdict === 'REQUIRE_REVIEWER') {
    if (incompleteWithBugs) {
      body += '\n\n@jeqcho, please coordinate fixes for the confirmed bugs below, then complete the remaining validation. Do not merge until both are addressed.';
      for (const b of review.blockers) body += `\n\n- ${publicText(b.file)}:${b.line}: ${publicText(b.actual)} Fix: ${publicText(b.fix)}`;
    } else body += '\n\n@jeqcho, please complete or delegate the remaining checks below before deciding whether to merge. The review is incomplete.';
    body += '\n\n**Remaining checks:**\n\n' + review.limitations.map(item => `- [ ] ${publicText(item)}`).join('\n');
  }
  else {
    // Only a login supplied by the trusted publisher may become a live mention.
    const authorAvailable = /^[a-z\d](?:[a-z\d-]{0,37}[a-z\d])?$/i.test(author) && author.toLowerCase() !== 'ghost';
    body += `\n\n${authorAvailable ? `@${author}, please address` : '@jeqcho, please coordinate fixes for'} the ${review.blockers.length === 1 ? 'blocking finding' : `${review.blockers.length} blocking findings`} detailed below.`;
  }
  body += `\n\n<details>\n<summary>Review details, findings and checks</summary>\n\nAutomated independent review of commit \`${job.head}\` (base \`${job.base}\`).\n\nWorthwhile: ${review.worthwhile}\nScope: ${review.scope}\nRecommendation: ${review.recommended_action}\n\n${review.body.trim() ? publicText(review.body) + '\n\n' : ''}Contracts and tests: ${publicText(review.contract_and_test_review)}`;
  for (const b of review.blockers) body += `\n\n${publicText(b.file)}:${b.line}: ${publicText(b.trigger)}\nExpected: ${publicText(b.expected)}\nObserved from code: ${publicText(b.actual)}\nImpact: ${publicText(b.impact)}\nSuggested fix: ${publicText(b.fix)}`;
  body += `\n\nChecks completed:\n\n${review.checks.map(item => `- ${publicText(item)}`).join('\n')}\n\n${executions.length ? 'Sandbox execution records (CI is checked separately):\n' : 'No sandbox commands were executed; CI is checked separately.'}`;
  for (const execution of executions.slice(0, 10)) body += `\n- Revision ${execution.revision.slice(0, 12)}, exit ${execution.exitCode ?? 'unknown'}${execution.limit ? `, ${publicText(execution.limit)}` : ''}: ${publicText(execution.command.slice(0, 500)).replace(/\n/g, ' ')}`;
  if (review.limitations.length && review.verdict !== 'REQUIRE_REVIEWER') body += `\n\nLimitations:\n\n${review.limitations.map(item => `- ${publicText(item)}`).join('\n')}`;
  return body + '\n\n</details>';
}
