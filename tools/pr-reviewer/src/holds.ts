import { z } from 'zod';
import type { Job } from './common';

const reasons = {
  codex_review_incomplete: ['The Codex review session ended without a complete verdict.', 'Inspect its run status and remaining budget before rerunning. No partial output has been treated as approval.'],
  sandbox_execution_failed: ['The isolated Codex environment could not finish the review.', 'Inspect the private sandbox run, fix the environment issue, then rerun.'],
  file_too_large: ['A file exceeds the reviewer’s 100,000-byte per-file limit.', 'Add bounded large-file handling to the reviewer, then rerun with `/review`. Repeating the unchanged run will hit the same limit.'],
  file_not_inspectable: ['GitHub returned a file in a format the reviewer cannot inspect.', 'Check the affected files and add supported inspection before rerunning.'],
  uninspectable_diff: ['GitHub omitted a required text diff, and the file could not be verified as empty.', 'Inspect the omitted diff and resolve the coverage gap before rerunning.'],
  binary_file: ['A required file contains binary data that the text reviewer cannot inspect.', 'Arrange a manual review of the binary change and resolve the inspection gap.'],
  review_too_large: ['The PR exceeds the reviewer’s file-count or total-context limit.', 'Add bounded handling for this PR’s size or review it manually.'],
  context_too_large: ['The review context exceeds the 200,000-token limit.', 'Reduce duplicated review context or review the PR manually.'],
  insufficient_run_budget: ['A fresh review needs at least $2 of remaining allowance; this revision has less.', 'The service did not start a sandbox or make a model call. Review manually or explicitly authorize a budget adjustment before retrying.'],
  budget_exhausted: ['The remaining review budget cannot cover another model call.', 'Check the per-head, per-PR and monthly spending limits before requesting another run.'],
  budget_or_duplicate_reservation: ['A model-call reservation was declined by the budget or duplicate-call guard.', 'Inspect the existing run and spending ledger before retrying; an earlier call may already be charged.'],
  billing_hold: ['The reviewer paused spending after usage exceeded a reservation.', 'Reconcile the spending ledger with provider usage before resuming.'],
  incomplete_model_response: ['The model did not return a complete review.', 'Check the run and any incurred usage before requesting another review.'],
  model_timeout: ['The model did not finish within the review time limit.', 'Check provider status and any incurred usage before requesting another review.'],
  review_round_limit: ['The reviewer exhausted its allowed inspection rounds without a final verdict.', 'Inspect the run’s evidence requests before rerunning or review the PR manually.'],
  workflow_interrupted: ['Cloudflare interrupted the review workflow before a recoverable result was stored.', 'Inspect the execution record before requesting another paid review.'],
  review_launch_uncertain: ['The reviewer could not confirm whether its process started.', 'Inspect the existing sandbox execution; do not start a duplicate review.'],
  review_failed: ['The reviewer encountered a service or validation error and could not finish.', 'Inspect the private service logs for the run reference below, fix the failure, then rerun with `/review`.'],
} as const;
const schema = z.object({
  code: z.enum(Object.keys(reasons) as [keyof typeof reasons, ...(keyof typeof reasons)[]]),
  path: z.string().regex(/^[a-zA-Z0-9_./ -]{1,500}$/).optional(),
  size: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).optional(),
});

// Only allowlisted diagnostics cross into public comments. Never forward raw
// exception text, provider response bodies, credentials, or model output.
export function holdReason(error: unknown) {
  if (error instanceof Error && error.name === 'WorkflowInternalError') return { code: 'workflow_interrupted' as const };
  const message = error instanceof Error ? error.message : '';
  if (message.startsWith('file_too_large:')) {
    try {
      const parsed = schema.safeParse({ ...JSON.parse(message.slice('file_too_large:'.length)), code: 'file_too_large' });
      if (parsed.success) return parsed.data;
    } catch { /* Fall back to the safe reason without file metadata. */ }
    return { code: 'file_too_large' as const };
  }
  const parsed = schema.safeParse({ code: message });
  return parsed.success ? parsed.data : { code: 'review_failed' as const };
}

export function renderHold(job: Job, details: unknown): string {
  const parsed = schema.safeParse(details);
  const reason = parsed.success ? parsed.data : { code: 'review_failed' as const };
  const [summary, next] = reasons[reason.code];
  const specific = reason.code === 'file_too_large' && reason.path && reason.size !== undefined
    ? `File \`${reason.path}\` is ${reason.size.toLocaleString('en-US')} bytes; the reviewer’s per-file limit is 100,000 bytes.` : summary;
  return `**REQUIRE_REVIEWER**. No verdict issued. ${specific}\n\n**Next step:** @jeqcho, ${next.charAt(0).toLowerCase()}${next.slice(1)}\n\nThis is a limitation or failure of the reviewer, not a finding against the contribution. No approval, merge or closure recommendation was issued.\n\nCommit: \`${job.head}\`\nRun reference: \`${job.id}\`. Reason: \`${reason.code}\`.`;
}
