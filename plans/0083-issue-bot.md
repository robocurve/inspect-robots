# Robocurve issue bot

## Outcome

New issues in robocurve/inspect-robots automatically receive an Astra/Codex
assessment. Confirmed serious bugs proceed through independently reviewed plans,
implementation, and independent code review. The bot publishes a draft PR, then
marks it ready only after its exact contents pass independent review. The PR
review bot then runs and routes bot-author actions to @jeqcho. Neither service
merges PRs or closes issues automatically.

## Architecture

Implement a separate tools/issue-bot package and three Cloudflare Workers:
robocurve-issue-bot (webhook, durable coordinator and budget gateway),
robocurve-issue-runner (one isolated container with a fixed-purpose supervisor), and
robocurve-issue-publisher (GitHub App credentials and bounded publication).
Do not modify or redeploy the existing PR reviewer while another agent fixes it.

The issue bot has its own GitHub App with issues:write, contents:write and
pull_requests:write on inspect-robots only, plus checks:read and metadata:read. A signed issues
webhook records issue-opened events; a maintainer-only /triage comment or private
management workflow can enqueue the selected existing issue. App registration
and installation may require the maintainer's browser. Do not replace the PR
reviewer's App webhook or use a human-authored PR as a stand-in for bot identity.

## Durable state machine

Persist signed event acceptance and deduplicate by semantic issue revision. Pin a base
main SHA and issue title/body/user. Hash title/body/state, not updated_at: the
bot's own comments must not invalidate its work. A durable FIFO admits one
issue job at a time before starting a container. No inference while waiting.
Persist each stage's request, launch claim, scoped model capability, terminal
output and publication intent. Recovery polls/replays an existing stage; it
never blindly restarts paid inference. Release the slot only after confirmed
container cleanup. A periodic reconciliation sweep recovers jobs and outbox
deliveries, including failure notices. Ambiguous failures retain cost reservations.

Each stage is a fresh ephemeral Codex exec using gpt-6-astra with high reasoning:

1. TRIAGE: independently inspect pinned source and issue evidence. Emit
   CONFIRMED, NEEDS_INFO, NOT_REPRODUCED, DUPLICATE, FIX_PROPOSED or REQUIRE_REVIEWER.
   DUPLICATE refers to another issue; FIX_PROPOSED refers to an existing open fix PR.
   CONFIRMED requires concrete reproduction/evidence, severity and bounded scope.
   Only serious confirmed bugs (data loss, safety failure, security defect or
   substantial broken supported behavior) qualify for automatic implementation.
   Feature requests, ambiguous contract changes and hardware-only unverifiable
   behavior ask @jeqcho for judgment. Information requests tag the actual issue
   author; bot/deleted authors fall back to @jeqcho.
2. PLAN: draft implementation plan with regression tests and affected contracts.
3. PLAN_REVIEW: new independent agent accepts or requests revisions. Feed
   concrete feedback to a fresh planner until accepted, within spending/round caps.
4. IMPLEMENT: a new agent edits a writable source copy using the approved plan.
5. CODE_REVIEW: another new agent inspects the exact immutable resulting files,
   verifies regression and required checks. Changes requested return to a fresh
   implementer and another fresh reviewer. Missing validation cannot approve.
6. PUBLISH: trusted publisher creates one bot branch and draft PR idempotently,
   tying its exact tree/base to the accepted review. Publish reviewed plan,
   findings and actual checks. Recheck issue openness/revision and base SHA;
   changes require revalidation instead of silent rebasing. Mark the matching
   draft PR ready, triggering the existing reviewer. Never merge or close.

Model outputs have strict schemas and cannot set GitHub paths, owners, budgets,
credentials or stage transitions. Every loop has a configured round cap; the
whole issue lifetime and month have atomic microdollar caps. The user approved
$20 per issue and $200 per month, including the one-issue live trial. Exhaustion records
REQUIRE_REVIEWER with saved progress and concrete next steps.

## Trust boundaries

Run every Codex/build/test process as an unprivileged UID with no supplementary
groups. Root-owned canonical source snapshots, issue context, policies and output schema
sit under non-writable parent directories. Agent edits use a separate writable
tree. Build/test commands run without GitHub/provider secrets or checkpoint
capabilities; fresh processes and read-only snapshots prevent one agent from
changing another's evidence. Only the trusted launcher captures stage results
and computes the artifact digest. Validate bounded regular files, reject
symlinks/path traversal and protected paths (.git, .github/workflows, automation
tools, credential/config files) before publication. Keep trusted runtime and
orchestration scripts outside any writable tree. No arbitrary outbound network.

Trusted rendering chooses CAPS statuses and author/maintainer mentions; sanitize
all model and issue prose to prevent injected mentions or markup.
GitHub publisher accepts a validated artifact and exact approved digest, never
shell commands or arbitrary API paths. Restrict branch names to the issue-bot
namespace and verify existing branch/PR provenance before updating or readying.
Use a durable outbox and deterministic markers/branch/commit identities to
recover lost acknowledgements without duplicate comments, commits or PRs.

## Validation and rollout

Independent subagent reviews this plan before implementation; a new subagent
implements the approved plan; a fresh reviewer evaluates final changes and
fixes iterate until accepted. Offline tests exercise forged webhooks, duplicate
events, concurrent FIFO admission, stage/approval/hash gates, retry exhaustion,
budget reservations, failed publication and recovery, bot-author routing,
malicious build evidence tampering and path validation. Run type generation,
TypeScript, Python lint, package tests and required repository gates.

Deploy disabled first; run isolated recovery probes, configure the separate App,
then enable new-issue intake. Use one verified jeqcho-authored existing issue
(candidate #401) for the live end-to-end trial, preserving actual failures and
costs. Finish only when the trial's issue review and fix workflow have reached a
truthful terminal state, with a ready PR if approved or a concrete hold if the
issue cannot safely qualify. Record deployment versions, exact revisions and
public links. Verify the existing PR reviewer actually receives the ready event
and routes bot-author actions to @jeqcho. Do not duplicate an existing contributor
fix just to complete a trial; a duplicate triage is a valid trial outcome, with
the fix path verified separately in offline/probe tests.
Any App/account configuration blocker must be reported explicitly.

Independent plan review: APPROVE after semantic revision, unprivileged execution,
trusted rendering and real handoff verification corrections (2026-09-20).
