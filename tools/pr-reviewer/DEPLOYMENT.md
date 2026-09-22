# Deployment record

Date: 2026-09-20. Repository: `robocurve/inspect-robots`.
Cloudflare account: `7f405baff0972dc740a02ee0f700d2c1` (Robocurve).
GitHub App: `robocurve-pr-reviewer`, ID `5012304`, installation `163290338`.

Current status: live in advisory mode for new non-draft PRs and revisions.

- OpenAI key verified for `gpt-6-astra`.
- Live standard-tier background structured-output request with high reasoning
  completed successfully. Input count matched actual usage: 37 input tokens,
  13 output tokens, approximately $0.00102 at standard rates.
- Worker bundles passed Wrangler dry-run compilation.
- TypeScript and 64 offline policy, ledger, gateway, queue and orchestration tests passed. Obsolete custom-loop tests were replaced by Codex gateway/lifecycle tests.
- Core checks passed: Ruff, formatting, mypy, 1,720 pytest tests with 100% core
  coverage. Six optional rerun-sdk tests skipped because that extra is absent.
- Clean npm install succeeded; dependency audit reported zero vulnerabilities.
- Workers Paid enabled by the maintainer. The reviewer and publisher deployed successfully.
- Publisher version: `f8911de7-7b2b-4999-b999-ceb2e6c38fc1`.
- Reviewer version: `b8b2f1c6-dcbc-458c-987d-4d3e42a27472`.
- Receiver: https://inspect-robots-reviewer.jay-7f4.workers.dev/webhook
- Health endpoint reports advisory mode and `enabled: true`.
- GitHub private key, OpenAI key and generated HMAC secret uploaded securely.
- Production signed ping accepted (200); invalid signature rejected (401).
- CodeQL sanitizer alerts addressed by escaping each HTML delimiter; regression
  tests pass and the updated CodeQL check no longer reports a failure.
- GitHub App subscribed to `pull_request` and `issue_comment`; URL and secret
  configured. GitHub-generated ping redelivery accepted (200).
- Live draft-PR probe accepted (202) through the private publisher and GitHub API;
  no Workflow created, confirming draft deferral without model charges.
- Fixed an outbound request compatibility failure found by the live probe.
  Explicit Worker Requests use manual redirect handling and reject all 3xx
  responses, preventing credential forwarding. Regression test included.
- No merge rules were changed. No existing PR backlog was imported.
- Restored Sravanthi's merged-and-reverted PR #453 as trial PR #456 by
  reverting #454. Its tree matches the original merged implementation exactly.
- The first trial stopped before inference because GitHub omits patches for
  empty files. Fixed by verifying complete immutable file contents are empty;
  zero diff counts alone cannot bypass the missing-patch guard. Regression
  tests cover additions, removals, binary data and BOM-only files.
- A `/review` rerun reached the existing 100,000-byte per-file cap: the
  restored `uv.lock` is 524,568 bytes. Workflow
  `679ba5b31f920ab2086ff632283630c0520279906ccaa88b` published a held result
  and tagged Jay. Neither trial reached model submission or budget reservation;
  no model-review verdict was produced. PR #456 remains open with green CI.
- Hold notices now include an allowlisted reason, safe file-size details when
  available, a next step and a run reference. Budget, context-size and model
  failures have distinct explanations. Raw exceptions and provider responses
  never become public comment text.

No further setup is required for advisory processing.
Turning the independent review check into a merge requirement is a separate
rollout decision. Implementation is tracked in PR #455; deployment is already live.

## Codex engine rollout

- Replaced the hand-built model/tool loop with Codex CLI 0.155.1 and a natural-language review policy, using Astra/high.
- Public source snapshots at the head and merge base are staged in an isolated Cloudflare Sandbox. Codex uses its own diff/search/file/shell tools and fresh session.
- Added a private model gateway: session-scoped expiring access, pinned model/reasoning, atomic per-request reservations, duplicate-submission prevention, streaming usage settlement. Provider keys remain outside the sandbox.
- Sandbox has no GitHub credentials and no outbound access except its budgeted Responses proxy. One basic instance; 20-minute CLI limit and independent 22-minute shutdown.
- Native CLI mock integration passed: initial request, shell execution, continued request containing its tool output, and structured final JSON. No provider credentials or live inference used for that test.
- Offline Docker test on the trial PR passed 14 focused plugin tests with local package setup and networking disabled.
- First production CLI attempt stopped at the launcher before model review. Corrected the executable path for Cloudflare's shell.
- Live native CLI sessions performed multiple model turns, diff inspection and source/context reads. The shared per-head budget then prevented further inference; no completed review verdict was issued.
- Diagnostic run `230721d16d351acd424c9633cb672b40619b430a06f4d553` confirmed $0.244662 remaining in the conservative head ledger. This is not an invoice total: earlier settlement charged every input token at the non-cached ceiling.
- Corrected future settlement to credit confirmed cache reads at $1/M, added final-turn budget steering, and preserved budget-stop reasons independently of CLI stderr. Existing charges remain unchanged because historical cache usage was not retained.
- No further paid trial was started after these fixes. A complete end-to-end verdict on PR #456 remains pending additional authorized trial allowance. The $5/head, $15/PR and $200/month limits remain unchanged.
- Runner version: `2b88873c-0106-44ce-8886-a8515cacb2f0`.

## Authorized trial budget exception

- The maintainer authorized raising PR #456 head `696fbaa9a00d7c345a81dd179fa10934f51ade89` from $5 to $10 after earlier attempts consumed its shared allowance.
- The deployment-only exception preserves charges and leaves all other heads at $5, with the $15 PR and $200 monthly ceilings unchanged.
- All 34 offline reviewer tests passed, including concurrent reservations against the exact-head exception and the unchanged PR ceiling.
- Live workflow `90dfed5db293fd070613dafecef32a618ee95003f12e97a9` completed and published ESCALATE: scope needs Jay's decision and technical inspection remained incomplete. Codex reported 131 plugin tests passing.
- The session completed 12 model calls, booked $2.132670 in model usage plus the $0.10 sandbox allowance, and issued no approval. These ledger amounts are conservative allowances, not invoice totals.
- Published review: https://github.com/robocurve/inspect-robots/pull/456#issuecomment-5752014167

## Review presentation

- Future scope/value decisions use `NEED_REVIEWER`; escalation and merge notifications still mention `@jeqcho`.
- Verdict comments start with a bounded, one- or two-sentence TL;DR and the requested action. Supporting findings, test results and command records are in an expandable section. Incomplete-run notices also start with a TL;DR.
- The prompt prohibits repeating the summary, scope decision and test results across paragraphs. Existing published reviews were not rewritten and no paid review was triggered for this presentation change.
- TypeScript and all 34 reviewer tests passed.

## Review policy version 2

- Missing prior approval is not an automatic scope blocker. Technical review continues while a concrete product decision is pending; routine work can fit established scope without a separate approval comment.
- Product escalations must state the actual maintenance/design choice. Incomplete reviews must identify unchecked files or behavior, the limiting reason and the next check. Internal authority terminology is excluded from public prose.
- The CLI now starts with a changed-file inventory and bounded per-file diffs, prioritizes changed code/tests, tracks coverage, and avoids repeated full-file/context dumps.
- Verified maintainer comments are separated from other discussion, without calling every comment a decision. Bare review commands and prior bot verdicts are excluded, including linked discussions.
- TypeScript, 35 offline reviewer tests, and Python launcher lint/format checks passed. No paid rerun was requested for this policy update; future behavior has not yet been evaluated in a live review.

## Review policy version 3: setup, cutoff and accounting

- Fixed the premature final-turn guard: investigation no longer requires room for two full uncached request histories. Atomic per-call reservations and all existing spending caps remain enforced.
- Fresh reviews require at least $2 remaining before launching a sandbox or submitting inference. This prevents another fresh run from consuming a nearly exhausted revision allowance; it does not guarantee completion for $2.
- The launcher installs core and affected Python packages offline, without dependencies, in a disposable source copy as UID 65534. Setup is bounded to two minutes inside the existing 20-minute session deadline. Failures and command records are available to Codex.
- INCOMPLETE/COMPLETE_REVIEW now represents unfinished inspection and setup/resource gaps, with empty decision_needed and specific remaining checks. Actual product decisions still use ESCALATE and mention Jay.
- Published results and failure notices include per-run settled model costs, unresolved reservations, sandbox allowance and cumulative revision spending. A management-only inspectOnly workflow retrieves historical accounting without inference or publication.
- Verified: TypeScript and 39 reviewer tests; native Codex two-turn shell/structured-output smoke test; automatic package installation and all 131 Jev plugin tests in an offline AMD64 container, with one existing degenerate-calibration warning; Ruff, formatting, mypy and 1,720 core tests at 100% coverage (six optional rerun-sdk skips).
- Live health confirms policy 3. Read-only diagnostic budget-audit-policy3 confirmed latest run 241f23cdf3337ce7a29fa055107c00cb83c95e4d046ba9d3 booked $0.568857 for four model calls plus $0.10 sandbox allowance, with no unresolved model reservations.
- The current PR456 head has used $8.974748 of its authorized $10 cumulative cap; $1.025252 remains. No new paid review was launched because that is below the admission floor. No limits or historical charges were reset. Policy 3 has not yet produced a new live model review.

## Authorized policy 3 live verification

- The maintainer authorized increasing only PR456 head 696fbaa9a00d7c345a81dd179fa10934f51ade89 from $10 to $15 cumulative for a fresh verification run. The $15 PR lifetime and $200 monthly caps, and other heads' $5 default, remain unchanged. Prior charges remain intact; the run starts with $6.025252 available.
- TypeScript and all 39 reviewer tests passed, including concurrent reservations against the updated exception and refusal to exceed the PR cap through another revision.
- Reviewer deployment: 5df86445-18bc-490f-a6c1-e21902a51080. A single /review command on PR456 started live workflow 256f8224ace3e0937f6e445d5032d0e71995febc701a8e06.

- The verification run made 21 model calls and settled $3.982732 in model usage plus the $0.10 sandbox allowance. Revision total is $13.057480 of $15; $1.942520 remains. No model_budget stop was reported.
- Cloudflare failed the execution step after roughly six minutes with WorkflowInternalError: "Attempt failed due to internal workflows error". Both instance diagnostics and the full-step recovery API report an errored step with no output. The service published a held notice with costs; no review verdict or test execution record from this run was recovered. This is an unresolved execution/durability failure, not a verified successful policy 3 review.
- Public result: https://github.com/robocurve/inspect-robots/pull/456#issuecomment-5755747746
- No second paid run was launched. A result-persistence/recovery improvement is needed before further paid verification; the exact underlying Cloudflare internal fault is not exposed by the available diagnostics.


## Durable execution and result recovery

- Replaced the single long-running Workflow RPC with a named background Codex process and short, retryable polling steps. SQLite atomically records the execution identity, scoped capabilities and one sandbox allowance before launch.
- The root launcher saves its bounded terminal output directly to the ledger before exit. The trusted process poller can deliver the same artifact if the callback fails. Terminal output is immutable and blocks further model calls; ambiguous launch acknowledgements cannot launch a second process.
- Result delivery uses a separate root-only capability. Codex and untrusted package builds cannot read its request file, and the model capability cannot authorize a checkpoint.
- Workflow interruptions resume the same execution, including below the fresh-run admission floor. Cleanup errors cannot discard a saved review. Publication failures retain the validated verdict for the ten-minute reconciler.
- TypeScript and 51 real Workers/SQLite tests passed, including concurrent preparation, eviction, lost acknowledgements, exhausted-budget recovery, capability separation, automatic resumption and delayed publication.
- The original PR456 result predates this persistence path and remains unrecoverable. No new paid PR review was started, charges were not reset, and all spending limits remain unchanged.
- Live isolated Cloudflare workflows `recovery-final-first` and `recovery-final-replay` completed using the real Codex CLI and synthetic model responses. Both reported exactly one launch, two model turns and one injected completion-acknowledgement failure. Repeated injected cleanup failures did not discard the result. Replay skipped launch/poll entirely. Both offline package installations and the unprivileged credential-isolation command exited successfully.
- Earlier isolated probe `recovery-resume` recovered a separately saved result after its original workflow failed. A later probe encountered mixed deployment versions; it was discarded, and the final probe above used a fresh record after rollout settled. Production intake was paused during the two-service update.
- Read-only production diagnostic `durability-budget-audit` completed after the ledger upgrade: $13.057480 cumulative PR456-head usage, $1.942520 remaining, zero unresolved reservations. This test submitted no model request and published no GitHub comment.
- Deployed reviewer `41f24283-c5ad-4125-b011-39fa9ca42c44` and runner `d9eae883-166d-48d6-98a5-68ceb1f78d01`; publisher unchanged. Health confirms advisory mode, policy 3 and enabled intake. The temporary probe Workflow, container application and Worker were deleted after verification. No new OpenAI charges were incurred by recovery testing.


## Authorized $25 trial allowance

- The maintainer requested another budget increase and one fresh review. Raised only PR456's lifetime cap and head `696fbaa9a00d7c345a81dd179fa10934f51ade89` to $25 cumulative, leaving $11.942520 before launch. Existing charges remain intact. Other heads retain $5, other PRs retain $15 lifetime, and the shared monthly cap remains $200.
- Added deployment-only PR lifetime exceptions, validated against the monthly cap. TypeScript and all 52 Workers/SQLite tests passed, including concurrent reservations, the exact-head restriction, unchanged defaults for other PRs, and the shared monthly ceiling.
- Deployed reviewer `5654f1a3-9285-49cd-94f4-ed4a2a0facca`. Read-only `cap25-budget-audit` verified the preserved $13.057480 usage and $11.942520 remaining. A single `/review` command started workflow `82b6d76a1771181c06ccb05cf0eda0a4681a6b4b12be06f1` on the unchanged PR456 head.
- The fresh model run completed successfully and persisted REQUEST_CHANGES with two reproduced blockers, 32 command records and reported inspection of all 39 files. The 131-test plugin suite passed. Publication initially stopped because the validator required nonempty supplemental prose even though the review policy asks the agent not to repeat evidence already present in other fields.
- Corrected this presentation mismatch: supplemental body text may be empty, while a nonempty summary and all substantive verdict/approval checks remain required. Added a management-only `inspectOutput` diagnostic to read saved output without inference or publication. The recovered live artifact passes the corrected validator; all 54 Workers/SQLite tests pass.
- Deployed reviewer `55ff476f-6cdb-4305-8bb7-9ee0b419efe6` and publisher `33c91289-7c0e-42ee-8aa9-3319b40d0fe5`. Recovery workflow `cap25-recover-verdict` reused the original job and saved output, skipped process launch and polling, and successfully published REQUEST_CHANGES. No additional model call occurred.
- Final published review: https://github.com/robocurve/inspect-robots/pull/456#issuecomment-5756059575 . Two reproduced blockers concern scoring an unreleased cube and applying intersected arm bounds to an individual arm. The service issued no approval or merge/closure recommendation.
- Final accounting: 15 model calls, $3.062492 settled model usage, $0.100000 sandbox allowance, no unresolved reservations. This run booked $3.162492; cumulative revision spending is $16.219972 of $25, leaving $8.780028. Saved-result recovery did not add charges. These are conservative ledger totals, not invoice amounts.


## Review statuses and action owners

- Every final review begins with APPROVE, REQUEST_CHANGES, ESCALATE or REQUIRE_REVIEWER, followed by a short summary and tagged next action. Service failures use REQUIRE_REVIEWER; saved legacy INCOMPLETE results remain readable and render with the new name. Hidden tracking metadata moves to the end of comments.
- REQUEST_CHANGES tags the actual PR opener from GitHub metadata. All other verdicts tag @jeqcho. Approvals awaiting CI mention Jay without requesting a merge until CI passes. Unmentionable, missing or deleted author accounts fall back to Jay to coordinate fixes.
- PR456 was opened by jeqcho, so its edit requests consistently tag @jeqcho. No original-contributor exception is configured.
- TypeScript and all 56 Workers/SQLite tests pass, including status prefixes, trusted author selection, mention-injection rejection, CI gating and legacy saved-result compatibility.
- Deployed publisher `0bab7490-98b0-4d3b-b54c-58c7547f8dc9` and reviewer `e0b9473d-325a-4ee6-9690-73f331a53e3f`. The maintainer explicitly requested a fresh end-to-end review rather than a saved-result refresh; one new `/review` command was submitted for PR456 with $8.780028 available. No spending limits were changed.
- Fresh workflow `d9a2115383a52d40b7ea268bbc84d1ea0f2de0d48fcebdda` completed through launch, saved verdict, cleanup, publication and delivery without recovery. Published REQUEST_CHANGES with three reproduced blockers and 131 passing plugin tests at 100% reported statement/branch coverage.
- Verified the new comment starts with `**REQUEST_CHANGES**.` and immediately tags the actual opener with `@jeqcho, please address the 3 blocking findings`. Review: https://github.com/robocurve/inspect-robots/pull/456#issuecomment-5756257176 . No original-contributor override was used.
- The fresh test made 16 model calls, settling $3.168671 plus the $0.100000 sandbox allowance. This run booked $3.268671; cumulative revision usage is $19.488643 of $25, leaving $5.511357. The temporary accounting monitor was stopped after verification.


## Durable queue and review integrity: 2026-09-21

- All webhook and manual triggers now share one atomic FIFO admission gate in the SQLite ledger. Workflows wait on Cloudflare and display queued checks without starting containers or reserving model budget. Cleanup acknowledgements fence slot release; recovery retains the same job and execution. No Mac dispatcher is required.
- Canonical snapshots, context and their parent directories are root-owned and immutable to build and review users. Builds use UID 65533 in a disposable copy; Codex uses UID 65534 with its own home, pre-created CODEX_HOME, scratch and output directory. Setup records remain launcher-owned.
- Failed workflow and scheduled-recovery hold notices remain in a pending-publication state until GitHub acknowledges delivery. Reconciliation retries delivery without new inference.
- Incomplete reviews lead with confirmed bugs and fixes when present. The visible Remaining checks checklist names unfinished validation; Checks completed is separate inside the details. REQUIRE_REVIEWER continues to tag jeqcho.
- TypeScript and 64 Workers/SQLite tests pass. Two offline Linux regressions pass: a real build backend fails all 11 evidence/output tampering attempts while installation and scratch work succeed; the real Codex CLI starts against a local synthetic Responses server and writes its result without any provider key.
- Initial live queue verification exposed a missing CODEX_HOME directory before model execution. PR404 and PR376 each booked a $0.10 sandbox allowance and zero model calls. Waiting workflows were paused, the directory creation was restored under the review UID, and a real-CLI startup regression was added. No charges or limits were reset.
- Production publisher: f8911de7-7b2b-4999-b999-ceb2e6c38fc1. Reviewer: b8b2f1c6-dcbc-458c-987d-4d3e42a27472. Runner: 2b88873c-0106-44ce-8886-a8515cacb2f0.
- The maintainer requires genuine new bot runs for review requests. A temporary saved-comment formatting refresh was removed; requested reruns use fresh review commands and the existing cumulative caps.

## Management API and client isolation: 2026-09-21

- Paused admission and inference persistently, revoked the active session and destroyed its sandbox before investigating the reported localhost API bypass. Unix ownership and `enableInternet=false` alone were insufficient; prior permission-only verification did not establish this boundary.
- Codex now uses its native Linux sandbox with a separate PID namespace, read-only filesystem, explicit denial of client home/output and checkpoint files, network restrictions, and approval policy `never`. Commands retain UID 65534 but cannot reach the client through its filesystem, processes or environment. Package builds use UID 65533 inside new network/PID/mount/IPC/UTS namespaces with no capabilities and `no_new_privs`.
- The model capability travels in a client-only environment-backed HTTP header, never process arguments. Tool environments are allowlisted. Every production run checks both execution boundaries before untrusted builds or paid inference and stops if enforcement is unavailable.
- TypeScript and 67 Workers/SQLite regressions pass, including persistent pause across eviction and atomic refusal of new reservations/executions. Ruff and formatting checks pass.
- Exact deployed-image Cloudflare verification `cf_de353ac41694575219850b2ea62ecca5c6722f6fd7f422d4e3d61f519668b5c9` passed all three Linux regressions. Tests cover the live localhost management control, malicious package building, canonical/client/checkpoint file protection, PID and environment separation, IPv4/IPv6 HTTP and direct sockets, real Codex tool execution, denied explicit sandbox escalation, private-header delivery and valid client result output. Script contents were compared against the image before testing.
- Credential-free full-run transport verification `cf_29b39bf5d1e334c4fb4bb3998971a4531d42094deee133f4cd9ef4646f0192e4` completed with one launch and two synthetic model turns. Both offline package installs and the protected-checkpoint command succeeded. Injected lost completion/cleanup acknowledgements did not discard its result. These synthetic fixtures were never published as PR reviews and made no OpenAI calls.
- Production runner: `faf4ba9d-f649-42c4-8eea-c3c209c1a4dd`; image digest `sha256:35794a21c569c4889a36cdde424baecc1941819e8cd3e39202af1ddb672dc61f`. Reviewer: `375434cd-6c5d-4971-9ea1-194dd386b9a0`. Publisher unchanged.
- Existing charges and caps remain intact. PR373's earlier incomplete run used $2.474582 including unresolved reservations; $2.525418 remained. PR349 was stopped for the security fix after $0.601795; $4.398205 remained. Neither produced an approval. Replacements must be genuine fresh runs, not cosmetic edits or resumed compromised execution state.
