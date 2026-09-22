# Independent PR reviewer

Advisory review service for `robocurve/inspect-robots`, hosted in Robocurve's
Cloudflare account. It uses `gpt-6-astra` with high reasoning and a new context
for each run. Scope, usefulness and correctness are assessed separately. Missing
prior approval does not stop technical review or automatically trigger escalation.
The deployed,
versioned [policy](src/policy.md) incorporates the maintainer's review guidance;
PR content cannot replace it.

## Decisions

| Verdict | Meaning | Next step |
| --- | --- | --- |
| APPROVE | Worthwhile, established scope, sufficient evidence, no blockers | Tag `@jeqcho` to merge after `ci-ok` is green on the reviewed head |
| REQUEST_CHANGES | Established scope with concrete implementation defects | Tag the PR author with the trigger, expected/actual behavior, impact and fix |
| ESCALATE | Scope, necessity or competing proposals need human judgment | Tag `@jeqcho` with a precise decision; closure is only a recommendation |
| REQUIRE_REVIEWER | Inspection or checks could not finish | Tag `@jeqcho` to arrange completion of the remaining checks; no product decision |

Every review starts with its uppercase status: **APPROVE**, **REQUEST_CHANGES**,
**ESCALATE** or **REQUIRE_REVIEWER**. A one- or two-sentence summary follows, describing
the change and the reason for the verdict, then the tagged next action.
Service failures use **REQUIRE_REVIEWER** too; hidden tracking metadata goes at the end. Detailed
findings, immutable head/base, contract review, tests and command records are
in an expandable section. Scope and value decisions requiring a human use
`NEED_REVIEWER`; escalation and merge requests still mention `@jeqcho`.
Every final review names an action owner. Approvals awaiting CI still tag
`@jeqcho`, with the merge request deferred until CI passes. Service failures and
budget holds tag `@jeqcho`. Author mentions come from GitHub PR metadata; missing,
deleted or bot-only author accounts fall back to `@jeqcho` to coordinate changes.
They disclose automation and record actual sandbox commands separately from CI. Contributor
intent and personal characteristics are never grounds for a finding.

An unfinished review that has confirmed defects still uses `REQUIRE_REVIEWER`,
but leads with the bugs and required fixes above the expandable details. It tags
`@jeqcho` to coordinate fixes first and finish the remaining validation afterward.
Unfinished checks appear as a visible "Remaining checks" checklist before the
expandable section. "Checks completed" inside the details lists work already done.

The bot does not merge, close, label, submit formal approving reviews, edit code,
approve Actions, or modify branch rules. Its named check is initially advisory.
Making this check required needs a separate maintainer decision.

Review instructions prioritize changed code and tests, track per-file coverage,
page large diffs, and retrieve surrounding code only when needed. Product questions
name the actual maintenance or design tradeoff. Incomplete reviews identify exact
unchecked files or behavior, the reason, and the next verification step. Review
commands and earlier bot verdicts are excluded from discussion evidence; verified
maintainer comments are evidence whose actual meaning must still be read.

## Architecture and credentials

The public Worker authenticates GitHub HMAC signatures, restricts the repository
and installation, deduplicates jobs in a SQLite Durable Object and starts a Workflow.
The review engine is Codex CLI 0.155.1, running `gpt-6-astra` with high reasoning
in a new disposable Cloudflare Sandbox. It receives the versioned natural-language
review policy, immutable head and merge-base source snapshots, and PR/issue context.
Codex uses its own multi-turn file, search and shell tools to inspect local diffs
and run focused checks. We do not implement a separate model/tool loop or send
both complete versions of every changed file to the model. Large unchanged files
therefore do not trigger the former 100 KB per-file gate.

A private publisher Worker alone holds the GitHub App key. Its service binding
exposes bounded allowlisted reads and validated comments/check runs, with no merge,
close or arbitrary write method. Native `pull_requests:write` technically permits
closing, so protect the App key and publisher deployment. `contents:read` prevents
merges. The sandbox receives neither the App key nor a GitHub installation token.

A private model gateway outside the sandbox holds the OpenAI key and enforces
spending before each inference request. Codex receives only a short-lived capability
for its own review budget, supplied through a client-only environment-backed HTTP
header. Capabilities are never placed in process arguments. Outbound networking is denied except for the internal
Responses proxy; arbitrary hosts, provider endpoints and paid provider tools are
not allowed. Exact duplicate submissions cannot double-charge an ambiguous request.
Streamed usage settles reservations, while missing usage retains them. The CLI's
own automatic request/stream retries are disabled. Completed output is separately
checkpointed in the ledger before the launcher exits; delivery retries do not
repeat inference. The result-delivery capability is separate from the model
capability and stored in a root-only request file; unprivileged review commands
cannot use their model token to submit a terminal result.

Canonical head/base snapshots and context are root-owned and read-only, beneath
root-owned, non-writable parent directories. Archive permission bits cannot make
them writable. Build backends run as a separate unprivileged user in a disposable
copy; they cannot alter canonical evidence, the reviewer's home or its result
directory. Setup records are written by the trusted launcher. Codex receives a
writable scratch directory for reproductions and modified test copies.

The Cloudflare container's management API is **not** a Unix-user isolation
boundary: an ordinary process in its network namespace can reach the root API
on localhost. `enableInternet=false` does not prevent this. Repository commands
therefore run through Codex's pinned native Linux sandbox with a separate PID
namespace, a read-only root, explicit denial of the client's home/output and
request/checkpoint files, and network syscalls restricted by the native sandbox.
Only scratch is writable. Although the client and its sandboxed shell have UID
65534, the shell cannot access the client's files, environment, processes or
network endpoint. Shell environments are explicitly allowlisted without the model
capability, and requests to run outside the sandbox are rejected (`never`).

Package builds use UID 65533 with separate network, PID, mount, IPC and UTS
namespaces, no supplementary groups/capabilities, and `no_new_privs`. Every run
checks both boundaries before executing builds or making paid model calls; an
unsupported control stops the run, with no unsandboxed fallback. The client itself
retains access to the private model gateway. Scratch files
persist within that fresh session and are destroyed afterward. Python 3.11,
NumPy, pytest, pytest-cov, hypothesis, pip, Hatch, httpx, websockets, mypy, Ruff
and rg are preinstalled. These include the agent plugin’s declared runtime and
Python test dependencies; package versions are pinned in the Dockerfile. Offline
local package builds run automatically for core and changed Python packages, using a disposable
copy of the source and synthetic version 0.0.0. Installation is offline, without
dependencies or build isolation, as the build user; each package has
a 45-second limit and setup has a 2-minute total limit. Codex receives setup.json
and can repair routine setup failures. Network installs and hardware checks remain
unavailable. Setup and Codex together have a 20-minute deadline, with an independent container shutdown at 22 minutes.
Only one basic container can run at once. Provider credentials and GitHub publishing
remain outside this environment even though Codex can execute arbitrary review code.

Execution preparation atomically records the sandbox identity, scoped capability,
merge base and $0.10 allowance. The runner starts one named background process;
a durable launch claim prevents ambiguous acknowledgements from starting another.
Workflows poll in short retryable steps rather than holding one RPC open for the
whole session. The launcher checkpoints its bounded JSON output independently;
polling can deliver the same output if that callback fails. Conflicting terminal
outputs are rejected, and a saved output blocks further model calls.

Cleanup and publication happen after checkpointing. Cleanup errors cannot replace
a verdict, and a GitHub outage leaves the validated result pending for the
reconciler. Interrupted workflows resume the same execution and saved output,
including when remaining budget is below the admission floor. They never silently
start a new paid session. An expired execution with no output remains incomplete.
The old PR456 failure predates checkpointing and cannot be recovered retroactively.
Failure notices use a separate pending-publication state too. Both workflow and
scheduled-recovery errors remain retryable until GitHub acknowledges the notice;
retrying delivery does not run another model review.

Each sandbox session reserves $0.10 conservatively against the same spending caps.
That is a budget allowance, not a claim that Cloudflare charges ten cents. Its model
calls consume the remaining allowance. A partial/failed CLI result cannot approve;
complete output is validated against the review schema and cited file locations.
Comments include actual CLI command records separately from CI results. Model
judgments can still be wrong; Jay makes final decisions.

The job key includes both head and base. Every publication rechecks the live PR.
GitHub has no atomic compare-and-comment API: a push can race the final request,
so every comment names its exact revision and checks attach to that head only.

## Budget

| Limit | Amount |
| --- | --- |
| Model calls and sandbox allowances for one PR head, including reruns | $5 |
| All revisions/reruns of a PR, lifetime | $15 |
| All reviews per UTC calendar month | $200 |
| Monthly warning to Jay | $160 |

The maintainer authorized a $25 cap for trial PR #456 at head
`696fbaa9a00d7c345a81dd179fa10934f51ade89`. The deployment-only
`REVIEW_HEAD_LIMITS_JSON` and `REVIEW_PR_LIMITS_JSON` settings record the current
head and PR lifetime exceptions in microdollars. Existing charges are preserved;
both still share the $200 monthly ceiling. Other heads retain the $5 default,
and other PRs retain the $15 lifetime cap.

Reservations are atomic across the deployment and recorded before submission.
Input is counted using OpenAI's token-count endpoint, limited to 200,000 tokens,
and reserved at $13/M plus a small token margin. This conservatively covers the
published $10/M ordinary input and $12.50/M cache-write pricing. Output, including
reasoning, is reserved at $50/M. Each call allows at most 16,000 output tokens;
Codex continues across turns within the total budget and session deadline.
Confirmed cache reads settle at the published $1/M rate; other input settles at
the conservative $13/M ceiling. Reservations never assume a future cache hit.
The gateway supplies the remaining budget each turn and requests a final answer
when less than $0.30 remains after reserving the current uncached input. It no
longer requires space for two full uncached histories before allowing investigation.
If material evidence is missing, the result is REQUIRE_REVIEWER, not a product escalation. The ledger can still reach its limit before
the OpenAI bill does. See [Astra pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

Fresh runs require at least $2 remaining before any sandbox/model spending. This
is an admission floor, not a guarantee that a large PR will finish for $2. Reruns
share the original caps; failed runs do not reset the ledger. Published results
and holds include separate per-run settled model charges, unresolved reservations,
sandbox allowance, cumulative revision spending and remaining allowance.

No inference submission retries automatically. Ambiguous failures retain the
full reservation, including across a process restart. Only retrieval/publication
retries. Unexpected usage exceeding a reservation freezes further inference.
Responses use the standard service tier; no paid provider tools are enabled.
Monthly assignment is the UTC month in which a call is reserved. Set a separate
$200 hard project limit in OpenAI as a second boundary, especially across month
boundaries or if another service uses that project. Pricing is pinned in code
and must be rechecked before changing the model. Cloudflare hosting is separate.

## Operations

New non-draft PRs and new revisions trigger review. Drafts and closed PRs are
ignored. There is no initial backlog scan. A ten-minute reconciliation schedule
recovers queued jobs and updates approved comments when CI turns green.

Every trigger shares a durable FIFO queue in the SQLite ledger. Workflows display
a queued GitHub check and sleep on Cloudflare until they atomically acquire the
single review slot. Waiting does not reserve sandbox or model budget. Admission
rechecks whether the PR is still open, non-draft and on the same revision. The
slot is released only after acknowledged sandbox cleanup, or if no sandbox was
prepared. Uncertain cleanup retains ownership and is retried by reconciliation.
Restarts preserve order and ownership; interrupted waiting workflows resume the
same job. Long waits are continued in a new Workflow instance before step history
can grow indefinitely. Backlog commands use this same queue and need no local
dispatcher or computer left online. Budgets and one-container concurrency remain
unchanged.

Only GitHub user ID `42904912` (`jeqcho`) can request these commands:

```text
/review
/review scope <full-current-head-sha> <scope decision and rationale>
```

A command creates a fresh run but shares the same head/PR/month budgets. A scope
decision is evidence of authorization, never proof of correctness. To increase
limits, release an uncertain reservation, or clear a billing hold, inspect usage
and logs first, then make an explicit operator change; the public endpoint cannot
change budgets. Repeated failed webhooks can be redelivered from GitHub App
settings. No model API key is needed in GitHub Actions.

```sh
npm ci
npm run types
npm run check
npm run deploy:runner
npm run deploy:publisher
npm run deploy:reviewer
node scripts/setup.mjs secrets
node scripts/setup.mjs webhook https://<reviewer>.workers.dev/webhook
```

The setup script reads the key file in `~/.config/robocurve-pr-reviewer/` and the
downloaded App key (override its path using `REVIEWER_PRIVATE_KEY_PATH`). It
generates a private webhook secret there and uploads credentials through Wrangler
stdin. It never prints values. Deploy initially with `ENABLED=false`, upload
secrets, enable App webhooks and subscribe to **Pull request** and **Issue comment**
in GitHub App settings, configure the URL/secret, then set `ENABLED=true` and
redeploy. Cloudflare account ID is explicit in all three configs.

The configured CPU allowances require Workers Paid (currently a $5/month base
subscription); Free's 10 ms invocation limit is unsuitable for reliably parsing
large review contexts. See [Cloudflare pricing](https://developers.cloudflare.com/workers/platform/pricing/).

To pause new reviews, set `ENABLED=false` and redeploy. Terminate in-flight
Workflows too if immediate cessation is required. `/health` reports enabled state
and policy version without secrets. Logs contain opaque job IDs and safe error
categories; do not enable SDK request debugging. Cloudflare Workflow state retains review context; OpenAI requests use `store=false`
and `background=false`. No cross-PR conversation is used.

Tests use the real local Workers/SQLite runtime with all network calls mocked.
They cover concurrent spending, replay, head changes, untrusted inputs, decision
consistency, publication boundaries and ambiguous model failures. CI runs them
without production credentials.
They also cover concurrent ready events, durable FIFO ownership, zero spending
while waiting, cleanup failures and both failure-notice publication paths.
An unstarted job abandoned when a PR becomes draft or closes is replaced on a
ready/reopened event, even at the same revision. Repeated events share that new
job; its predecessor cannot start or spend. Historical charges still count
against the same head, PR and monthly limits. Started or already reviewed jobs
require an explicit `/review` to request another run.
`sandbox/test_evidence.py` runs offline as root inside the review image. Its real
Python build backend attempts source/context replacement, parent-directory
renames, permission changes and result forgery; all must fail while package
installation and writable scratch reproductions still work.
It also starts the real Codex CLI against a local synthetic Responses server,
exercises real tool execution, attempts sandbox escalation, and verifies the
client-only header, filtered tool environment and final result. Both build and
tool paths attempt the localhost management exploit over IPv4/IPv6, client file
tampering and process/environment access. Positive checks require package
installation and scratch writes to succeed. Native Linux is required; AMD64
emulation on an ARM Mac cannot validate seccomp. `isolation-probe.wrangler.jsonc`
runs these exact sources in a credential-free Cloudflare container; it has no
OpenAI key, GitHub publisher or production ledger and never posts a PR review.

A management-only workflow payload `{"id":"safety-pause","pauseReviews":true}`
persistently pauses queue admission and model reservations, revokes the active
review session and destroys its sandbox. Waiting jobs are retained. Resume only
after deployed boundary verification with `pauseReviews:false`; do not reuse
an execution terminated by the safety pause.

Hosting estimate (before adding sandbox execution): a few hundred reviews per month should fit the included
Workers, Workflows and SQLite allowances, so the expected incremental hosting
charge is $0 beyond the $5 base subscription. Allow $1-$2 headroom pending real
usage; this is an estimate, not a hard hosting cap. Quotas are shared across the
account. Workflows include 500,000 steps and 1 GB-month of state; API waiting and
step sleeps do not incur Workflow CPU time. See [Workflow pricing](https://developers.cloudflare.com/workflows/reference/pricing/)
and [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/).

Sandbox execution additionally uses Cloudflare Containers CPU, memory and disk. The runner scales to zero and permits one basic instance. Budget allowances bound requested executions conservatively, but the Cloudflare invoice is separate from OpenAI and its $5 base subscription. See [Containers pricing](https://developers.cloudflare.com/containers/platform/pricing/). Docker is needed to build/deploy the runner image.

### Read-only run accounting

Cloudflare operators can retrieve ledger totals for an existing run without
starting Codex or publishing a comment:

```sh
npx wrangler workflows trigger inspect-robots-review '{"id":"EXISTING_RUN_ID","inspectOnly":true}' --json
npx wrangler workflows instances describe inspect-robots-review RETURNED_INSTANCE_ID --json
```

Use `{"id":"queue-status","inspectQueue":true}` with the same trigger command
to inspect the active slot owner and FIFO backlog without inference or publishing.

Operators can also use `"inspectOutput":true` instead of `"inspectOnly":true` to
read a saved terminal artifact and its costs when diagnosing validation failures.
This does not launch a process or publish a comment. After a validator fix, an
operator can revalidate a held job's saved output using
`{"id":"ORIGINAL_JOB_ID","recoverSaved":true}`. This explicitly returns only a
held job with saved output to the queue, preserving its execution and charges.
Stale revisions, security-stopped jobs, a paused queue and jobs without output
cannot use this path. The full workflow revalidates and publishes the saved
result without inference. GitHub `/review` still requests a genuine fresh run.

This management-only diagnostic also works for historical run charge IDs. It
does not reset charges, change limits, or accept parameters from PR text.

### Isolated recovery integration test

`recovery-probe.wrangler.jsonc` deploys a separate temporary Worker, Workflow,
ledger and container. `test/recovery-probe.ts` serves synthetic Responses events
(no OpenAI key), runs the actual Codex CLI and one shell tool, and injects lost
completion/cleanup acknowledgements. It has no GitHub publisher and does not
review or comment on a real PR. Replay must preserve its terminal output with
one launch and two synthetic model turns total. Delete the probe resources after
verification; they are not part of the production service.

The credential-free isolation probe also checks dependency availability and
runs agent tests under the real native Codex tool sandbox. Trigger it
with `{"head":"HEAD_SHA","base":"MERGE_BASE_SHA"}` to test those immutable
snapshots. It checks installed versions against the plugin’s declarations,
builds the local packages offline, then records pytest collection, the affected
policy suite and a full-suite attempt. The full suite includes localhost-server
tests that the production network boundary denies; their failures are retained,
and a completed probe does not claim that every test passed. Omit the parameters
to run the malicious-build and client-isolation regressions. Neither mode calls a model or publishes a PR review.
