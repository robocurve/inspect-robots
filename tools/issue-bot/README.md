# Robocurve issue bot

New issues in `robocurve/inspect-robots` receive a fresh GPT-6 Astra/high
assessment through the native Codex CLI. A confirmed serious defect can proceed
through a reviewed plan, an implementation, and independent code review before
the bot opens a draft PR. Successful CI and approval of the exact code artifact
allow the bot to mark that PR ready, triggering the existing PR reviewer.

Neither bot merges. The issue publisher exposes no merge or issue-closure
operation. GitHub also enforces the repository's active
[Only mergers can merge](https://github.com/robocurve/inspect-robots/rules/23732858)
ruleset on the default branch: only the designated human team can bypass its
update restriction, through PRs. Neither App is a bypass actor. The issue App
has Contents write for fix branches, so preserve this ruleset as the independent
barrier against bot merges into main.

## Public statuses

| Status | Meaning and owner |
| --- | --- |
| FIXING | A confirmed serious defect entered the plan, implementation and independent review workflow; no action needed yet. Triage and plan-review limitations are listed and passed on but do not block the fix; code review still requires none. |
| CONFIRMED | Concrete evidence establishes the bug, but it was not judged serious, so no automatic fix was started; Jay owns judgment. |
| NEEDS_INFO | Ask the actual issue author for specific missing evidence. Deleted/bot authors fall back to Jay. |
| NOT_REPRODUCED | Explain the actual checks and limits; tag Jay. |
| DUPLICATE | Another issue reports the same bug; cite that issue and tag Jay. |
| FIX_PROPOSED | An existing open PR proposes a fix; tag Jay to review it without implying it is verified or merged. |
| REQUIRE_REVIEWER | A technical, execution, budget, scope or validation blocker needs Jay. Include the latest findings and remaining checks. |
| PR_READY | The reviewed fix passed CI and the PR was marked ready; tag Jay. |

Trusted rendering chooses the status and mention. Model/issue prose cannot add
mentions or HTML. Detailed evidence is expandable and bounded to GitHub's limit.
Comments lead with `STATUS:`, `Issue summary:`, the owner mention on its own
line, and `Your action:`. Missing-information requests mention the issue author;
review and judgment requests mention `@jeqcho`.

## Workflow

Each stage starts a separate ephemeral Codex session. An independent reviewer
must accept the exact plan before implementation. The implementation gets a
new reviewer; requested changes return to a fresh implementation session and
another fresh review. The bot permits three review rounds per loop and stops
with an explanation if approval or validation remains missing.

The default spending limits are $20 per issue across all stages, revisions and
manual retries, and $200 per UTC month for this bot. Each stage records a $0.10
container allowance. Atomic reservations precede provider requests. Confirmed
cache reads use the lower price; uncertain failures retain their reservation.
These are conservative accounting bounds, not Cloudflare or OpenAI invoice
amounts. The existing PR review bot has its own limits.

Every entry point joins one durable FIFO before container execution. A waiting
issue incurs no inference. Stage inputs, launch claims, structured results and
full changed-file artifacts persist outside the disposable container. Recovery
polls the existing stage or replays publication; it does not repeat a paid stage
whose launch/result is uncertain. Queue ownership is released only after
container destruction succeeds. Comment delivery, including stopped-work
notices, uses a durable outbox.

Input is pinned to a main commit when the issue reaches the front of the queue,
and to semantic issue title/body/state. Every stage, including planning,
implementation and review, runs on that recorded commit after main advances,
and the fix PR is committed on it; nothing is silently rebased. Publishing
requires the pinned commit to remain in main's history (`stale_base` otherwise),
and a PR GitHub reports as conflicting holds with `merge_conflict` instead of
waiting for CI that never runs; that draft PR stays open, so close it before
requesting `/triage` again or the new run stops at the competing-fix check. The bot's comments do not invalidate that input.
An issue edit or closure stops the workflow (`issue_changed`). Duplicate
checks run at intake and again before publishing/readying a fix. No backlog is
automatically imported.

## Isolation and authority

The public coordinator verifies GitHub HMAC signatures, repository identity and
installation ID. Its private model gateway pins Astra/high and enforces spending.
Only the separate publisher holds the App private key. Its installation tokens
are restricted to `inspect-robots`.

The runner uses a minimal authenticated container supervisor. It accepts only a
fixed stage start and status reads, never arbitrary shell/file-management RPCs.
The generic Cloudflare Sandbox management server is not started. A separate
native Codex exec-server runs repository tools under a different Unix UID from
the Codex client. Provider, GitHub and checkpoint credentials are absent from
tool environments, process arguments and writable source trees.

Canonical source, issue context, policies and schemas are root-owned with
non-writable parents. Only implementation work copies and scratch directories
are writable by repository tools. A trusted launcher captures bounded regular
UTF-8 file changes after stopping tool processes. Symlinks, hardlinks, binary
changes, traversal, hidden files, workflow changes, automation tools, credentials,
agent guidance and dependency-manifest changes are rejected. Such fixes need a
human-reviewed path; the bot reports that limitation.

GitHub publication is restricted to deterministic `issue-bot/` branches and
bot-authored draft PRs. The accepted code digest must match the published files.
Existing unrelated branches/PRs are never taken over. Ready requires the same
head, unchanged source inputs and a successful GitHub Actions `ci-ok` check.

## Setup and operations

Install isolated dependencies with `npm ci`. Register a separate GitHub App:

```sh
node scripts/setup.mjs register
```

Open the displayed local link, create `robocurve-issue-bot`, and install it only
on `robocurve/inspect-robots`. The manifest requests Contents write, Issues write,
Pull requests write and Checks read. The helper stores generated credentials in
ignored `.secrets/github-app.json` with mode 0600. It verifies the complete App
installation, rather than merely narrowing a broad installation token.

`node scripts/setup.mjs install` verifies an existing installation and updates
non-secret config IDs. `node scripts/setup.mjs secrets` securely uploads the
private key, webhook secret and OpenAI key. By default the OpenAI key is read
from `.secrets/openai-api-key`; override its path with
`ISSUE_OPENAI_KEY_FILE`. Never print or commit either key.

Deploy with `ENABLED=false` until checks and isolated probes pass. The three
configs name separate services and do not alter the PR reviewer deployment.
Use `/health` to inspect enablement. Enable only after App setup and validation.

New `issues.opened` events trigger triage. Only Jay's immutable GitHub user ID
can request `/triage` in an issue comment. Explicit retries share the original
issue budget and cannot duplicate an active issue job. A management-only
Cloudflare workflow payload `{"issue":401}` can test a Jay-authored existing
issue; `{"inspect":true}` reports saved state without model spending or posting.

Set `ENABLED=false` to stop new stage admission and provider requests. Existing
result cleanup and pending public notices can still be reconciled. An expired
or ambiguous stage produces a concrete hold; it is never treated as approval.

A launch error is kept as evidence rather than treated as failure, because the
container may still have started. If the container is still not running (no
launch claim, a stopped container, or a supervisor that never accepted `/start`)
five minutes after the stage was prepared, the coordinator stops waiting instead
of running out the 45-minute timeout. A stage that never reached the model is
replaced once per job with a fresh sandbox, which costs another $0.10 container
allowance. A second failure, or any stage that already made a model request,
holds with the recorded reason, e.g. `stage_launch_failed (stage_launch_uncertain)`
or `stage_container_stopped`. Public reasons carry only fixed error codes; prose
error text goes to Worker logs as `issue_stage_launch_error`.

## Verification

```sh
npm run types
npm run check
python3 -m unittest discover -s sandbox -p 'test_*.py'
docker build --platform linux/amd64 -t robocurve-issue-bot:local .
docker run --rm --platform linux/amd64 --network none \
  --entrypoint /opt/issue-env/bin/python \
  -v "$PWD/sandbox:/tests:ro" \
  -v "$PWD/Dockerfile:/Dockerfile:ro" \
  -v "$PWD/.dockerignore:/.dockerignore:ro" robocurve-issue-bot:local \
  -m unittest discover -s /tests -p 'test_*.py'
```

Worker tests mock external services. Native container tests use synthetic model
responses and spend no API credits. The repository CI includes this package's
checks in `ci-ok`; core lint, typing and 100% coverage gates remain required.
Actual deployment and trial evidence belongs in `DEPLOYMENT.md`.
