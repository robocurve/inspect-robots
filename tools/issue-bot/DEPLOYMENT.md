# Issue bot rollout evidence

Rollout date: 2026-09-21. Repository base:
`4d35fe4b81a643c0e61e8d287810b3c26f9d386a`.

## Installation and authority

Separate GitHub App `robocurve-issue-bot`, App ID `5017788`, installation
`163417136`. Setup verified a selected-repository installation containing only
`robocurve/inspect-robots`, with Contents write, Issues write, Pull requests write,
Checks read and implicit Metadata read. Private credentials remain ignored.

GitHub ruleset `23732858` restricts default-branch updates with bypass only for
human team `19612023`, through pull requests. Neither bot App is a bypass actor.
Ruleset `18603265` separately requires the up-to-date `ci-ok` check. The publisher
has no merge or issue-closure operation. Preserve these rulesets: Contents write
alone is not a technical prohibition on merging.

## Disabled deployment

Cloudflare account: `7f405baff0972dc740a02ee0f700d2c1`.

| Service | Initial version | Exposure |
| --- | --- | --- |
| `robocurve-issue-publisher` | `32c43b9b-c4e3-4e8f-8970-39ba5702a450` | Private service binding |
| `robocurve-issue-runner` | `6b6ab2d9-4561-4290-9864-ae773b8eebd0` | Private service binding; one container maximum |
| `robocurve-issue-bot` | `a8c18df5-673d-49df-bd17-ea6dffe93fbc` | Signed webhook and public health check; processing disabled |

Endpoint: `https://robocurve-issue-bot.jay-7f4.workers.dev`.
Live health returned `{"ok":true,"enabled":false}`; an unsigned webhook returned
HTTP 401. The management-only `initial-state-check` workflow completed without
model calls or public writes.

The runner uses an authenticated fixed-purpose Python supervisor, with no generic
Sandbox management server. Initial deployed image digest:
`sha256:973e57469240c92a13fb7fde09fcb0aecfe803b501c2660ac70b7815fe23a935`.

The reviewed command-normalization correction was then deployed, still disabled:
coordinator `44f70a65-7dd3-4f4f-ba6a-c65819718bcc`, runner
`4752dab9-189d-49be-a809-33a09a339abe`, image digest
`sha256:ac63911a71227f681afd8ca1d1d0ed48328075d788a9f83389f97df0ac277819`.

Final client-only environment authentication runner:
`eade74f4-0259-4bc2-80ba-a7b711d16364`, image digest
`sha256:08b7f1d3528c7349a7967eeb1432ed183c93fd56b4c2f0132e1e60203ba57fdd`.
The user subsequently approved uploading the credentials and explicitly approved
activation and the #401 trial. The dedicated OpenAI key and both GitHub secrets
were uploaded to their named private services. Enabled coordinator version:
`2904f553-fa56-4a65-9f62-c7450907c2c3`. Live health reports enabled.

## Verification and outstanding rollout

Core regression suite: 1,720 passed, six optional rerun-sdk skips, 100% coverage.
Core Ruff and mypy passed. Bot TypeScript and all 57 Worker tests passed. All 26
Python tests passed in the Linux container with external networking disabled,
including the real native Codex fixture using synthetic responses. The fixture
verified that native command records normalize to the original shell script.
Python Ruff checks passed. The final independent functional reviewer approved
after the command-evidence normalization correction and client-only environment
authentication change. The capability is absent from the Codex configuration
file, process arguments and tool environment. Initial GitHub CI exposed a test
interpreter path assumption; using the active interpreter resolved it and the
issue-bot job passed. CodeQL also prompted explicit read-only CI permissions and
fixed-origin URL construction in the App setup helper.

The pretrial management inspection returned an empty queue. Workflow
`live-trial-401` enqueued job `f6eafe68895e28fcaef23121414879d7f2e0ddae3fcf2f9d`
at 2026-09-21 07:47 UTC. Stage `af9c95d9-c2da-47c7-bc0e-b4edcacd2d2e` entered
triage. It completed at 07:50 UTC and published
[DUPLICATE on #401](https://github.com/robocurve/inspect-robots/issues/401#issuecomment-5757167152),
tagging `@jeqcho`. Recorded lifetime issue spending was $1.126, including the
$0.10 container allowance; this is ledger accounting, not a provider invoice.

Astra independently reproduced missing final JSON logs for SafetyAbort and
EmbodimentFault raised from observe_parked and before_scoring on trial 40. Both
rollout-halt controls and the normal control wrote their expected final logs.
The supplied open-PR evidence identified #404 as an existing fix, so the bot
correctly avoided a competing implementation. The assessment explicitly states
that #404's implementation, physical hardware and the full suite were not
validated. The durable checkpoint was consumed, container cleanup succeeded,
and the single public assessment was acknowledged in the outbox.

No fix PR was created in this duplicate trial. Consequently, the automatic
plan/implementation/review path and bot-authored ready-event handoff to the
separate PR reviewer are covered by offline tests, but not a live fix trial.

Selected trial issue: [#401](https://github.com/robocurve/inspect-robots/issues/401),
authored by `jeqcho`. Existing open fixes made a duplicate assessment the correct
outcome. No competing PR was manufactured to demonstrate the fix path.

Approved limits: $20 per issue lifetime and $200 per UTC month, including the
trial, all model stages and conservative container allowances.
The user set the shared OpenAI project budget to $400/month for both bots;
this does not raise either bot's own $200/month allowance.

## Comment format and status correction

On 2026-09-21, deployed the requested `STATUS`, `Issue summary`, owner mention,
and `Your action` layout. `FIX_PROPOSED` now identifies an existing open fix PR;
`DUPLICATE` refers only to another issue. Missing-information requests mention
the issue author, with bot/deleted-author fallback to `@jeqcho`; review decisions
mention `@jeqcho`.

The #401 trial comment was corrected in place to `FIX_PROPOSED`, preserving its
original evidence, spending and historical deduplication marker. No additional
model run was performed. The predeployment queue/outbox inspection was empty,
all 57 Worker tests and TypeScript passed, and independent review approved.

- Publisher: `adcdd321-ac07-4f73-aa20-f7784f16f37f`
- Coordinator: `56be36f5-dd7a-4c4e-b113-c5d87abbddc7`
- Runner: `0330dd2f-f4c8-43dc-9ed6-046a9f20a082`
- Image: `sha256:6bd48d77597659588aca1d56fa266ac96d186da68341bd7e324d2600526f576f`

## Backlog queue admission correction

The 2026-09-21 jeqcho backlog exposed a freshness bug: two main merges held
waiting jobs and interrupted active read-only triage. Before further retries,
changed the coordinator to pin main at FIFO admission and allow an unconsumed
triage stage to finish on its recorded immutable base. Edited issues still stop;
later fix stages and publisher checks still reject stale bases. Job identities
and lifetime charges are retained. No plan or approved artifact is rebased.

All 62 Worker tests and TypeScript passed; independent review approved. The
predeployment queue, outbox, and running Workflows were empty. Coordinator
version `baecba13-51b9-470f-8933-3675095b4664` contains the repair. Publisher and
runner were not changed. Backlog scope is issues 7, 236, 263, 303, 332, 354, 355,
370, 396, 407, and 408; the completed #401 trial is excluded. Recovery uses
request ID `backlog-queue-repair-20260921` without resetting prior spending.
