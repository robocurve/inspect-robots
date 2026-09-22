# 0082: Automatic independent PR review

Approved scope: implement and test a GitHub App reviewer for inspect-robots,
then deploy in advisory mode. Required-check enforcement is a separate rollout.
Main baseline: 4d35fe4b81a643c0e61e8d287810b3c26f9d386a.

Use gpt-6-astra with high reasoning. Every revision starts a fresh context with
repo policy, full diff, relevant source/tests, issue context and human decisions.
Evaluate necessity, scope, invariants and test integrity. Require explicit
maintainer judgment where existing policy is insufficient. Produce APPROVE,
REQUEST_CHANGES or ESCALATE and MERGE/REVISE/CLOSE/NEEDS_DECISION recommendation.
Only Jay merges or closes. Scope approval does not waive correctness or CI.

Two Workers isolate credentials. The public receiver/review workflow holds the
OpenAI key and webhook secret. A private publisher holds the GitHub App key and
only exposes bounded repository reads and validated comment/check publication
over a service binding. No generic write proxy, merge, close, label or shell API.
GitHub's native PR write permission remains broader than the publisher interface.

SQLite Durable Object state for this repository records deduplication, per-head
review jobs and atomic budget reservations. Limits: $5/review, $15/PR lifetime,
$200/calendar month UTC; warn at $160. Reserve the maximum input/output charge
before each inference; uncertain failed requests retain the reservation. Disable
SDK/workflow automatic inference retries to avoid unaccounted spend. Reconcile
known usage, including reasoning. Missing configuration/budget/inspection holds.

Webhooks: new/reopened/updated/ready PRs and jeqcho's explicit /review command.
Verify signature, repository and installation IDs. Fetch authoritative head; bind
verdicts to head/base SHAs. Superseded runs cannot publish success on a new head.
Drafts defer review. Ignore closed PRs and bot feedback loops. A scheduled sweep
only handles already registered jobs and CI notification; never imports backlog.
Human scope decisions are explicit /review scope <sha> <rationale> comments from
GitHub user ID 42904912 and retained for fresh reviews. No automatic scope/budget
waivers via untrusted free text.

Offline tests must cover signature verification, input validation, stale heads,
publisher endpoint restrictions, schema/verdict consistency, concurrent budgets,
monthly rollover, unknown charge reservation, webhook deduplication and CI gating.
Deploy disabled, upload secrets without displaying them, exercise health and
signed controlled test events, then enable app webhooks in advisory mode. Record
actual deployed versions and any external configuration blocks in the runbook.

The numpy-only framework, existing test assertions and merge rules are retained.
