# 0029 — `prior_learnings`: feed a past run's lessons into the agent's system prompt

Issue: #196. Status: revised after one critique round (CaP-X error-class
reality acknowledged; `-P` coercion edge cases and cap semantics pinned).

## Problem

Agent transcripts are write-only. `LLMAgentPolicy.on_trial_end` persists the
conversation to `<log_dir>/transcripts/<run_id>/<scene>-e<epoch>.jsonl`
(agent `policy.py:389-400`) and nothing in the codebase reads it back.
`--epochs` rebuilds the conversation from scratch every trial
(`reset()`, agent `policy.py:365-383`), and `retry_attempts` on `eval_set()`
is a documented inert stub (`eval.py:544-547`). An agent that failed a task
yesterday starts today knowing nothing.

This is the consumer half of a retry-with-learning loop. The producer half
(plan 0028, issue #195 — developed in parallel on its own branch, not yet in
this tree) distills a log into a markdown learnings file at
`<log_dir>/learnings/<log_stem>.md`. Contract between the halves: plain
markdown, treated as opaque here, path always explicit — this plan works with
a hand-written notes file just as well as with a generated one.

## Design

One new constructor parameter on `LLMAgentPolicy`, mirrored on `CapxPolicy`:

```
-P prior_learnings=logs/learnings/adhoc_084f91f0.md
```

reachable today through the existing `-P k=v` passthrough (`cli.py:151`,
`_parse_kvs`) — no new CLI surface.

### Construction (fail fast)

`prior_learnings: str | None = None` — a filesystem path. At `__init__`,
every failure raises `errors.ConfigError` with a `fix:` line (house
convention, #168), because that is the class `_resolve_or_exit`
(`cli.py:464-476`) converts into a guided message — a `ValueError` would
surface as a raw traceback. In the agent plugin this matches the neighboring
checks; **CaP-X currently raises plain `ValueError` throughout and never
imports `ConfigError`** (capx `policy.py:157-171`), so there the new checks
deliberately break local style: add `from inspect_robots.errors import
ConfigError` and raise it for `prior_learnings` failures anyway. Migrating
CaP-X's existing `ValueError`s to `ConfigError` is a worthwhile follow-up,
not this plan.

Checks, in order, before touching the filesystem and then on the content:

- not a `str` (the `-P` parser coerces: `-P prior_learnings=` → `""`,
  `=none` → `None` which silently disables the feature, `=42` → `int`,
  `_defaults.py:34-55`) — `""` and non-`str` non-`None` values →
  `ConfigError` naming the coercion and showing the quoted-string escape
  hatch; `None` stays "feature off" since that is the default;
- unreadable/missing file → `ConfigError` with a `fix:` line;
- empty/whitespace-only file → `ConfigError` (an empty learnings file is
  always a mistake upstream, better loud than silently absent);
- decoded text longer than `_PRIOR_LEARNINGS_TEXT_LIMIT = 32 * 1024`
  characters → `ConfigError` telling the user to summarize it first (the
  producer's output is ~1-2 KiB; the ceiling stops someone pointing this at
  a raw transcript and silently bloating every system prompt). The constant
  is duplicated in both plugin packages — they share no code — with a
  cross-reference comment in each so the copies can't drift silently.

The text is read once at construction and stored; `reset()` must not do I/O
per trial. A sha256 of the stored text is computed at the same time.

### Prompt assembly

In `reset()`, after the existing embodiment-docs block and with the same
concatenation idiom (agent `policy.py:369-372`):

```python
if self._prior_learnings_text is not None:
    formatted = (
        formatted
        + "\n\nNotes from a previous attempt at tasks like this one. They may "
        + "be wrong or stale; the current observation always wins:\n"
        + self._prior_learnings_text
    )
```

The framing line is part of the design, not decoration: injected lessons are
hints, and an agent that trusts a stale note over its own camera is worse
than a cold-start agent.

CaP-X (`plugins/inspect-robots-capx/.../policy.py:335-359`) gets the identical
block in its `reset()` — same template-then-docs-then-learnings order.

### Eval hygiene: memory-assisted runs must be distinguishable

A run with injected learnings is not statistically poolable with cold-start
runs. Recording rides the existing config plumbing:
`AgentPolicyConfig` (agent `policy.py:117`) and `CapxPolicyConfig`
(capx `policy.py:107`) each gain

```python
prior_learnings: str | None = None          # resolved absolute path
prior_learnings_sha256: str | None = None   # hash of the injected text
```

`eval()` already serializes the config dataclass into
`EvalSpec.policy_config` (`eval.py:291`), so every log self-documents both
that learnings were injected and which learnings. The recorded path is
resolved to absolute (a cwd-relative path recorded verbatim may not resolve
later). Precise provenance claim: the hash *identifies* the injected text —
it lets you verify a candidate file, but if the file is edited or deleted
the text itself is recoverable only from transcript sidecars (which embed
the system message). We accept hash-as-identity rather than storing the text
in every log; the learnings file lives in `log_dir` alongside the logs, so
in practice it survives with them. No core changes needed.

## Files

- `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py` —
  constructor param + validation, config fields, `reset()` block.
- `plugins/inspect-robots-capx/src/inspect_robots_capx/policy.py` — same.
- `plugins/inspect-robots-agent/tests/`, `plugins/inspect-robots-capx/tests/`
  — new cases (plugins run ruff D1 + `mypy --strict`: new params and config
  fields need docstrings and full annotations; plugin coverage is
  report-only, but the new branches are all cheap to cover).
- Both plugin READMEs document the new parameter; top-level README (and the
  Docusaurus docs if they gain a page for this) get one short "retry with
  learning" subsection showing the two-command loop with plan 0028's
  `summarize`.

## Testing

- Happy path: construct with a tmp learnings file, `reset()`, assert the
  system message contains the delimiter line and file text after the
  embodiment-docs block, and that `config.prior_learnings` /
  `config.prior_learnings_sha256` are set.
- Default off: no param → system prompt byte-identical to today's; config
  fields `None`.
- Error paths: missing path, empty file, oversize file, `""` (the
  `-P prior_learnings=` coercion), and a non-string value (e.g. `42`) — each
  a `ConfigError` whose message contains a `fix:` line, raised in both
  plugins.
- Interaction: learnings + embodiment docs both present → order is template,
  docs, learnings.
- Epochs: two `reset()` calls both carry the learnings block (state survives
  reset; no re-read — assert by deleting the file between resets).

## Out of scope (follow-ups)

- Honoring `retry_attempts` to run summarize→inject automatically between
  attempts.
- Auto-discovery of "the latest learnings file for this task".
- Accumulating learnings across more than one prior run.
