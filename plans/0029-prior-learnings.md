# 0029 — `prior_learnings`: feed a past run's lessons into the agent's system prompt

Issue: #196. Status: draft.

## Problem

Agent transcripts are write-only. `LLMAgentPolicy.on_trial_end` persists the
conversation to `<log_dir>/transcripts/<run_id>/<scene>-e<epoch>.jsonl`
(agent `policy.py:389-400`) and nothing in the codebase reads it back.
`--epochs` rebuilds the conversation from scratch every trial
(`reset()`, agent `policy.py:365-383`), and `retry_attempts` on `eval_set()`
is a documented inert stub (`eval.py:544-547`). An agent that failed a task
yesterday starts today knowing nothing.

This is the consumer half of a retry-with-learning loop. The producer half
(plan 0028, issue #195) distills a log into a markdown learnings file at
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
alongside the existing guided-`ConfigError` checks (house convention, #168):

- unreadable/missing file → `ConfigError` with a `fix:` line;
- empty/whitespace-only file → `ConfigError` (an empty learnings file is
  always a mistake upstream, better loud than silently absent);
- file larger than 32 KiB → `ConfigError` telling the user to summarize it
  first (the producer's output is ~1-2 KiB; a 32 KiB ceiling stops someone
  pointing this at a raw transcript and silently bloating every system
  prompt).

The text is read once at construction and stored; `reset()` must not do I/O
per trial. A sha256 of the content is computed at the same time.

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
prior_learnings: str | None = None          # the path, as passed
prior_learnings_sha256: str | None = None   # content hash at construction
```

`eval()` already serializes the config dataclass into
`EvalSpec.policy_config` (`eval.py:291`), so every log self-documents both
that learnings were injected and *exactly which* learnings (the hash pins the
content even if the file is later edited). No core changes needed.

## Files

- `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py` —
  constructor param + validation, config fields, `reset()` block.
- `plugins/inspect-robots-capx/src/inspect_robots_capx/policy.py` — same.
- `plugins/inspect-robots-agent/tests/`, `plugins/inspect-robots-capx/tests/`
  — new cases.
- README / agent plugin docs — one short "retry with learning" subsection
  showing the two-command loop with plan 0028's `summarize`.

## Testing

- Happy path: construct with a tmp learnings file, `reset()`, assert the
  system message contains the delimiter line and file text after the
  embodiment-docs block, and that `config.prior_learnings` /
  `config.prior_learnings_sha256` are set.
- Default off: no param → system prompt byte-identical to today's; config
  fields `None`.
- Error paths: missing path, empty file, oversize file — each a `ConfigError`
  whose message contains a `fix:` line.
- Interaction: learnings + embodiment docs both present → order is template,
  docs, learnings.
- Epochs: two `reset()` calls both carry the learnings block (state survives
  reset; no re-read — assert by deleting the file between resets).

## Out of scope (follow-ups)

- Honoring `retry_attempts` to run summarize→inject automatically between
  attempts.
- Auto-discovery of "the latest learnings file for this task".
- Accumulating learnings across more than one prior run.
