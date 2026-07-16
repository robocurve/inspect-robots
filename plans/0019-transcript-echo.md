# 0019 — `-P transcript_echo=true`: follow the agent conversation live on stderr

Issue: #123. Status: draft.

## Problem

During a run the agent policy's conversation lives only in
`LLMAgentPolicy._messages` and is persisted to the eval log at rollout end
(plan 0015). The operator standing at the rig cannot see what the model is
thinking or which tool it called until the run is over. The Rerun viewer shows
frames and actions live, but not the text.

## Design

A single opt-in constructor param on the agent plugin. No core changes.

`LLMAgentPolicy(transcript_echo=False)` — when true, the policy prints
conversation events to **stderr** as they happen. The CLI already coerces
`-P transcript_echo=true` to a real `bool` (`parse_value` in
`_defaults.py`), so no parsing work is needed. Stderr, not stdout: the run's
summary/log-path lines go to stdout and must stay machine-consumable.

No LLM is involved anywhere: every echoed line is string formatting of values
the policy already holds in local variables.

### What gets echoed, and where the calls sit in `policy.py`

All echo sites are inside `reset()` / `act()`. Most sit right where the
corresponding message is appended to `self._messages`; two are deliberately
one-way (echo without a transcript message, or message without an echo), as
called out below:

| Event | Line format (one `print` each) |
|---|---|
| `reset(scene)` | `[agent] goal: {scene.instruction}` |
| observation appended in `act()` | `[agent] >> step {n}: {k} camera(s), state[{key}]: {rounded}` (one line; **never** the base64 payloads) |
| assistant message returned | `[agent] << {text}` for the text content (verbatim, may be multi-line) and one `[agent] << tool_call {name}({arguments})` per tool call |
| tool result appended (executed call) | `[agent] -- {result.error or result.note}` |
| extra tool calls in one turn | `[agent] -- ignored: one tool call per turn` (one per extra, mirroring the filler message appended for each) |
| budget exhausted (`_forced_give_up`) | `[agent] -- LLM call budget exhausted; forcing give_up` |

Not echoed: the retry nudge (`"Respond with exactly one tool call."`) — it is
bookkeeping the plugin generates, not model output — and the system prompt
built in `reset()` (static template plus embodiment docs; the goal line
carries the per-trial information). The ignored-filler for extra tool calls
*is* echoed (unlike the nudge) because without it an operator seeing two
`tool_call` lines followed by one `--` result could not tell which call
executed. The convention follows the code (`call, *extras =
message.tool_calls`; `raw()` preserves wire order): the **first**
`tool_call` line is the executed call, each subsequent line is an extra,
then one `-- ignored` line per extra, and the final plain `--` line is the
executed call's result — exactly matching the transcript order. Do not
reorder the echo to put the executed call last; the lines mirror the
message order verbatim.

Two sites are one-way by design and must not be "fixed" into symmetry:
`reset()` appends two messages (system + goal) but echoes only the goal, and
`_forced_give_up` echoes a line while appending nothing to `_messages` (the
synthetic give_up call never enters the transcript today; this plan does not
change that).

The observation summary reads the step from `observation.extra["env_step"]`
(reserved by plan 0018; rollout injects it) with the **same
`isinstance(step, int)` gate the prompt label uses** — extract the current
suffix expression into a shared helper so any `env_step` value renders
identically in both places (string and `np.int64` are rejected; `True` is
accepted and renders as `step True`, since bool is an int subclass — the
pinned fallback test asserts exactly that, so do not add a bool
exclusion); a bare presence
check would let echo and prompt drift (the prompt's fallbacks are pinned by
`test_observation_content_step_label_fallbacks`). When the gate rejects,
the step field is omitted: `[agent] >> observation: ...`. State lines reuse
the exact `state[{key}]:` rendering already produced for the prompt — the
summary calls the same formatting helper, so echo and prompt can never
drift. To keep it one line, multiple state keys are joined with `" | "`.

### Implementation sketch

```python
def _echo(self, text: str) -> None:
    if self._transcript_echo:
        print(text, file=sys.stderr, flush=True)
```

- `flush=True` per line: the whole point is liveness; stderr is typically
  line-buffered under a TTY but block-buffered when piped (e.g. `2>&1 | tee`).
- The state-text formatting currently inlined in `_observation_content()` is
  extracted into a module-level helper `_state_lines(observation,
  state_labels) -> list[str]` used by both the prompt builder and the echo
  summary. Pure refactor; prompt bytes are unchanged.
- Assistant echo: `message.raw()` is the wire dict. `content` may be `None`
  or `str` (this client never receives list content from the API). Echo the
  string form only when non-empty; then iterate `message.tool_calls`
  (already parsed `ToolCall` objects) for the `tool_call` lines —
  `arguments` is echoed as the raw JSON string from the wire, not re-encoded.

### Config surface

- Constructor: `transcript_echo: bool = False`.
- `AgentPolicyConfig` gains `transcript_echo: bool = False` so the setting is
  recorded in `EvalSpec.policy_config` like every other inference-time knob.
- No validation needed (any Python truthiness is acceptable; annotate `bool`).

## Versioning

`plugins/inspect-robots-agent` 0.7.0 → **0.8.0** (new feature, minor). Update
the version pin in `plugins/inspect-robots-agent/tests/test_package.py`
together with `pyproject.toml`, and run `uv lock` (workspace lockfile tracks
plugin versions).

## Tests (plugin suite; not under the core 100% gate but plugin CI runs them)

In `plugins/inspect-robots-agent/tests/` (reuse the existing fake-transport
harness from `test_policy_e2e.py`):

1. Default off: a full reset→act cycle writes nothing to stderr (capsys).
2. `transcript_echo=True`: stderr contains, in order, the goal line, a
   `>> step 0:` observation summary, the assistant `<<` echo (text and
   `tool_call` line), and the `--` tool-result line. The existing response
   helpers are either/or (`_tool_response` hardcodes `content: None`,
   `_text_response` has no tool_calls), so add a small combined
   text-plus-tool-call response helper for this test.
2b. Multi-tool-call turn: two `tool_call` lines followed by one
   `-- ignored: one tool call per turn` line and the executed call's `--`
   result, in that order, and the **first** `tool_call` line names the
   executed call. Use the existing `_multi_tool_response` helper in
   `test_policy_e2e.py` — no new helper needed for this test (the combined
   text-plus-tool-call helper is only for test 2).
2c. `flush=True` is pinned: run one echoing `act()` with `sys.stderr`
   monkeypatched to a fake TextIO recording `flush()` calls (`print`
   forwards `flush=True` to the stream); assert at least one flush per
   echoed line. capsys alone cannot observe flushes.
3. The observation summary contains the camera count and `state[...]` text
   but no `data:image` substring (the no-base64 guarantee).
4. Missing `env_step` (direct `act()` with plain extra): summary line uses the
   stepless form and does not raise.
5. Budget exhaustion path echoes the forced-give-up line
   (`max_llm_calls=1`, second `act` → forced give_up).
6. Assistant message with no text content (tool-call only): no empty `<<`
   text line is printed.
7. `AgentPolicyConfig` round-trip: `transcript_echo=True` lands in
   `policy.config`.

## Docs

- Plugin README: the README has no parameter table — the knobs live in a
  prose "Configuration knobs (all `-P key=value`)" sentence. Add
  `transcript_echo` to that list and follow with one sentence on the stderr
  format plus the `-P transcript_echo=true` example. Do not introduce a new
  table. Follow the writing-style rules (no em dashes, no mid-sentence bold).

## Out of scope

- Streaming the transcript into Rerun (plan 0020 / issue #124).
- Echoing token usage or latency (the `inference` transcript event already
  records latency).
- Truncating or wrapping long assistant messages: terminals wrap, and
  truncation would destroy information the operator opted into seeing.
