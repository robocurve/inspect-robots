# 0020 — Stream the LLM transcript onto the Rerun step timeline

Issue: #124. Status: approved (critique rounds 1-4 applied; round 4 found no substantive design issues).

## Problem

An agent-policy run with the Rerun viewer open shows camera frames, state,
and actions live, but the conversation driving those actions is invisible
until the eval log is written. The transcript should scrub alongside the
video: pause the timeline at step 480 and see what the model said there.

## Design overview

Three small pieces, each following an existing pattern in this codebase:

1. **Policy side** — a new optional duck-typed hook `transcript_delta()`
   (same mechanism as the existing `bind()` / `transcript()` hooks): return
   the sanitized transcript messages appended since the previous call.
   Implemented by the agent plugin; absent everywhere else.
2. **Rollout side** — `rollout()` already knows the exact step at which an
   inference happened: the `len(inferences) > prev_inferences` check on the
   controller store that emits `inference_event`. At that seam, fetch the
   delta and hand it to the sink via a new **optional** sink hook
   `log_policy_messages(t, messages)`.
3. **Sink side** — `RerunSink` renders each message as one `rr.TextLog`
   entity row under `{trial-prefix}/llm` on the existing `step` timeline,
   through the existing backpressure-safe worker queue.

Live streaming is best-effort visualization, like everything else the sink
does: any failure degrades to "no live text" and never affects the control
loop, the eval log, or the persisted transcript (which keeps coming from the
end-of-trial `_collect_transcript` path, unchanged).

## Core: the sink hook (`logging/sink.py`, `eval.py`)

`log_policy_messages(self, t: int, messages: Sequence[Any]) -> None`

- **Not added to the `LogSink` Protocol.** The Protocol is structural and
  `runtime_checkable`; adding a sixth method would silently break every
  third-party sink (isinstance checks and mypy conformance). Instead the
  hook is duck-typed and documented in the `sink.py` module docstring as the
  optional extension, mirroring how optional policy hooks are documented in
  `policy.py`.
- `NullSink` does **not** gain a stub either. A no-op on `NullSink` would
  be inherited by every sink built on it ("a convenient base for partial
  implementations"), advertising the hook and opening the `_Broadcast`
  gate below just to feed a no-op — exactly the per-inference
  `transcript_delta()` waste that gate exists to avoid. The hook is purely
  duck-typed. (Note the policy side is *not* a precedent for stubs-off:
  `PolicyBase` does ship `bind()`/`transcript()` defaults; the sink side
  deliberately diverges because a stub here has a per-inference cost via
  the gate, and the divergence is documented.) The contract lives in the
  `sink.py` module docstring: called at most once per control step, only for steps
  where the policy performed an inference. On message shape, the docstring
  words it as an **expectation on policy implementations**, not a
  core-enforced invariant: policies are documented to return plain JSON
  types (the same shape as `TrialRecord.policy_transcript` entries), but the
  core does **not** json-normalize the delta path the way
  `_collect_transcript` does, so sinks must render defensively. Sinks must
  not mutate the messages.
- `_Broadcast` (eval.py) fans to exactly those child sinks where
  `getattr(s, "log_policy_messages", None)` is callable, and — key detail —
  exposes the hook **only when at least one child implements it**, computed
  at construction: `__init__` collects the **getattr-resolved callables**,
  not the sinks (`self._sinks` is `list[LogSink]` and the Protocol has no
  such method, so calling it on a `LogSink`-typed element is an
  `[attr-defined]` mypy-strict error; `getattr` returns `Any`, collected as
  `list[Callable[[int, Sequence[Any]], None]]`), and binds
  `self.log_policy_messages = self._fan_policy_messages` as an *instance*
  attribute only when that list is non-empty (no class-level definition —
  an ordinary `__init__`-inferred attribute, not a `[method-assign]`
  violation, and the API snapshot is untouched since `__all__` doesn't
  change).
  Otherwise the rollout's own `getattr` gate sees no hook and the policy's
  `transcript_delta()` deepcopy never runs — without this, a plain
  JsonLogSink-only run would pay an O(delta) deepcopy every inference and
  discard the result. Accepted residual: a `RerunSink` in the list whose
  `rerun-sdk` turns out missing/disabled still opens the gate (its
  `log_policy_messages` is an unconditional method; availability is lazy
  and only known after construction), so that configuration pays the
  deepcopy for calls that `_ensure_rerun()` turns into no-ops. The cost is
  small and the configuration is transient; do not engineer around it. `JsonLogSink` does not implement the hook (the JSON
  log already gets the full transcript at trial end).

## Core: the rollout seam (`rollout.py`)

Both hooks are resolved **once before the loop** (not per step, and not via
a per-call helper):

```python
delta_hook = getattr(policy, "transcript_delta", None)
messages_hook = getattr(sink, "log_policy_messages", None)
stream_ok = callable(delta_hook) and callable(messages_hook)
```

Inside the existing inference-detection block:

```python
inferences = store.get(_INFER_KEY, [])
if len(inferences) > prev_inferences:
    latency, chunk_len = inferences[-1]
    record.events.append(inference_event(t, latency, chunk_len))
    if stream_ok:
        try:
            delta = delta_hook()
            entries = list(delta) if delta is not None else []
            if entries:
                messages_hook(t, entries)
        except Exception as exc:  # visualization must never kill the loop
            stream_ok = False
            warnings.warn(..., RuntimeWarning)
```

- Note the emptiness test happens **after** `list(delta)`: an exhausted
  generator (or any empty non-list iterable) is truthy as an object, so a
  bare `if delta:` would call the sink with zero entries. Materialize
  first, then test the list.
- `stream_ok` starts as the callable check and doubles as the failure
  latch. If the policy hook or the sink hook raises, emit **one**
  `RuntimeWarning` naming the exception and clear `stream_ok` for the rest
  of the trial — visualization must not spam or crash a multi-hour run.
  (`_collect_transcript` degrades similarly at trial end.) `rollout.py`
  does not currently import `warnings`; add the import, and pass
  `stacklevel=2` to satisfy ruff B028.
- The sink hook is called only when the materialized delta is non-empty.
- Ordering: the delta is fetched *after* `inference_event` is appended and
  *before* `sink.log_step(t, ...)` for the same `t` — sinks see the text at
  the same step index as the frames the model was acting from. (`log_step`
  for step `t` happens later in the same loop iteration, so both land on the
  same timeline point.)
- The hook is consumed only here. The end-of-trial `_collect_transcript`
  still calls `transcript()` and is unaffected: `transcript_delta()`'s
  cursor is internal to the policy, and `transcript()` remains the full
  conversation.

## Core: policy hook contract (`policy.py` docstring)

Document `transcript_delta()` next to the existing `transcript()` hook
documentation. The `Policy` Protocol docstring currently says policies "may
additionally define two optional hooks" and that "`PolicyBase` ships
defaults for both" — reword to **three** hooks and state explicitly that
`transcript_delta` has **no** `PolicyBase` default (deliberate: a default
returning `None` would make every `PolicyBase` policy pass the rollout's
`callable(delta_hook)` check and pay a no-op call per inference; policies
opt in by defining it). `PolicyBase` itself is unchanged.

Contract for the new hook:

- Returns `list` of plain-JSON-type messages appended since the previous
  `transcript_delta()` call (or since `reset` for the first call), or
  `None`/empty when nothing is new.
- Must be O(new messages): implementations sanitize only the delta slice.
- Must already be sanitized (image bytes elided) — the core forwards it
  verbatim to visualization sinks.
- `reset()` must rewind the cursor so a new trial starts from its first
  message.

## Core: RerunSink (`logging/rerun_sink.py`)

New frozen payload dataclass:

```python
@dataclasses.dataclass(frozen=True)
class _TranscriptPayload:
    prefix: str
    t: int
    entries: tuple[tuple[str, str], ...]  # (level, text) per message
```

- `log_policy_messages(t, messages)` **mirrors `log_step`'s exact
  sequence**: `_ensure_rerun()` guard first (return if the SDK is
  unavailable/disabled), render, `_ensure_worker()`, then `_enqueue`. It
  is a public method on a public class, so ruff D102 requires a
  contract-style docstring (state the best-effort semantics and the
  do-not-mutate expectation, not the method name).
  Skipping `_ensure_worker()` would be a real bug, not an optimization:
  on step-error paths a transcript payload can be the *first* thing
  enqueued in a trial, and without the worker-spawn call it would sit in a
  workerless queue, stalling `flush()` for its full timeout at trial end
  with no drop accounting.
- Rendering happens eagerly on the caller thread (pure string work, no SDK
  calls — SDK timeline state stays worker-only), snapshotting
  `self._prefix`.
- Rendering `_render_message(msg) -> tuple[str, str]` (levels are plain
  uppercase string literals, see below):
  - Non-dict message: `("INFO", str(msg))` (defensive; contract says dicts).
  - `role` maps to a Rerun `TextLog` level so the viewer color-codes rows:
    `assistant → "INFO"`, `user → "INFO"` (the observation text is a
    primary thing an operator scrubs for; the `role: ` text prefix keeps
    the two apart), `tool → "DEBUG"`, `system`/other → `"TRACE"`. Rerun's
    levels are **uppercase** (lowercase strings lose the viewer's color
    coding). Pass the string literals `"INFO"`/`"DEBUG"`/`"TRACE"`
    directly to `rr.TextLog(text, level=...)` — the SDK's `TextLogLevel`
    constants are a `str` subclass with exactly these values and the
    `level` parameter accepts any string, so a
    `getattr(rr, "TextLogLevel", ...)` resolution would add an
    SDK-present/absent branch that the fake-`rr` test harness can never
    exercise (100% branch gate). No resolution, no branch.
  - Implementation checkpoint: eyeball one run in a real viewer to confirm
    DEBUG/TRACE rows are visible under the default TextLog view filter;
    the docs subsection documents the level filter either way.
  - Text: `role: ` prefix, then `content` if it is a non-empty `str`; if
    `content` is a list (multi-part observation), join the `text` fields of
    dict parts with newlines and render any non-text part as
    `[{type} part]`; if the message has `tool_calls`, append one
    `tool_call {name}({arguments})` line per call. The wire shape is
    **nested**: each entry is
    `{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}`
    (`_llm.py` stores them exactly as received), so name/arguments come
    from `call.get("function", {})`, not the top level. Defensive `.get`s
    throughout — a malformed entry must render, not raise.
- `_enqueue` / eviction bookkeeping: the queue becomes
  `deque[_StepPayload | _TranscriptPayload]`. The camera-frame watermark
  drop and the `len(evicted.images)` counters apply only to `_StepPayload`
  (isinstance-guarded); an evicted transcript payload increments a **new
  third counter** `_dropped_transcripts` (not `_dropped_steps` — the
  end-of-run warning says "dropped N camera frame(s) and M full step(s)"
  and its "reduce camera bandwidth" advice would be wrong for text rows,
  so transcript drops get their own count reported as "K transcript
  update(s)"). Transcript payloads are small text and never trigger the
  image watermark. Update the warning message and the module docstring's
  degradation ladder ("camera frames are dropped first ... then whole
  steps") to mention transcript rows. Three parts of the drop-report block
  must change together, not just the message: the warning **gate** becomes
  `if self._dropped_frames or self._dropped_steps or
  self._dropped_transcripts:`, and **all three** counters are reset inside
  it — otherwise a run whose only drops are transcript payloads warns
  never, and the stale transcript count leaks into the next eval's report.
  (Branch coverage cannot catch a missed `or` arm, so the transcript-only
  test below is the real guard.) The existing assertion
  `"dropped 1 camera frame(s) and 1 full step(s)"` in
  `tests/test_rerun_sink.py` must survive the reword: the transcript
  fragment is appended only when its counter is non-zero.
- `_shutdown`'s wedged-drain accounting has the same mixed-queue problem:
  today it does `self._dropped_frames += sum(len(p.images) for p in
  self._queue)` over the disowned backlog, which raises `AttributeError`
  on a `_TranscriptPayload`. Guard with the same isinstance split: step
  payloads add their image count to `_dropped_frames` and one to
  `_dropped_steps`; transcript payloads add one to `_dropped_transcripts`,
  consistent with eviction.
- `_emit` dispatches on payload type; `_emit_transcript` does
  `self._set_step(rr, payload.t)` then one
  `rr.log(f"{payload.prefix}/llm", rr.TextLog(text, level=level))` per
  entry. Multiple messages at one step become multiple rows at the same
  timeline point, in conversation order (the worker preserves queue order).
- The `llm` segment joins the existing per-trial namespace
  (`trial/<scene_id>/e<epoch>/llm`), so successive trials never interleave.

## Agent plugin (`plugins/inspect-robots-agent`)

- Extract the elision loop from `transcript()` into a module-level
  `_sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]`
  (deepcopy + image-part elision; the bare-`dict` spelling would fail the
  plugin's mypy-strict `disallow_any_generics`); `transcript()` becomes
  `_sanitize(self._messages)` on non-empty.
- New method:

```python
def transcript_delta(self) -> list[dict[str, Any]] | None:
    """Sanitized messages appended since the previous call (core live-stream hook)."""
    new = self._messages[self._delta_cursor :]
    self._delta_cursor = len(self._messages)
    return _sanitize(new) if new else None
```

- `__init__` and `reset()` set `self._delta_cursor = 0`. The cursor is
  independent of `transcript()`, which stays the full conversation.
- Cost: O(delta) deepcopy once per chunk boundary (seconds apart), the
  elision then drops the only large objects (base64 parts) before the list
  crosses into the sink queue.

## Versioning

- Agent plugin 0.7.0 → **0.8.0** (minor: new hook). If plan 0019 merges
  first and already claims 0.8.0, this becomes 0.9.0 — whichever lands
  second takes the next minor. Update the `test_package.py` version pin and
  `uv lock` together with the bump.
- Core: minor release (new sink capability), version derived from the git
  tag at release time as usual.

## Tests

Core (`tests/`, 100% branch gate):

- **rollout** (extend `tests/test_rollout*.py` with the mock world):
  1. Policy with `transcript_delta` + sink with `log_policy_messages`: hook
     receives exactly the delta at the inference step `t`, before that
     step's `log_step` (capture call order in a recording sink).
  2. Policy without the hook: sink hook never called (gate branch).
  3. Sink without the hook (a plain `NullSink` works — it deliberately
     does not implement the hook): no error, no call (gate branch).
  4. Hook returns `None` / empty list: sink hook not called (both branches).
  5. Hook raises: exactly one `RuntimeWarning`, rollout completes, sink hook
     not called again within the trial (`stream_ok` latch), trial record
     unaffected.
  6. Sink hook raises: same latch behavior.
  7. Non-list iterable delta is materialized (`list(delta)`), and an
     **empty generator** delta does not call the sink hook (the truthy-
     generator trap: emptiness is tested after materialization).
- **eval `_Broadcast`**: fans only to sinks defining the hook; mixed sink
  list works; and when **no** child sink implements the hook,
  `getattr(broadcast, "log_policy_messages", None)` is not callable — the
  rollout gate stays closed and a policy `transcript_delta` spy is never
  called.
- **`NullSink`** does not expose `log_policy_messages`
  (`getattr(NullSink(), "log_policy_messages", None) is None`) — pins the
  deliberate omission so a helpful future refactor doesn't re-add it.
- **RerunSink** (extend `tests/test_rerun_sink.py` fake-`rr` harness —
  note the harness's `fake.TextLog = lambda t: ("TextLog", t)` must first
  grow a `level` keyword, since the real call becomes
  `rr.TextLog(text, level=...)`; without it the new call raises inside
  `_emit`, gets swallowed by `_warn_emit_failure`, and test 1 fails
  confusingly. No existing test asserts on a `TextLog` tuple (nothing
  drives `terminated=True`), so nothing else ripples; the separate 1-arg
  fake in `tests/test_coverage_completion.py` runs only `ScriptedPolicy`,
  which has no `transcript_delta`, and must be left alone):
  1. `log_policy_messages` → worker emits `TextLog` rows at the right
     entity path and step, correct level per role, conversation order.
     For the step assertion the fake harness must also record `set_time`
     calls (today `fake.set_time = lambda *a, **k: None` discards them).
  2. Rendering table: str content; list-of-parts content (text parts joined,
     image part → `[image_url part]`); tool_calls line; non-dict message;
     missing role; empty content with tool_calls only. Fixtures must be
     **`raw()`-shaped**, i.e. tool_calls nested under `"function"` exactly
     as `_llm.py` stores them — a flat `{"name": ..., "arguments": ...}`
     fixture would let a flat-shape rendering bug pass.
  3. Eviction: a transcript payload evicted under queue pressure increments
     `_dropped_transcripts` and neither `_dropped_steps` nor
     `_dropped_frames`; a `_StepPayload` evicted past a queued transcript
     payload still counts its images; the end-of-run warning mentions
     "transcript update(s)" only when that counter is non-zero.
  3b. Transcript-only drops still warn: a run where the **only** drops are
     transcript payloads emits the drop warning (pins the widened
     `or self._dropped_transcripts` gate, which branch coverage alone
     cannot force), and the test asserts **all three counters are zero
     afterwards** — that reset assertion is the real pin (a bare
     "second shutdown is quiet" check would pass trivially via the
     worker-is-None early return, proving nothing about the reset). The
     existing `"dropped 1 camera frame(s) and 1 full step(s)"` assertion
     must keep passing unchanged (frames-and-steps-only run gets no
     transcript fragment).
  4. `log_policy_messages` as the **first** call of a trial (before any
     `log_step`) spawns the worker and the row is emitted — pins the
     `_ensure_rerun`/`_ensure_worker`/`_enqueue` sequence against the
     workerless-queue stall.
  5. Wedged shutdown with a `_TranscriptPayload` in the disowned backlog:
     `_shutdown`'s drain accounting takes the isinstance branch for both
     payload kinds (transcript → `_dropped_transcripts += 1`, step → its image
     count), no `AttributeError`; the existing wedged-shutdown test only
     fills the backlog with `_StepPayload`s, so this covers the new branch
     the 100% gate demands.
  6. Disabled sink: hook is a silent no-op. Use the `sink._disabled = True`
     path to cover the `rr is None` return arc — it is environment-
     independent and truly silent. (The SDK-missing path is *not* silent:
     the first `_ensure_rerun()` emits the "rerun-sdk is not installed"
     RuntimeWarning and needs a `skipif(_RERUN_INSTALLED)` gate, like the
     existing `test_noop_and_warns_when_absent`.)

Agent plugin:

- `transcript_delta()` returns only new messages across two `act()` calls;
  a second call with no intervening messages returns `None` (the
  `if new else None` branch); cursor resets on `reset()`; returned
  messages contain no `data:image` substring; `transcript()` still returns
  the full conversation afterwards.

## Docs

- `docs/guide/logging-and-rerun.md`: new subsection "Live transcript in the
  viewer" — entity path `trial/<scene>/e<epoch>/llm`, how to add a TextLog
  view, the view's log-level filter (tool results log at DEBUG and system
  prompts at TRACE, so show those levels to see the whole conversation),
  note that scrubbing the `step` timeline highlights the rows for that
  step, and that the stream is best-effort (drops under backpressure) while
  the eval log transcript stays complete.
- `src/inspect_robots/CLAUDE.md` module map: update the `policy.py`,
  `rollout.py`, and `logging/` rows to mention the new hook pair.
- `logging/sink.py` and core `policy.py` docstrings as described above.
- Plugin README: one line that the agent policy supports live Rerun
  transcript streaming automatically when a Rerun sink is attached.
- `CHANGELOG.md`: an entry under `Unreleased`/Added referencing #124
  (the repo keeps a manual Keep-a-Changelog file; the API snapshot test's
  docstring instructs noting changes there).

## Out of scope

- stderr echo of the transcript (plan 0019 / issue #123).
- A `TextDocument` "latest message" panel or any custom viewer blueprint;
  the default TextLog view covers the scrub-alongside-video ask.
- Streaming for policies other than the agent plugin (the hook is public
  and documented; VLA policies have no text to stream).
- Replaying transcripts into Rerun from a saved eval log.
