# 0015 — Policy transcript persistence (issue #100)

## Problem

An `agent`-policy episode is unauditable after the fact. The LLM conversation
(assistant text, tool calls, tool results) lives only in
`LLMAgentPolicy._messages` and dies with the process. The eval log records what
the run *was* (config, metrics, termination reasons) and the frames sidecar
records what the model *saw*, but nothing records what the model *said or did*.
For an LLM-driven policy that conversation is most of the forensic value of a
log, especially for failed episodes.

Terminology note: the repo already uses "transcript" for the typed `Event`
stream (`transcript.py`, `TrialRecord.events`) — that stream never reaches the
persisted log. This feature is the **policy transcript** (the policy's own
audit record); all user-facing copy says "policy transcript" or "agent
conversation" so a future event-stream viewer is not boxed out of the bare
word. The CLI flag is `--transcript` (there is no competing flag today).

## Design

### 1. Hook: optional, duck-typed `transcript()` on the policy

Mirrors the existing optional `bind()` hook: not part of the `Policy`
Protocol, so every existing policy remains conformant. `PolicyBase` ships a
default returning `None` (same precedent as `bind()`'s no-op default).

```python
def transcript(self) -> Any | None:
    """JSON-serializable audit record of the policy's decision process for
    the current trial (e.g. an LLM conversation), or None."""
```

Contract (documented in the `Policy` Protocol docstring, alongside `bind()`):

- The framework calls it **once per trial, at trial end** (after the loop
  exits or the trial errors) — but the hook must be idempotent and safe to
  call at any point between `reset()` calls: direct `rollout()` callers (and
  anyone holding the policy object) may call it too.
- It must not mutate policy state, and mutating its return value must not
  affect the live policy. Called on the rollout thread; no thread-safety
  obligations beyond that.
- The return value must be JSON-serializable and *small* (text-scale, not
  binary-scale). Camera images must not be embedded; they are already on disk
  via `frames_dir`. The framework enforces both properties defensively (§3).
- Best-effort: a raising or misbehaving `transcript()` must never change the
  trial's outcome (it degrades to an error marker in the log, see §3).

### 2. Record and log schema (stays schema v1)

- `TrialRecord.policy_transcript: Any = None` — new field, populated by
  `rollout()`. Sinks see it via `on_trial_end` (e.g. a future viewer).
- `SceneResult.policy_transcripts: tuple[Any, ...] = ()` — new field, strictly
  parallel to `epochs` (the established pattern of `operator_judgements` and
  `termination_reasons`): one entry per recorded trial, `None` when the policy
  has no hook (or it returned `None`). Errored trials keep their transcript —
  that is when it matters most.
- `EvalLog.from_dict` coerces the new field with
  `tuple(sample.get("policy_transcripts", ()))` (the `tuple()` wrap matches
  the three existing coercions — a bare list in the frozen dataclass would
  break round-trip equality) so logs written before this field
  existed still read back (the schema's newer-reads-older guarantee; same move
  as the two prior field additions, no version bump).
- `eval()` threads `record.policy_transcript` into the per-scene list in both
  the errored and the scored branches, storing the normalized object as-is.
  Like the other dict-valued log fields, the entries are not deep-frozen;
  extend `log.py`'s shallow-immutability docstring to name
  `policy_transcripts` (a sink mutating the record's transcript in
  `on_trial_end` would alias the log — same standing caveat, now stated).

Storing inline (not a sidecar) is deliberate: after image-stripping the
transcript is metadata-scale (a 100-call conversation is ~100–500 KB of text),
and plan 0001 §3.9 reserves sidecars for large binaries. The size guard below
keeps the inline guarantee honest.

### 3. Collection point: `rollout()`, one site, all post-reset paths

`rollout()` initializes a local `policy_reset_ok = False` **before** the
`try` (a literal implementation without the init would `NameError` inside
`finally` on the reset-failure path and mask the typed error), then sets it
`True` immediately after `policy.reset(scene)` returns. In the existing
`finally` block (next to the latency preservation):

```python
if policy_reset_ok:
    record.policy_transcript = _collect_transcript(policy)
```

The guard prevents stale-transcript misattribution: if `policy.reset()` itself
raised, the policy may still hold the *previous* trial's conversation, and
recording it as this trial's audit data would be actively wrong. Every other
exit path — normal return, mid-trial `PolicyError`, `EmbodimentFault` (incl.
embodiment-reset failure), `SafetyAbort` — is covered by the single `finally`
site. `_record_failure` attaches `record` to the exception *before* the raise
unwinds through `finally`, and the `finally` mutates that same object, so
`exc.record` carries the transcript too.

`_collect_transcript(policy)` (module-private in `rollout.py`), with the
**entire** sequence below inside one `except Exception` backstop so a broken
audit hook can never mask a trial outcome or break the "eval() must always
persist a log" invariant (`BaseException` — e.g. Ctrl-C mid-hook — is
deliberately not caught; no log survives it anyway):

1. `getattr(policy, "transcript", None)`; not callable → `None`.
2. `raw = policy.transcript()`; `None` → `None`.
3. Normalize: `json.loads(json.dumps(raw, default=str))`. This guarantees
   `JsonLogSink` can never crash on a non-serializable transcript, and
   deep-copies so later policy-side mutation cannot alias the record.
   `default=str` makes stray objects (non-float64 numpy scalars such as
   `np.float32`, Paths, dataclass instances) degrade to strings instead of
   failing (`np.float64` is a `float` subclass and serializes as a JSON
   number without hitting `default`). Known lossy edge: a numpy *array*
   leaf becomes its truncated repr string — acceptable degradation, pinned by
   a test so it stays chosen rather than accidental.
4. Size guard: if `len(dumped.encode())` exceeds `_TRANSCRIPT_BYTE_LIMIT`
   (2 MiB), store a small marker instead:
   `{"transcript_dropped": True, "bytes": n, "note": "exceeds inline limit; policies must not embed binary data"}`.
   2 MiB is ~4–20x the stated text-scale worst case but keeps a 100-trial
   overnight run's log bounded (~200 MB absolute worst case vs ~1 GB at
   10 MiB; the limit measures the compact `dumps` — `JsonLogSink` writes
   `indent=2`, so on-disk size exceeds it by a small constant factor).
   Dropping beats truncating: a syntactically-broken half transcript
   is worse than an honest marker.
5. Any exception from steps 2–4 (undumpable keys, circular refs,
   `RecursionError` — all `Exception` subclasses) → error marker
   `{"transcript_error": "<ExceptionType>: <message>"}`, so a broken hook is
   visible in the log instead of silently reading as "policy has no hook".
   Marker construction is made unconditionally safe by formatting the
   message defensively:

   ```python
   try:
       detail = f"{type(exc).__name__}: {exc}"
   except Exception:
       detail = type(exc).__name__
   return {"transcript_error": detail}
   ```

   Both branches are tested (a raising hook, and an exception whose
   `__str__` itself raises) — no untestable line survives for the 100%
   coverage gate.

### 4. CLI

- `inspect-robots inspect LOG.json --transcript` **appends** the policy
  transcript rendering after the standard summary, per scene and trial. Exit
  code stays status-based (`0 if log.status == "success" else 1`) — the flag
  never changes exit semantics. The renderer is shape-tolerant:
  - OpenAI-style chat list (dicts with a `"role"` key): print `role`, text
    content (list-of-parts content collapses non-text parts to `[image]`),
    tool calls as `-> name(arguments)`, tool results indented under the call.
  - Anything else: `json.dumps(..., indent=2)`.
  - No transcripts in the log: print `no policy transcripts recorded` (not an
    error; exit code unchanged).
- Plain `inspect` (no flag) appends a `policy transcripts: recorded
  (--transcript to print)` line when any transcript is present.
- The post-run summary hint gains a second line when the log has transcripts:
  `hint: agent conversation: inspect-robots inspect LOG --transcript`.

### 5. Plugin: `LLMAgentPolicy.transcript()`

Returns a sanitized deep copy of `self._messages`:

- Every `{"type": "image_url", ...}` content part is replaced by
  `{"type": "text", "text": "[image omitted: streamed camera frame]"}`. The
  adjacent `camera '<name>':` text part (already emitted by
  `_observation_content`) keeps the camera attribution readable.
- Everything else (system prompt, goal, state text, assistant messages, tool
  calls, tool results) is preserved verbatim.
- Returns `None` before the first `reset()` (no conversation exists).
- Pure function of current state: does not mutate `self._messages`; mutating
  the returned structure does not affect the live conversation (deep copy).

Plugin version bumps 0.2.2 → 0.3.0 (new public capability).

## Alternatives rejected

- **Sidecar transcript file written by the plugin**: the policy does not know
  the log path/dir (and must not — sinks own persistence), and a sidecar
  breaks the one-file immutable-log story for text-scale data.
- **New `Event` kind in the typed transcript stream**: `TrialRecord.events`
  never reaches the persisted `EvalLog` either; routing chat messages through
  step-indexed events distorts both abstractions (LLM turns are not aligned
  with control steps — one `act()` may span several LLM turns).
- **Collecting in `eval()` instead of `rollout()`**: direct `rollout()` callers
  and sinks would never see transcripts; `rollout()` owns `TrialRecord`
  assembly.

## Tests (TDD, core gate at 100% coverage)

Fixture note: hand-built `SceneResult` fixtures must use already-normalized,
list-only shapes (JSON round-trips turn tuples into lists, so a tuple-bearing
fixture would fail `from_dict(to_dict(x)) == x`).

Core — extend the existing files, do not create parallel ones:

- `tests/test_eval_log.py`: round-trip a log with `policy_transcripts`; a v1
  dict *without* the field reads back to `()` (golden backward-compat path).
- `tests/test_rollout_hardening.py`: hook present → captured on success;
  captured on the partial record of a mid-trial `PolicyError` (via
  `exc.record`); NOT collected when `policy.reset()` raises (stale-state
  guard); hook raising → `transcript_error` marker; exception whose
  `__str__` raises → type-name-only marker; hookless policy → `None`;
  `PolicyBase` default → `None`; `np.float32` leaf → string, numpy-array
  leaf → truncated-repr string (pinned); circular reference →
  `transcript_error` marker; oversized return → `transcript_dropped` marker.
- `tests/test_eval_orchestration.py`: `SceneResult.policy_transcripts`
  parallel to `epochs` for scored and errored trials; hookless policy yields
  all-`None` entries.
- `tests/test_registry_cli.py`: `--transcript` renders chat-shaped and
  unknown-shaped transcripts after the summary; exit code still reflects log
  status with the flag on an errored log; "no policy transcripts recorded"
  path; hint lines (post-run summary and plain `inspect`).
- `tests/test_strict_json.py`: a log carrying a transcript passes the strict
  RFC 8259 write path (non-finite floats in transcript leaves sanitize to
  null).
- API snapshot: unchanged (`SceneResult`/`TrialRecord` already exported; no
  new names in `__all__`).

Plugin (`plugins/inspect-robots-agent/tests/`):

- `transcript()` strips image parts, preserves text/tool structure, returns
  `None` pre-reset, and is decoupled from live state (mutation isolation both
  directions).
- e2e through the existing `httpx.MockTransport` harness
  (`test_policy_e2e.py`): after `eval()` on the mock world, the returned
  `EvalLog` contains the conversation with the scripted tool calls visible and
  no `data:` URLs anywhere in the serialized JSON.

## Docs

- `Policy` Protocol docstring: document the second optional hook (contract
  above); `PolicyBase.transcript()` default with docstring.
- `src/inspect_robots/CLAUDE.md` module map rows (`policy.py`, `rollout.py`,
  `log.py`, `cli.py`).
- `log.py` module docstring: add `policy_transcripts` to the
  shallow-immutability caveat list.
- README: one line in the agent-policy section pointing at
  `inspect --transcript`.
- Plugin README (if it documents the policy surface): mention `transcript()`.

## Files touched

```
src/inspect_robots/
  policy.py        (Protocol docstring + PolicyBase.transcript() default)
  rollout.py       (TrialRecord.policy_transcript, _collect_transcript, finally hook)
  log.py           (SceneResult.policy_transcripts, from_dict coercion, docstring)
  eval.py          (thread record → SceneResult)
  cli.py           (--transcript flag, renderer, hint lines)
  CLAUDE.md        (module map)
tests/
  test_eval_log.py / test_rollout_hardening.py / test_eval_orchestration.py /
  test_registry_cli.py / test_strict_json.py (per above)
plugins/inspect-robots-agent/
  pyproject.toml   (0.2.2 → 0.3.0)
  src/inspect_robots_agent/policy.py (transcript())
  tests/test_policy_e2e.py (+ unit tests)
README.md
plans/0015-policy-transcript-persistence.md (this file)
```
