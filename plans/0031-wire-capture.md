# 0031 — Wire capture: the log records 100% of what the LLM saw

Closes robocurve/inspect-robots#206.

## Problem

The persisted transcript diverges from actual wire traffic four ways:

1. **Tools never persisted.** `Toolset.schemas()` (`_tools.py:113`) — the
   move tool name, bounds text, and valid dimension labels the model is
   given — is sent on every request and appears nowhere in the log.
2. **Images stripped.** `_sanitize()` (`policy.py:107`) replaces every
   `image_url` part with `[image omitted: streamed camera frame]` in both
   `policy_transcripts` and the JSONL sidecar written by `on_trial_end`
   (`policy.py:523`). Depth composites (`depth="render"`) are never stored.
3. **Evicted-view divergence.** The wire sends `_evicted_view()`
   (`policy.py:123`) — only the last `image_horizon` frame messages, elision
   stubs; on the Anthropic wire the anchor becomes a `cache_control`
   `{"type": "ephemeral"}` breakpoint in the translated body — while the log stores the
   full un-evicted conversation. The logged object is not the object the
   model received.
4. **Per-call bodies unrecoverable.** Retries, per-attempt status codes,
   exact request params, and raw response payloads are not captured
   anywhere.

Debugging "why did the model do that?" (e.g. today's #206 trigger: auditing
whether a run's model was told about `move_joints` at all) currently
requires inference from code, not evidence from the log.

## Design

Replay-grade, always-on capture at the wire-client serialization point —
the only place the actual outbound object exists — plus viewer integration
so what-the-model-saw is browsable per call.

### Capture sink (`plugins/inspect-robots-agent/src/inspect_robots_agent/_capture.py`)

New module, stdlib-only (`json`, `hashlib`, `base64`, `pathlib`, `time`).

```python
class WireCapture:
    def begin_trial(self, log_dir: str, run_id: str, trial_id: str) -> None: ...
    def record(self, *, attempt: int, endpoint: str, request: dict,
               status: int | None, response_text: str | None,
               error: str | None, t_start: float, duration_s: float) -> None: ...
    def end_trial(self) -> str | None:  # relative path for record.metadata
    @property
    def began(self) -> bool: ...  # True once begin_trial has ever run on this sink
```

Clients pass the raw `response.text` (or `None` on transport error); the
**sink** does one guarded `json.loads` — the parsed value when it is a
JSON object, the first 2000 chars of text otherwise — so the dict-vs-text decision lives in one tested place,
not three clients, and `record()` can run before the client's own
`response.json()` raise path.

The sink owns the `call` index: clients cannot know trial boundaries, so
they never pass one. `record()` increments the sink's internal call
counter whenever `attempt == 0` (a retry shares its call's index); the
counter resets in `begin_trial`; the first call of a trial stamps `call: 0`. Clients pass `t_start` (their
`time.time()` taken before **that attempt's** post — with backoff
sleeps, attempts diverge by seconds) and `duration_s`, both per-attempt.

- `begin_trial` stores the resolved `<log_dir>/wire/<run_id>/<trial_id>/`
  target; directories and `calls.jsonl` are created lazily on the first
  `record()`, so a zero-LLM-call trial (e.g. `reset()` raising before any
  `act()`) leaves no empty files, and `end_trial` returns `None` when no
  row was written (no metadata pointer for empty capture). `<trial_id>` is
  `f"{scene_id}-e{epoch}"` with the scene id passed through a local
  replica of core `_safe` (`frames.py:21-33`) **in full — character
  class plus the crc32 disambiguation suffix** (stdlib-only, no private
  core import): scene ids are foreign text; a `/` or `..` must not nest
  or escape the wire dir, and without the crc32 suffix two ids differing
  only in unsafe characters would silently append into the same
  `calls.jsonl`, cross-attributing trials. The metadata pointer records the sanitized path.
  (The transcript sidecar at `policy.py:533` predates this and stays
  as-is.)
- `record` deep-copies the request, walks it for image parts, and replaces
  each with a blob reference; blob bytes go to
  `<log_dir>/wire/<run_id>/blobs/<sha256>.png` (one `blobs/` dir per run,
  shared across trials — frames repeat across attempts and dedupe by
  content hash; write skipped when the file exists). One JSON line per
  **attempt** — retries are calls the provider saw, so each is a row.
  **Each `record()` flushes its row to disk before returning** (append +
  flush; no buffered handle may hold the final row) — crash durability is
  the design's rationale, and the row a SIGKILL would strand in a buffer
  is exactly the call the run died on.
- Image-part shapes recognized, per wire:
  - chat: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
  - responses: `{"type": "input_image", "image_url": "data:image/png;base64,..."}`
  - anthropic: `{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}`
  Substitution happens at the **base64-payload** level, not the part
  level. The coherent triple, stated once as the contract:
  - **What is replaced:** exactly the base64 payload text — the substring
    after `base64,` in a chat/responses data URL, or the whole
    `source.data` value on the Anthropic wire (which is bare base64) —
    becomes the sentinel `$blob:<sha256>`. Data-URL prefixes and **every
    key of the part are preserved verbatim**, including `cache_control`,
    which the Anthropic translation attaches to the final turn's last
    block (`_anthropic.py:289-298,400-406`), i.e. routinely to an image
    part.
  - **What is hashed and stored:** the sha256 is over the **decoded PNG
    bytes**, which are what `blobs/<sha256>.png` holds (directly viewable
    files).
  - **How readers find and restore references:** scan string values for
    `\$blob:[0-9a-f]{64}` (unanchored — the sentinel may follow a
    data-URL prefix); restore by substituting the sentinel with
    `base64(blob file bytes)`. Wire-agnostic, no re-wrapping.
- JSONL row schema (the **format contract**, documented in the module
  docstring and `docs/`):

```json
{"call": 3, "attempt": 0, "endpoint": "/chat/completions",
 "t": 1753668000.123, "duration_s": 2.41,
 "request": {"model": "...", "messages": [...], "tools": [...]},
 "status": 200, "response": {...}}
```

  `response` is the parsed body whenever it parses as a JSON **object**
  (any status), else the first 2000 chars of text; transport errors record
  `"status": null`, `"response": null`, and `"error": "<str(exc)>"` (the
  `error` param; omitted from the row when `None`). `call` is the sink's
  0-based logical call index (attempts of one call share it). `t` is the
  caller's `t_start`.

**Row-write ordering is the forensic core.** Each client calls `record()`
at the point where (status, body-or-error) is first known and **before any
raise or parse that could skip it**: before the non-retryable-4xx raise
(`_llm.py:218-226`, `_anthropic.py:161-165`), before `response.json()`
parsing that could throw on a malformed 200, and before the Anthropic
terminal-`stop_reason` raise (`_anthropic.py:424-426`) — that last one is
an ordinary 200 row; the raise is client-layer semantics, not wire truth.
The calls the run *died on* are exactly the ones capture exists for.

### Core hook: `on_trial_start`

`PolicyBase` (`src/inspect_robots/policy.py:96`) gains
`on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None`
— a `# noqa: B027` no-op default mirroring `on_trial_end`, invoked by the
rollout in `eval.py` immediately before the first observation of each
trial, symmetric with the existing `on_trial_end` call site. Backward
compatible: existing policies inherit the no-op. This exists so capture
*streams* — a crashed or killed run keeps every call captured up to the
failure, which is precisely the run you want forensics on. (The
`on_trial_end`-only alternative would buffer capture in memory and lose it
on crash; rejected.)

The core `Policy` Protocol docstring enumerates "four optional hooks"
(`src/inspect_robots/policy.py:57-59`); that contract text is updated to
five as part of the change. The addition is additive with a no-op default,
the same shape `on_trial_end` took when it landed.

**Hook failure contract.** At hook time no `TrialRecord` exists yet (the
record is born inside `rollout()` or synthesized by the `PolicyError`
fallback, `eval.py:367-373`), so "degrade the record" is not expressible.
Instead: `eval.py` wraps the invocation; on exception it **skips the
rollout** and synthesizes an errored `TrialRecord` exactly as the
`PolicyError` fallback does — keeping `SceneResult`'s parallel tuples
(`epochs`, `operator_judgements`, `operator_notes`, `trial_metadata`,
`termination_reasons`, `policy_transcripts`, `eval.py:456-469`) aligned —
counts it toward `errored_trials`/`fail_on_error`, and fires the sink
bus's trial-end — but **not** `policy.on_trial_end`: the transcript
contract is "at trial end after a successful `reset()`" (core
`policy.py:67-68`, honored via `policy_reset_ok` at `rollout.py:329-330`),
and `reset()` lives inside the skipped `rollout()`; firing the policy hook
here would let this very plugin persist the *previous* trial's still-held
`_messages` under the new trial's id. A raising hook must never lose the
whole log. This carves an exception into `on_trial_end`'s documented
"called for errored and cancelled trials too" promise (core
`policy.py:62-65` and `:105-111`) — both docstrings are amended in the
same commit: "...except trials whose `on_trial_start` raised, which never
reached `reset()`". (Call site: immediately
before `rollout()`, **after** the sink-bus `on_trial_start` at
`eval.py:331` — so a raising policy hook still yields a matched sink
`on_trial_start`/`on_trial_end` pair, preserving the sink protocol's
documented per-trial order (`logging/sink.py:3-5`). Same hook name,
different protocol; both docstrings note the distinction.)

**Sink lifecycle contract.** `record()` outside an open trial —
before `begin_trial`, after `end_trial`, or when `begin_trial` never ran —
is a **no-op**. `begin_trial` resets all per-trial state (target dir,
lazy-file handle, call counter, dead flag), so a failed or skipped
`begin_trial` can never bleed rows into a previous trial's file. This also
defines version skew: a plugin running against an older core whose
`eval()` never calls `on_trial_start` simply captures nothing; the
plugin's `on_trial_end`, seeing `capture.began` false with
`wire_capture=True`, prints one stderr note **once per policy instance**
(`[agent] wire capture inactive: core predates on_trial_start`) instead
of writing a metadata pointer. `began` is a permanent latch —
set by the first `begin_trial`, cleared by nothing — so the plugin may
call `end_trial` first and consult `began` after without a spurious skew
warning. (`began` also disambiguates
"begun-but-zero-rows" — silent, no pointer — from "never begun".)

### Wire-client integration

Each of the three clients (constructors `_llm.py:171`,
`_responses.py:21`, `_anthropic.py:51`) takes
`capture: WireCapture | None = None` at construction and, in `complete()`, wraps the existing post: on every
attempt, after the response (or transport error) is known, calls
`capture.record(...)` with the exact `body` it posted. No control-flow
changes; the retry loop, error guidance, and parsing stay as they are.

`LLMAgentPolicy.__init__` grows `wire_capture: bool = True` (recorded in
`AgentPolicyConfig`); when true it owns a `WireCapture` passed to the
client it constructs, `on_trial_start` calls `begin_trial`, and
`on_trial_end` calls `end_trial` and — when it returns a path — sets
`record.metadata["wire_capture"] = "wire/<run_id>/<trial_id>/calls.jsonl"`
next to the existing `metadata["transcript"]` pointer. The `end_trial`
call happens **before** the existing empty-transcript early return
(`policy.py:525-527`), so a trial whose LLM calls all failed still gets
its capture pointer.

### Failure semantics

Capture must never fail an eval. The sink's public methods are wrapped
whole: on the first exception of **any type** (`OSError`, decode errors,
serialization surprises — the invariant is absolute, so the catch is
too) the sink prints one
warning line (`[agent] wire capture disabled for this trial: <err>`) **unconditionally to
stderr** — not via the policy's `_echo`, which is gated on
`transcript_echo=False` by default (`policy.py:841-843`) and reachable only
from the policy, not the wire clients that call `record()`. The sink is
self-contained: it owns its warning path, marks itself dead for the trial,
and subsequent `record()` calls no-op. A run with broken capture is a run
with a warning, not a failed trial. Capture never mutates the live request (`record`
receives the body after the post result is known; blob substitution
operates on a deep copy).

### Viewer: HTML report (`src/inspect_robots/_html.py`)

`render_html` already renders per-trial chat transcripts with optional
frame images (`_FrameContext`, `_frame_image` → data-URI `<img>`). New:
when `record.metadata["wire_capture"]` resolves, each trial section gains a
collapsible **"Wire"** subsection: one `<details>` per call/attempt showing
endpoint, status, duration, the request's non-`messages` prompt-bearing and
param fields per wire — on the Anthropic wire that includes `body["system"]`
(where the system prompt and `prior_learnings` live, `_anthropic.py:122-129`)
— rendered **once per trial** like tools, with a note on any call where
it changed (it is built once at `reset()` and can carry 32 KB of
`prior_learnings`; repeating it per call would defeat the text bounding)
— plus model, effort/temperature, tool count — the messages — **delta-rendered**: each call shows only the messages
new since the previous row, plus a one-line note per earlier message
whose as-sent form changed (eviction stubs arriving), with elision stubs
and `cache_control` breakpoints visible on whatever is shown — and blob
images. Delta rendering is what bounds the wire section's *text*: the
JSONL embeds the full history per row (quadratic — see Storage), and
repeating it across ~100 `<details>` blocks would contradict the
openable-document rationale the budget exists for. Total rendered text
stays proportional to the transcript. The full as-sent body of any call
remains available via `inspect --wire`.
Tools render once per trial (the toolset is fixed at `bind()`; schemas are
identical across calls by construction).

**Pointer resolution.** No viewer resolves `metadata["transcript"]` today;
the only code resolving it is `_summarize.py:53-68`
(`log_path.parent / pointer` with an `is_relative_to` traversal guard,
because the pointer is foreign text). That helper moves to a new core
module `_pointers.py`, imported by both `_summarize.py` and `_html.py`
(`_summarize` already imports from `cli`, and `cli` imports `_html` — a
shared leaf module avoids the cycle). `render_html` gains a `log_path`
parameter (plumbed from `_cmd_view`) so relative pointers resolve. The
per-run blob dir derives from the pointer as
`pointer.parent.parent / "blobs"` (`wire/<run_id>/<trial_id>/calls.jsonl`
→ `wire/<run_id>/blobs/`). `$blob` values read from foreign JSONL are
validated as 64-char lowercase sha256 hex before any path is built from
them (a hostile `"$blob": "../../x"` is rejected, the reference rendered
as broken).

**Payload budget.** The existing report bounds frame embeds with
`_FrameBudget` (`_html.py:30-37`; the 50 MB default lives in the
`render_html` signature, `_html.py:557`) precisely to keep
documents openable. `render_html` already constructs one budget object unconditionally
(`_html.py:613`); it just never reaches trial rendering without a
`_FrameContext`. The change is threading: pass the existing budget to the
wire renderer too, and generalize the header chip label from "frames
truncated at N MB" to "embedded media truncated at N MB". Each distinct blob is
embedded **once per trial** at its first reference, with a
document-unique anchor id (`wire-<safe_trial_id>-<sha[:12]>`, trial id
passed through `_safe()` as the frames path already does, `_html.py:464`
— scene ids are foreign text and land in an HTML attribute); every later
reference to the same sha renders as an internal link to that anchor —
**only if the anchor was actually emitted**; when the first reference was
budget-denied, later references render the elision placeholder too (the
evicted view re-references the same frames on consecutive calls — naive
inlining would repeat ~10 refs/call × 100 calls into a multi-hundred-MB
document). A blob denied by the budget renders an inline
`[blob elided: media budget]` placeholder (the existing frame path degrades to its `[image omitted]` placeholder
text plus the chip; the wire placeholder names the budget explicitly).

### Viewer: `inspect` CLI (`src/inspect_robots/cli.py`)

`inspect-robots inspect <log> --wire [CALL]`: without `CALL`, prints a
call table for every trial (trial id, call#, attempt, endpoint, status,
duration, image count, new-blob bytes); with `CALL`, pretty-prints **every attempt row** for that call index
(retries share it; the retried call is the forensically interesting one),
request and response, `$blob` references left symbolic.
Because a log holds many trials, the single-call dump takes an optional
`--trial SCENE-eEPOCH` selector; it is required (guided error listing the
trial ids) when the log has more than one trial with capture. Follows the
existing `--transcript` flag's structure (`cli.py:1214`).

### Storage

A 100-call, 3-camera trial with the default `depth="render"` ≈ 6 new
frames/call (RGB + depth composite per camera, `_depth.py:73`) × ~150 KB
≈ 90 MB of blobs. The JSONL is **quadratic in call count** — each row
embeds the full message history to date plus the toolset, and the
responses wire additionally carries `reasoning.encrypted_content` both
ways (`_responses.py:63,81`) — so a 100-call trial's `calls.jsonl` runs
tens of MB, not single digits; text is cheap next to the blobs but not
free. Same order overall as `store_frames=True` (already the rig
default). Not gated in v1;
`wire_capture=false` is the escape hatch. Blobs and JSONL live under the
log dir, so log-dir lifecycle management covers them.

## Tests

Plugin (`plugins/inspect-robots-agent/tests/` — CI gates plugins at
`--cov-fail-under=0`; the 100% bar here is self-imposed, as this
plugin's suite has held to date):
- `test_capture.py`: blob extraction/substitution per wire shape; dedup
  (same frame twice → one file); JSONL row schema; retry rows (attempt>0
  shares the call index; attempt==0 increments it); transport error rows;
  dead-sink semantics after an injected exception; closed-sink no-ops
  (`record()` without `begin_trial` and after `end_trial` — the
  version-skew path); `begin_trial` resets state so a skipped trial
  cannot bleed rows into the previous trial's file; the version-skew
  stderr note fires exactly once per policy instance (latched);
  deep-copy (live body unmutated). All of this lives in the **plugin**
  suite — core CI installs only the root package and cannot import
  `inspect_robots_agent`.
- Row-ordering: a non-retryable 4xx, a malformed-JSON 200, an Anthropic
  terminal-`stop_reason` 200, a responses-wire 200 with
  `payload["status"] == "failed"` (`_responses.py:181-184`), and a
  well-formed-but-wrong-shape 200 (parser `KeyError`, `_llm.py:236`) each
  leave their row in `calls.jsonl` even though `complete()` raises.
- e2e (`test_policy_e2e.py` pattern, mock transport): run a trial, then
  assert the captured request for call N — after inlining blobs — is
  **dict-equal** (compare after `json.loads`; asserting raw bytes would pin
  httpx's serializer) to the body the mock transport received, including
  the evicted view, elision stubs, and (anthropic wire) the
  `cache_control` breakpoint placement produced by
  `_with_cache_breakpoint`; assert `metadata["wire_capture"]` resolves.
- Zero-LLM-call trial (policy error before first call): no `wire/` files,
  no metadata key, `end_trial` returns `None`.
- `wire_capture=false` → no `wire/` dir, no metadata key.

Core (`tests/`, 100% scope):
- `on_trial_start` invoked per trial before first `act()` (CubePick mock),
  no-op default harmless for hook-less policies; a *raising*
  `on_trial_start` skips that trial's rollout, synthesizes an errored
  record with all `SceneResult` parallel tuples aligned, counts toward
  `errored_trials`/`fail_on_error`, does **not** fire
  `policy.on_trial_end` (never-reset trial; sink bus trial-end still
  fires), and the eval completes with a log.
- `_html.py`: wire section renders (blob inlined, stub visible), absent
  cleanly when metadata key missing or files deleted; hostile
  `$blob:../../x`-style values rejected (rendered broken, no path
  escape) — this is a security guard over foreign JSONL and gets its own
  test; budget-denied blob renders `[blob elided: media budget]`; repeat
  reference renders an internal link to the first embed's anchor.
- CLI `--wire` table and single-call dump; missing capture → guided message.
- No API-snapshot change: the hook is a method on `PolicyBase`, not an
  `__all__` entry; `tests/test_api_snapshot.py` cannot register it
  (verified — the snapshot asserts `__all__` membership only).

## Tasks

1. Core: `on_trial_start` hook + eval-loop call + tests.
2. Plugin: `_capture.py` + unit tests (no wiring yet).
3. Plugin: wire-client `capture` param + policy wiring + config field +
   e2e dict-equality tests + docs (format contract).
4. Core: HTML wire section + `inspect --wire` + tests.
5. Docs: README (agent plugin) + `docs/` log-format page; CHANGELOG.

Each task is one commit passing all gates (`ruff check`, `ruff format
--check`, `mypy` strict, `pytest --cov` at 100% in its scope).

## Out of scope

- A replay *executor* (re-sending captured calls); the format contract is
  the deliverable — replay is `jq` + inlining away.
- Capture for non-agent policies (gr00t, molmoact2 run no LLM wire; the
  format is documented so future LLM policies can adopt the sink).
- Redaction: request bodies carry no credentials (auth lives in headers,
  which are never captured).
- Retention/pruning of `wire/` dirs.
