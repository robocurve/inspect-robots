# 0018 — Camera frames in agent transcripts carry the environment step

Issue: #117. Status: draft.

## Problem

Persisted policy transcripts (plan 0015) elide image bytes: each frame becomes
a text part `[image omitted: streamed camera frame]`. The camera name is
recoverable from the preceding label part (`camera 'top_cam':`), but the
environment step is recorded nowhere, so there is no exact join from a
transcript observation back to the `.npy` frames the policy actually saw
(`{trial}_{camera}_{t:06d}.npy` in the FrameStore). Chunk lengths quoted in
tool-result strings allow only a fragile textual reconstruction.

Root cause: the policy never learns the step. `Observation` has no step field
and `Policy.act(observation)` receives nothing else; only `rollout()` and the
`Controller` know `t`.

## Design

Two small changes: core exposes the step to the policy through the existing
`Observation.extra` channel; the agent plugin embeds it in the camera label
text at message-build time. The label serves both audiences at once — the LLM
gains explicit temporal context, and the persisted transcript becomes
self-describing with zero elision-time logic.

### Core: rollout injects a reserved `extra` key

In `rollout()`, the observation handed to the controller (and thus to
`policy.act`) gains the current step under a reserved key:

```python
action = controller.next_action(
    policy, replace(obs, extra={**obs.extra, "env_step": t}), t, store
)
```

- `dataclasses.replace` on the frozen `Observation` is a shallow copy; images
  and state arrays are shared, not copied.
- Only the policy-facing observation is modified. `sink.log_step(t, obs, ...)`
  and the `StepRecord` keep the embodiment's original observation, so logs,
  Rerun streams, and stored records are byte-identical to today (`log_step`
  already receives `t` separately).
- Key name `env_step` (not `step`) to reduce collision odds with embodiment
  extras. If an embodiment emits its own `env_step`, the rollout's value wins;
  the `Observation.extra` docstring documents the reservation. No embodiment
  in this repo or the first-party plugins emits `env_step` today (verified by
  grep).
- The initial `policy.reset(scene)` carries no observation and is unaffected;
  the first `act` sees `env_step == 0`.

This is additive for every policy: dict consumers that ignore unknown keys see
no change. Not a `Policy` protocol change; no API-snapshot impact
(`Observation`'s field set is unchanged).

### Agent plugin: step-labeled camera parts

`_observation_content` in `plugins/inspect-robots-agent/.../policy.py`:

```python
step = observation.extra.get("env_step")
suffix = f" (step {step})" if isinstance(step, int) else ""
# suffix is computed ONCE, before the camera loop (env_step is
# per-observation, not per-camera), then reused for every label:
parts.append({"type": "text", "text": f"camera {name!r}{suffix}:"})
```

- `isinstance(step, int)` guards the fallback: under an older core (no
  injection) or a non-int value, the label is exactly today's, so the plugin
  keeps its "no minimum core version bump" property. `bool` passes this guard
  (`bool` subclasses `int`): on a new core the rollout's overwrite makes that
  unreachable, and under an older core an embodiment that sets
  `env_step=True` produces a cosmetic `(step True)` label — no worse than an
  embodiment injecting a wrong-but-real int, which no guard can detect. Keep
  plain `isinstance(step, int)` and pin the `True` behavior with a plugin
  unit test so the choice is deliberate, not accidental. Same deliberateness
  for numpy ints: an older-core embodiment emitting `env_step` as `np.int64`
  fails `isinstance(step, int)` (numpy scalars don't subclass `int`) and
  silently gets no label — consistent with the fallback framing, unreachable
  on a new core (rollout injects a Python `int`), pinned in test 5.
- The elision placeholder in `transcript()` stays
  `[image omitted: streamed camera frame]` — the join key lives in the
  adjacent label part, which the persisted transcript keeps verbatim.
- The system prompt is not changed; the label is self-explanatory in context.

### Versioning

The agent plugin's `0.6.0` is already on PyPI (released as part of v0.13.1,
2026-07-15). Because `skip-existing` makes an unchanged plugin version a
silent no-op at release time, this PR MUST bump the plugin's static version —
to `0.7.0` (new feature) — or the feature never publishes. Core needs no
version action (hatch-vcs, next tag).

### Docs

- `Observation` class docstring in `types.py` (there is no per-field
  docstring for `extra` today, and the class docstring does not mention it):
  add the `env_step` reservation ("the rollout injects the current step into
  the policy-facing observation's `extra`; embodiments should not set this
  key").
- Agent plugin README: the README has no observation-format heading (its
  headings are Install / Quickstart / How it works); anchor the new line to
  the existing `transcript()` sentence (~line 95, inside "How it works"),
  showing the labeled form `camera 'top_cam' (step 480):` as the join key to
  stored frames.
- `docs/guide/logging-and-rerun.md`: write a short "Policy transcripts"
  paragraph from scratch — plan 0015's transcript persistence was never
  documented here (the file currently has no mention of transcripts). Cover:
  what is persisted, image elision, and the step label as the join key from
  transcript observations to stored frames. Precision caveat to include:
  FrameStore sanitizes trial and camera names (`_safe()` in frames.py) before
  building `{trial}_{camera}_{t:06d}.npy`, so for camera names containing
  characters the sanitizer rewrites, the authoritative mapping is
  `StepRecord.image_refs` / `FrameRef.path`, not string assembly.

## Testing (100% branch coverage for core; agent plugin's own suite)

Core (`tests/`):
1. Rollout injection: a probe policy that emits single-action chunks (so
   `act()` fires at every step and each call maps to a known `t`) records the
   observations it receives; assert the recorded `extra["env_step"]` sequence
   equals `[0, 1, ..., n-1]` exactly, and that the observations in
   `record.steps` / `sink.log_step` do NOT contain `env_step` (original obs).
2. Merge, not replace: a mock embodiment that sets
   `extra={"env_step": "theirs", "unrelated": 1}` on EVERY observation it
   returns (`reset()`'s and each `StepResult.observation` — `obs` is
   reassigned per iteration, so reset-only extras would vacuously pass past
   t=0); the policy-facing observation has `env_step == t` (rollout's int
   wins) AND `extra["unrelated"] == 1` (embodiment extras survive the merge —
   kills the `replace(obs, extra={"env_step": t})` mutant that drops
   `**obs.extra`).
3. Sharing, not copying: the policy-facing observation's `images` mapping is
   the same object as the embodiment's (identity assert), pinning the
   shallow-copy claim.

Agent plugin (`plugins/inspect-robots-agent/tests/`):
4. `_observation_content` with `extra={"env_step": 480}` and TWO OR MORE
   cameras → every camera label carries the suffix (`camera 'top_cam'
   (step 480):`, `camera 'left_cam' (step 480):`), each preceding its image
   part — a single-camera test would pass a "label only the first camera"
   mutant.
5. Without `env_step` (and with a non-int value, e.g. a string) → today's
   unlabeled text; with `env_step=True` → `(step True)` (pins the deliberate
   bool behavior from the design section).
6. Transcript round-trip: build a conversation via the policy, call
   `transcript()`, assert the label text survives elision next to the
   `[image omitted: ...]` placeholder.

## Out of scope

- Storing `FrameRef` paths in the transcript (duplicates the join the step
  label already provides, couples the plugin to FrameStore layout).
- A `transcript` CLI viewer (possible follow-up once labels exist).
- Changing the `Policy` protocol signature.
