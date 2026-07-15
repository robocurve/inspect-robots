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
...
parts.append({"type": "text", "text": f"camera {name!r}{suffix}:"})
```

- `isinstance(step, int)` guards the fallback: under an older core (no
  injection) or a non-int value, the label is exactly today's, so the plugin
  keeps its "no minimum core version bump" property. `bool` is an `int`
  subclass but the rollout only ever injects real ints; guarding `bool` out
  adds an untestable branch for a value that cannot occur — keep plain
  `isinstance(step, int)`.
- The elision placeholder in `transcript()` stays
  `[image omitted: streamed camera frame]` — the join key lives in the
  adjacent label part, which the persisted transcript keeps verbatim.
- The system prompt is not changed; the label is self-explanatory in context.

### Versioning

The agent plugin's static version is already `0.6.0`, unreleased (bumped by
PR #115 which is not yet on PyPI). This feature rides `0.6.0`; no further
bump. Core needs no version action (hatch-vcs, next tag).

### Docs

- `Observation.extra` docstring in `types.py`: document the `env_step`
  reservation ("the rollout injects the current step for the policy-facing
  observation; embodiments should not set this key").
- Agent plugin README: one line in the observation-format section showing the
  labeled form `camera 'top_cam' (step 480):` and the join to
  `{trial}_{camera}_{t:06d}.npy`.
- `docs/guide/logging-and-rerun.md`: extend the transcript paragraph — the
  step label is the join key from transcript observations to stored frames.

## Testing (100% branch coverage for core; agent plugin's own suite)

Core (`tests/`):
1. Rollout injection: a probe policy records the observations it receives;
   assert `extra["env_step"] == t` for each call and that the observations in
   `record.steps` / `sink.log_step` do NOT contain `env_step` (original obs).
2. Collision: a mock embodiment that sets `extra={"env_step": "theirs"}`;
   the policy sees the rollout's int.
3. Sharing, not copying: the policy-facing observation's `images` mapping is
   the same object as the embodiment's (identity assert), pinning the
   shallow-copy claim.

Agent plugin (`plugins/inspect-robots-agent/tests/`):
4. `_observation_content` with `extra={"env_step": 480}` → label text
   `camera 'top_cam' (step 480):` precedes the image part.
5. Without `env_step` (and with a non-int value) → today's unlabeled text.
6. Transcript round-trip: build a conversation via the policy, call
   `transcript()`, assert the label text survives elision next to the
   `[image omitted: ...]` placeholder.

## Out of scope

- Storing `FrameRef` paths in the transcript (duplicates the join the step
  label already provides, couples the plugin to FrameStore layout).
- A `transcript` CLI viewer (possible follow-up once labels exist).
- Changing the `Policy` protocol signature.
