# Feature 0002 — Temporal-ensembling controller

## Why

The foundation (plan 0001) made action chunking core and the `Controller` a
single-method, stateful middleware *specifically* so an ACT-style temporal
ensembler could compose without forking the rollout loop (resolution R3). Until
that controller exists, the claim is unproven. This feature implements it and
thereby validates the chunking + `ActionSemantics` design end-to-end.

## Background: ACT / π0 temporal ensembling

Chunked VLAs predict `H` future actions per inference. With overlapping
inference (query every control step), timestep `t` is predicted by several recent
chunks: the chunk queried at time `q` predicts `t` via its action at index
`t - q` (valid when `0 <= t - q < len(chunk)`, using each chunk's own length —
chunks may differ in length). Temporal ensembling blends these overlapping
predictions with exponentially decaying weights.

**Weight direction (matches ALOHA/ACT exactly):** gather contributing predictions
in ascending query-time order (**oldest first**) and weight
`w_i = exp(-m * i)`, `i = 0` for the **oldest** prediction. So the *oldest*
prediction gets the *largest* weight and larger `m` makes the oldest dominate —
this is what smooths motion (ALOHA uses `m ≈ 0.01`). Always normalize
`w /= w.sum()` (so `m = 0` ⇒ plain mean).

## Design

`EnsemblingController` (in `controller.py`), implementing the existing
`Controller` protocol:

```python
class EnsemblingController:
    def __init__(self, action_space: Box, m: float = 0.1): ...
    def next_action(self, policy, observation, t, store) -> Action:
        # 1. query policy.act(obs) EVERY step -> chunk; record (latency, len)
        # 2. append (origin_t=t, actions) to a rolling buffer in `store`
        # 3. gather predictions for the current global step from all buffered
        #    chunks: chunk with origin q contributes actions[t-q] if 0<=t-q<H
        # 4. weighted-average with w=exp(-m*age), age = (#preds-1 .. 0) newest=0
        # 5. evict chunks that can no longer contribute (origin <= t-H)
        # 6. return Action(data=blended)
```

- **Global step counter.** `t` passed to `next_action` is the per-rollout step
  index (the rollout already increments it), so it is the correct "global" time.
- **Inference accounting.** It calls `policy.act` every step, so it appends to
  `_controller_inferences` every step (the rollout already reads this for the
  transcript + latency stats). Tests assert `num_inferences == num_steps`
  (contrast with `DefaultController`, where steps > inferences).
- **Semantics safety (R8).** Linear averaging is only valid for additive action
  representations. The constructor inspects `action_space.semantics`:
  - allow: `control_mode` in {joint_pos, joint_vel, eef_delta_pos, eef_abs_pose,
    eef_delta_pose} **only when** `rotation_repr in {"none", "rot6d"}` (rot6d is
    safe to average then re-normalize; we average without renorm for v0 and note
    it). Continuous/none gripper dims average fine.
  - refuse (raise `ValueError`) when `rotation_repr` is a quaternion / euler /
    axis-angle, because naive averaging is incorrect for those — fail loud rather
    than silently corrupt rotations. (A rotation-aware blend is future work.)
  - if semantics are `None`, warn once and proceed (can't verify).

## Testing (TDD)

- Blend math: with `m=0` (uniform weights) and two overlapping length-2 chunks of
  known constant actions, the blended action equals the plain mean; with large
  `m`, it approaches the newest chunk's prediction.
- Inference cadence: over an N-step rollout, `policy.num_inferences == N`.
- Eviction: buffer never exceeds `H` chunks.
- Semantics guard: constructing with a quaternion action space raises; with
  `eef_delta_pos` succeeds; with `None` semantics warns.
- Integration: `eval(CubePick, ScriptedPolicy, controller=EnsemblingController(...))`
  still succeeds (the toy world is `eef_delta_pos`, which is ensemble-safe) and
  produces smoother action norms than `DefaultController` (sanity, not asserted
  strictly).

## Scope / non-goals

- No rotation-aware (SLERP) blending — explicitly refused for now.
- No change to the rollout loop or any other module; this is purely an additive
  controller. Keeps the repo green.
