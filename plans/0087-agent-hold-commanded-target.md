# 0087: Hold the last commanded target in absolute moves

Issue: [#595](https://github.com/robocurve/inspect-robots/issues/595)

## Outcome

In absolute modes (`move_joints`, `move_to`), a move starts from the last commanded target, and
unnamed dimensions hold it. The stop action (`done`, `give_up`) holds it too. The observed state
is the reference only until a trial's first accepted move. This replaces two rules: plan 0008
§4c ("unnamed dims hold their current observed value"; "Hold-still = repeat the current observed
state") and the interpolation start in plan 0014 ("from the current observed state").

## Why

A position-controlled gripper closed on an object reads the object's width. Commanding that
opening ends the squeeze. Under the old rules:

- a move that left the gripper unnamed commanded the observed opening and released the object;
- a move that restated the gripper as closed still ramped down from the observed opening, so the
  grip loosened at the start of every move;
- the stop action commanded the observed state, and the rollout steps it once before scoring.

Between agent calls the embodiment holds the last commanded action, so the last command, not the
observation, is the state a move should continue from.

## Design

- `Toolset` keeps the last accepted absolute target. Rejected moves (bounds, fixed dimensions,
  the pre-check, the playout cap) leave it unchanged.
- `Toolset.reset()` clears it, and `LLMAgentPolicy.reset()` calls it at the start of each trial.
- The move tool descriptions say unnamed dimensions "hold their last commanded value (the
  measured value before the first move of a trial)".
- Displacement modes are unchanged.

## Alternative

Hold the last command only for gripper dimensions and keep arm axes on the observed state. It
changes less, but needs a reliable way to tell gripper dimensions apart, and an arm axis pressed
against a surface shows the same jump from its command to its reading. Open for the maintainers
to choose in #595.

## Limits

If an approver rewrites a move, the next move holds the target the tool asked for, not the
rewritten one. That is the target the tool already reports for the residual.
