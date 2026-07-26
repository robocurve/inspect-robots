# 0027 — Agent on-demand camera capture and chained tool calls

Closes #173.

## 1. Problem

The agent policy attaches every camera frame to every observation message
(`_observation_content`). Two costs follow.

**Token cost.** On a 60-step trial with two cameras the frames dominate the
bill, and the model has no way to say "the state vector is enough right now".

**Round-trip cost.** `act()` executes the first tool call of an assistant turn
and answers every extra with `ignored: one tool call per turn`. The natural
loop for a model that controls its own perception is "move, then look at where
that got me". Without chaining that costs a whole extra LLM call after the
motion has already finished playing.

Chaining a capture onto a motion is only correct if the frames come from the
observation delivered *after* the speed-limited chunk has played out. The move
tools synthesize an N-step interpolation whose length depends on
`max_speed_frac`, `control_hz`, and the distance travelled; a capture that
resolved against the pre-motion observation would hand the model a picture of
where the arm *was* while telling it the motion is done.

## 2. Scope

In scope, all inside `plugins/inspect-robots-agent`:

- A policy-level `images` mode selecting force-fed frames (today's behavior) or
  on-demand capture.
- A `take_pic` tool, exposed only in on-demand mode.
- Chained tool calls: a motion plus a trailing `take_pic` in one assistant
  turn, with the capture resolved against the post-motion observation.
- Arrival accounting so a motion cut short by early replanning or termination
  is reported as such rather than silently labelled post-arrival.

Out of scope: changes to the core rollout, controller, or approvers; video or
multi-frame capture; chaining two motions; a `take_pic` equivalent for state
(state text is cheap and stays force-fed).

## 3. Design

### 3a. `images` mode

`LLMAgentPolicy.__init__` gains `images: str = "always"`, validated against
`{"always", "on_demand"}` and recorded on `AgentPolicyConfig` so the eval log
carries it. The default preserves today's behavior exactly, so existing
benchmark numbers stay comparable and no CLI invocation changes meaning.

`"always"`: `_observation_content` behaves as it does today, and `take_pic` is
not in `schemas()`. Nothing in this plan is reachable.

`"on_demand"`: observation messages carry the state text plus one line naming
the cameras that `take_pic` can reach. No image parts.

### 3b. `take_pic`

Exposed by `Toolset.schemas()` only in on-demand mode:

```
take_pic(cameras?: string[], note: string)
```

`cameras` omitted means every camera. `note` is required, matching the move
tools: the note stream is the human-readable narration the user follows live,
and a capture is a decision worth narrating ("I cannot tell whether the gripper
cleared the rim, so I am looking before descending").

Validation lives in `Toolset.execute` because it holds the observation:

- unknown camera name → error naming the valid names
- `cameras` present but not a list of strings, or empty → structured error
- the observation carries no images at all → error saying so

`build_toolset` refuses `on_demand` at bind time when
`observation_space.cameras` is empty, with a message naming the fix (drop
`-P images=on_demand`). Failing at bind rather than mid-trial matches how every
other unsupported configuration in this module behaves, and an embodiment that
serves frames without declaring them is already outside the compatibility
contract.

`take_pic` yields neither a chunk nor an error. `ToolResult` gains
`capture: tuple[str, ...] | None` holding the resolved camera names. The policy,
not the toolset, decides whether that capture resolves now or after a motion,
and writes the tool-result text accordingly, because that decision depends on
loop position rather than on the observation.

### 3c. Immediate capture

A `take_pic` that is the turn's only call, or precedes the motion in the turn,
resolves against the current observation. The policy appends:

1. a `tool` message: `captured 2 frame(s): 'top', 'wrist'`
2. a `user` message whose parts are the labelled image parts, built by the same
   helper `_observation_content` uses so the `camera 'top' (step 480):` join key
   into stored frames stays identical

and then loops for another LLM call. The capture spends one `max_llm_calls`
unit; budget exhaustion already forces `give_up`.

Two `user` messages in a row (tool results, then frames) is the shape the
Anthropic wire already produces today between turns, so no wire client changes.

**Repeat guard.** Frames cannot change without stepping the robot, so within
one `act()` each camera is revealed at most once. The policy tracks the
revealed set for the current observation; a request for cameras already shown
returns `error: camera 'top' was already captured for this observation and
cannot change until the robot moves`. A request mixing new and already-shown
cameras succeeds for the new ones and says which were skipped.

### 3d. Chained capture

`act()` stops answering extra tool calls with a blanket `ignored` and instead
walks the turn's calls in order:

- calls before the first chunk-producing call: `take_pic` executes immediately
  (3c); anything else is answered `ignored: only one motion per turn`
- the first chunk-producing call (a move, `done`, or `give_up`) is executed as
  today
- calls after it: a `take_pic` is **queued** and answered
  `queued: frames arrive with the next observation, after the motion completes`;
  anything else is answered `ignored: only one motion per turn`. After `done` or
  `give_up` a queued capture can never be delivered, so it is answered
  `ignored: the trial ends with this call` instead

Every `tool_call_id` still gets exactly one answer before the next assistant
turn, which is what the wire requires.

The queue holds one `_PendingCapture(cameras, issued_step, chunk_len)`. A
second queued `take_pic` in the same turn merges its camera names into it.

### 3e. Arrival accounting

`_PendingCapture` records `issued_step` (`observation.extra["env_step"]` when
the motion was issued) and `chunk_len` (the number of actions in the emitted
chunk). At the top of the next `act()`, the policy compares the new
`env_step` against `issued_step + chunk_len`:

- advanced by `chunk_len` or more: frames are labelled
  `camera 'top' (step 42, after the motion completed)`
- advanced by fewer steps: frames are still attached, and the observation
  message carries a line
  `the motion was interrupted after 3 of 12 steps; this is not the requested
  target position`. A short advance means the controller replanned mid-chunk
  (`DefaultController(replan_interval=k)`) or the embodiment terminated, both of
  which are outside the policy's control and neither of which may be reported as
  arrival.
- `env_step` missing or not an `int` on either observation (a direct
  `policy.act()` call outside `rollout()`): fall back to the neutral label
  `camera 'top' (after the motion)` with no step arithmetic, matching how
  `_step_label` already degrades.

The guarantee this buys is exactly the one the issue asks for: the frames come
from the observation the rollout produced after playing every action of the
speed-limited chunk, so a slow embodiment or a long interpolation delays the
picture rather than taking it early.

A pending capture is consumed by the next `act()` whatever happens, so it can
never leak into a later observation. `reset()` clears it along with the
revealed set.

### 3f. System prompt

The on-demand mode swaps two sentences of `_SYSTEM_TEMPLATE`:

- perception: cameras are not attached automatically; call `take_pic` to see
  them
- turn shape: exactly one motion per turn, and `take_pic` may be chained in the
  same turn — placed after a motion it returns frames once the motion has
  finished, placed alone it looks before deciding

The `always` mode keeps today's text verbatim.

### 3g. Echo and transcript

`transcript_echo` gains lines for captures (`[agent] -- captured 2 frame(s)`,
`[agent] -- queued capture: 'top'`). `_sanitize` already replaces image parts
with omission markers, so persisted transcripts and the Rerun stream need no
change.

## 4. Files

```
plugins/inspect-robots-agent/
├── src/inspect_robots_agent/
│   ├── _tools.py          # take_pic schema + validation, ToolResult.capture,
│   │                      # build_toolset(images=...) bind-time refusal
│   └── policy.py          # images knob, capture bookkeeping, chained-call walk,
│                          # arrival accounting, on-demand system prompt
│   README.md              # images mode, take_pic, chaining, arrival semantics
└── tests/
    ├── test_tools_motion.py   # schema exposure, argument validation, bind refusal
    └── test_policy_e2e.py     # on-demand flow, chaining, arrival vs interruption
plans/0027-agent-on-demand-vision.md
CHANGELOG.md
```

## 5. Testing

Unit (`test_tools_motion.py`):

- `take_pic` absent from `schemas()` in `always` mode, present in `on_demand`
- `build_toolset(images="on_demand")` raises `ToolsetError` naming the fix when
  the observation space declares no cameras
- unknown camera name, non-list `cameras`, empty `cameras`, missing `note`,
  and an imageless observation each return a structured error, not a raise
- omitted `cameras` resolves to every camera in the observation

End-to-end (`test_policy_e2e.py`, fake transports as today):

- on-demand observation message carries no image parts and names the cameras
- immediate `take_pic` appends the tool result and a user message whose parts
  are image parts, and the loop continues to a second LLM call
- a second `take_pic` for the same camera in the same `act()` errors
- a move chained with `take_pic` returns the chunk from `act()`, answers both
  tool calls, and attaches frames to the *next* observation message
- the chained capture labels frames post-arrival when the next `env_step` has
  advanced by the full chunk length, and reports interruption when it has not
- a `take_pic` chained after `done` is answered as ignored and never delivered
- `always` mode is byte-identical to today's message stream (regression guard)
- `images` appears in `AgentPolicyConfig` and an invalid value raises

Gates: `ruff check`, `ruff format --check`, `mypy --strict` over
`plugins/inspect-robots-agent/src/inspect_robots_agent`, and the plugin test
suite. Core coverage is untouched; plugin coverage stays report-only.

## 6. Risks

**A blind model.** In on-demand mode a model that never calls `take_pic` drives
on proprioception alone and will score worse. This is the point of the knob and
the reason `always` stays the default; the on-demand system prompt says plainly
that images exist and how to get them.

**Capture spam.** A model could burn its budget looking. The per-observation
repeat guard removes the only *useless* case (the same camera twice with no
motion between), and `max_llm_calls` bounds the rest.

**Consecutive failures.** A capture is a success but does not reset the
`_MAX_CONSECUTIVE_FAILURES` counter, so an alternating error/capture loop still
ends the trial after three errors. That is the intended reading: the errors are
what is persistent.
