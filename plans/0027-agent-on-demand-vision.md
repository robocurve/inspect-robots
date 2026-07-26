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

Chaining a capture onto a motion is only useful if the frames come from the
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
- Playout accounting so a chunk cut short by early replanning or termination is
  reported with its step count rather than presented as a finished motion.

Out of scope: changes to the core rollout, controller, approvers, or
`_html.py`; video or multi-frame capture; chaining two motions; a `take_pic`
equivalent for state (state text is cheap and stays force-fed).

No CLI change is required. `-P images=on_demand` reaches the policy through
`_parse_kvs` → `parse_value` as a plain string, and `cameras` is a *tool*
argument carried in the call's JSON, never a `-P` value. `parse_value` has no
list syntax and must not grow one for this.

## 3. Design

### 3a. `images` mode

`LLMAgentPolicy.__init__` gains `images: str = "always"`, validated against
`{"always", "on_demand"}` and recorded on `AgentPolicyConfig` so the eval log
carries it. An invalid value raises `ConfigError` with a `fix:` line, matching
the wire-gated checks at `policy.py:159-179` (the older `ValueError`s in that
constructor are the inconsistency, tracked in #168, and are not the pattern to
copy).

The default preserves today's behavior, so existing benchmark numbers stay
comparable and no CLI invocation changes meaning.

`"always"`: `_observation_content` behaves as it does today and `take_pic` is
not in `schemas()`, so nothing in §3b-§3e is reachable.

`"on_demand"`: observation messages carry the state text plus one line naming
the cameras that `take_pic` can reach. No image parts.

### 3b. `take_pic`

Exposed by `Toolset.schemas()` only in on-demand mode:

```
take_pic(cameras?: string[], note: string)
```

`cameras` omitted means every camera in the current observation. The schema
description enumerates the cameras the embodiment *declares*
(`ObservationSpace.cameras`), since that is all the toolset knows at build
time. `note` is required, matching the move tools: the note stream is the
human-readable narration the user follows live, and a capture is a decision
worth narrating ("I cannot tell whether the gripper cleared the rim, so I am
looking before descending").

Validation lives in `Toolset.execute`, which holds the observation, and
resolves names against `observation.images` — the truth, not the declaration.
A declared-but-absent camera (a dropped ROS frame) therefore returns the
unknown-name error listing the names actually present:

- unknown or absent camera name → error naming the cameras present now
- `cameras` present but not a list of strings, or empty → structured error
- `note` missing or blank → structured error, worded like the move tools'
- the observation carries no images at all → error saying so

`build_toolset` refuses `on_demand` at bind time when
`observation_space.cameras` is empty, with a message naming the fix (drop
`-P images=on_demand`). Failing at bind rather than mid-trial matches every
other unsupported configuration in this module, and an embodiment that serves
frames without declaring them is already outside the compatibility contract.

`take_pic` yields neither a chunk nor an error. `ToolResult` gains
`capture: tuple[str, ...] | None` holding the resolved camera names, and its
class docstring must stop claiming that exactly one of `chunk`/`error` is set.
The two `assert result.chunk is not None` sites (`policy.py:443`, and inside
`_forced_give_up`) are narrowed to the branches that produced a chunk, so
`mypy --strict` still follows the invariant where it holds.

The policy, not the toolset, decides whether a capture resolves now or after a
motion, and writes the tool-result text accordingly, because that decision
depends on position within the turn rather than on the observation.

### 3c. The call walk

`act()` stops executing only the first tool call. It walks the turn's calls in
order and appends **every** tool result before anything else, because both
wires require the results to sit immediately after the assistant message that
requested them: the Messages API rejects a `tool_use` whose `tool_result` is
not in the very next message, and `_translate_messages` flushes its pending
results the moment any non-tool message arrives. Slipping an image message
between two tool results is a 400, not a style question.

The walk:

1. A `take_pic` seen **before** any chunk-producing call is an *immediate*
   capture (§3d). It short-circuits the turn: every remaining call is answered
   `ignored: one tool call per turn` and no motion executes. The model asked to
   look before acting, so it re-decides with the frames in hand rather than
   executing a motion it chose blind.
2. Otherwise the first chunk-producing call (a move, `done`, or `give_up`)
   executes as it does today.
3. A `take_pic` **after** that call is *queued* (§3e) and answered
   `queued: frames arrive with the next observation, once the motion has
   finished playing`. Only one capture is queued per turn; a second is answered
   `ignored: one tool call per turn`. After `done` or `give_up` there is no next
   observation, so a queued capture there is answered
   `ignored: the trial ends with this call`.
4. Every other extra is answered `ignored: one tool call per turn`, the string
   used today.
5. A call that returns an error is answered with the error and ends the walk;
   remaining calls are answered `ignored: one tool call per turn` and the loop
   asks for a new turn, as today.

**This changes `always` mode too, in one respect: ordering.** Today every
extra's `ignored` result is appended *before* the executed call's result
(`policy.py:421-436`). The walk emits results in call order instead, which is
the correct mirror of the request and keeps a single dispatch path rather than
two divergent ones. Two existing tests pin the old order and must be updated:
`test_transcript_echo_marks_extra_tool_calls_before_executed_result` and
`test_extra_tool_calls_are_answered_but_not_executed` (`test_policy_e2e.py:417`
and `:836`). The `ignored: one tool call per turn` wording is deliberately
unchanged so only the order moves. CHANGELOG records it.

### 3d. Immediate capture

After the walk, an immediate capture appends one `user` message whose parts are
the labelled image parts, built by the same helper `_observation_content` uses
so the `camera 'top_cam' (step 480):` label stays **byte-identical**. That
label is not cosmetic: `_html.py` matches it with
`_FRAME_LABEL_RE.fullmatch` to pair a transcript label with its stored frame in
`inspect-robots view`, so any decoration inside the parentheses silently drops
the frame from the report. Narration never goes in the label; it goes in a
neighbouring text part.

The loop then asks for another LLM call. The capture spends one
`max_llm_calls` unit; budget exhaustion already forces `give_up`.

Two consecutive `user` messages (tool results, then frames) is the shape the
Anthropic wire already produces between turns today, so no wire client changes.

**Repeat guard.** Frames cannot change without stepping the robot, so within
one `act()` each camera is revealed at most once. The policy tracks the
revealed set for the current observation; a request naming any camera already
shown returns a structured error naming those cameras and saying the view
cannot change until the robot moves. No partial success: one branch, one error
string.

**Failure counter.** A successful immediate capture resets `failures` to `0`.
Without that, the counter initialised once per `act()` (`policy.py:393`) stops
meaning "consecutive" the moment a success stops returning, and the two
messages that call it consecutive (`policy.py:415`, `:441`) become wrong. With
the reset, three scattered typos across a long capture sequence no longer kill
the trial and the messages stay true.

### 3e. Queued capture and playout accounting

`_PendingCapture` records the resolved camera names, `issued_step`
(`observation.extra["env_step"]` when the motion was issued), and `chunk_len`
(the number of actions in the emitted chunk).

At the top of the next `act()` the policy consumes it: the resolved cameras'
image parts are appended **into the observation message itself**, via
`_observation_content(observation, state_labels, reveal=cameras)`, and those
cameras are entered into the new observation's revealed set so the model cannot
immediately re-request them. Keeping the frames inside the observation message
holds the message count and delta-stream shape identical to `always` mode, and
is semantically right: the frames *are* that observation's.

The leading text part of that message gains one narration line derived from
`env_step` arithmetic (`advanced = new_env_step - issued_step`):

- `advanced >= chunk_len` → `the motion finished playing (12 of 12 steps).`
- `advanced < chunk_len` → `the motion played 3 of 12 steps before this
  observation; it did not run to the end.`
- `env_step` missing or not an `int` on either observation (a direct
  `policy.act()` call outside `rollout()`) → `these frames follow the motion.`,
  with no step arithmetic, matching how `_step_label` already degrades.

**What this does and does not guarantee.** It guarantees the frames come from
the observation the rollout produced after playing the chunk's actions, so a
slow embodiment or a long interpolation delays the picture instead of taking it
early. It does **not** guarantee the arm is at the requested target, and no
wording may imply that. The rollout hands each action to the approver chain and
never tells the policy about a rewrite (`rollout.py:267-278`); the CLI wires
Clamp and DeltaLimit by default, and the plugin README already documents that a
tight `--max-action-delta` truncates absolute interpolants. `SmoothingController`
EMA-blends every action, so the final commanded value is never the
interpolant's endpoint. The report is about playout, which the policy can
observe, not arrival, which it cannot.

**Degradation under other controllers, stated plainly in the README.**
`DefaultController` buffers `list(chunk.actions)[:replan_interval]`, so with any
`replan_interval` shorter than a typical interpolation the advance is always
`replan_interval` and every chained capture reports a partial playout.
`EnsemblingController` re-queries every control step, so the advance is always
`1` (and it rebuilds actions from chunk meta, so `done`/`give_up` do not
terminate there either — an existing limitation noted in `rollout.py:259-263`).
Reporting the observed numbers rather than a binary verdict is what keeps these
cases honest.

**A trial that ends drops the queued capture.** `rollout()` breaks on
`terminated`, `truncated`, or a policy-requested stop, and the `while/else`
ends on `max_steps`; in each case there is no next `act()`. The transcript's
last tool message then reads `queued: ...` with nothing following it, which is
accurate — the trial ended first. Cross-trial leakage is prevented by `reset()`
clearing the queue and the revealed set, not by the consume path.

### 3f. System prompt and nudge

On-demand mode swaps two sentences of `_SYSTEM_TEMPLATE`:

- perception: camera images are not attached automatically; call `take_pic` to
  see them
- turn shape: exactly one motion per turn, and `take_pic` may be chained in the
  same turn — placed after a motion it returns frames once the motion has
  finished playing, placed alone it looks before deciding

The no-tool-call nudge (`policy.py:417-418`, `"Respond with exactly one tool
call."`) becomes mode-dependent for the same reason; in on-demand mode it must
not contradict the turn shape the system prompt just taught. `always` mode
keeps both strings verbatim.

### 3g. Echo and transcript

`transcript_echo` gains lines for captures (`[agent] -- captured 2 frame(s)`,
`[agent] -- queued capture: 'top_cam'`). `_sanitize` and `transcript_delta`
need no change: the new image parts are `image_url` dicts inside list content,
which `_sanitize` already replaces with omission markers, and the rollout pulls
the delta once per inference.

## 4. Files

```
plugins/inspect-robots-agent/
├── pyproject.toml         # version 0.13.0 -> 0.14.0
├── README.md              # images mode, take_pic, chaining, playout semantics,
│                          # controller degradation, the -P knob list
├── src/inspect_robots_agent/
│   ├── _tools.py          # take_pic schema + validation, ToolResult.capture and
│   │                      # its docstring, build_toolset(images=...) bind refusal
│   └── policy.py          # images knob, call walk, capture bookkeeping,
│                          # playout accounting, on-demand prompt and nudge
└── tests/
    ├── test_package.py    # pinned __version__
    ├── test_tools_motion.py   # schema exposure, argument validation, bind refusal
    └── test_policy_e2e.py     # on-demand flow, chaining, playout reporting,
                               # the two reordered extras tests
plans/0027-agent-on-demand-vision.md
CHANGELOG.md
```

`inspect_robots_agent.__all__` and the core `tests/test_api_snapshot.py` are
unchanged: `_PendingCapture` and the `take_pic` plumbing are private and no
core API moves.

## 5. Testing

Unit (`test_tools_motion.py`):

- `take_pic` absent from `schemas()` in `always` mode, present in `on_demand`
- `build_toolset(images="on_demand")` raises `ToolsetError` naming the fix when
  the observation space declares no cameras
- unknown name, declared-but-absent name, non-list `cameras`, empty `cameras`,
  blank `note`, and an imageless observation each return a structured error
  rather than raising
- omitted `cameras` resolves to every camera in the observation

End-to-end (`test_policy_e2e.py`, scripted `httpx.MockTransport` as today):

- on-demand observation messages carry no image parts and name the cameras
- an immediate `take_pic` appends its tool result, then a user message of image
  parts, and the loop continues to a second LLM call
- `take_pic` before a move short-circuits: the move is answered `ignored`, no
  chunk is returned from that turn, and every `tool_call_id` is answered exactly
  once with the results contiguous
- a repeat `take_pic` for an already-revealed camera errors, and a later
  successful capture has reset the failure counter
- a move chained with `take_pic` returns the chunk, answers both calls in order,
  and attaches the frames to the *next* observation message with a
  byte-identical `camera 'x' (step N):` label
- the narration line reads "finished playing" when `env_step` advanced by the
  full chunk length and reports the observed counts when it did not
  (`DefaultController(replan_interval=1)` drives the short case)
- a capture queued behind a chunk on a step that terminates the trial is never
  delivered and leaves no residue in the next trial
- `take_pic` chained after `done` is answered `ignored: the trial ends with this
  call`
- `always` mode still emits no `take_pic` schema, and the two reordered extras
  tests assert the new call-order results
- `images` appears in `AgentPolicyConfig`; an invalid value raises `ConfigError`
  carrying a `fix:` line

Gates: `ruff check`, `ruff format --check`, `mypy --strict` over
`plugins/inspect-robots-agent/src/inspect_robots_agent`, and the plugin test
suite. Core coverage is untouched; plugin coverage stays report-only.

## 6. Risks

**A blind model.** In on-demand mode a model that never calls `take_pic` drives
on proprioception alone and will score worse. This is the point of the knob and
the reason `always` stays the default; the on-demand system prompt says plainly
that images exist and how to get them.

**Capture spam.** A model could burn its budget looking. The per-observation
repeat guard removes the only useless case (the same camera twice with no
motion between), and `max_llm_calls` bounds the rest.

**Chaining is a latency win, not a correctness win.** It saves one LLM
round-trip per look-after-move; it does not make the motion more accurate, and
under `replan_interval` or ensembling it degrades to reporting a partial
playout. The README says so rather than leaving users to infer it from a
surprising transcript.
