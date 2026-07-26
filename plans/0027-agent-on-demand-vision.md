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
- Playout accounting, plus a **measured** residual against the requested target
  so a motion the approvers truncated is reported as short rather than done.

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
not in `schemas()`.

`"on_demand"`: observation messages carry the state text plus one line naming
the cameras that `take_pic` can reach. No image parts. That line names the
cameras present in `observation.images`, **not** the declared ones, and when
the observation carries no frames it says so. Advertising a declared camera
that did not arrive would invite a call that can only return the "no images"
error, and three of those inside one `act()` would kill a trial over a dropped
frame — the condition §3e goes out of its way to survive on the delivery path.

Every capture branch in the walk is gated on `images == "on_demand"`, not on
the schema. Tool names come from the model, so a stray `take_pic` in `always`
mode must still fall through to the existing structured unknown-tool error
(`_tools.py:202-205`). That error's `available:` list becomes mode-dependent so
it names `take_pic` in on-demand mode and omits it in `always` mode — a stale
list is worst precisely where the model is guessing tool names.

### 3b. `take_pic`

Exposed by `Toolset.schemas()` only in on-demand mode:

```
take_pic(cameras?: string[], note: string)
```

The schema description enumerates the cameras the embodiment *declares*
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

An omitted `cameras` is resolved by the **policy**, not the toolset. On the
immediate path it resolves against the current observation's revealed set
(§3d). On the queued path it is stored verbatim — `None` for the omitted form,
the named tuple otherwise — with **no subtraction at queue time**, and is
resolved at delivery against the arriving observation, whose revealed set is
empty (§3e). Subtracting at queue time would reject exactly the flow this
feature exists for: look at observation *N*, then chain a capture onto a motion
in the same observation, whose frames belong to *N+1*.

`build_toolset` refuses `on_demand` at bind time when
`observation_space.cameras` is empty, with a message naming the fix (drop
`-P images=on_demand`). Failing at bind rather than mid-trial matches every
other unsupported configuration in this module, and an embodiment that serves
frames without declaring them is already outside the compatibility contract.

A *successful* `take_pic` yields neither a chunk nor an error; the validation
failures above still come back as `ToolResult.error` like any other tool's.
`ToolResult` becomes
`@dataclass(frozen=True, eq=False)` and gains two fields:

- `capture: tuple[str, ...] | None` — resolved camera names, or `None` when
  `cameras` was omitted, which the policy then resolves itself
- `target: npt.NDArray[np.float64] | None` — the clipped absolute target vector
  a successful `_move_absolute` computed (§3e). `None` for displacement modes,
  stops, and captures.

`eq=False` is not optional: the default `eq=True` on a frozen dataclass
generates `__eq__` and `__hash__` over the field tuple, and a NumPy field makes
both raise. `types.py:7-9` documents the same convention for the core's
array-carrying dataclasses.

`ToolResult`'s class docstring must stop claiming that exactly one of
`chunk`/`error` is set. The two `assert result.chunk is not None` sites
(`policy.py:443`, and inside `_forced_give_up`) are narrowed to the branches
that produced a chunk, so `mypy --strict` still follows the invariant where it
holds.

`Toolset` also gains `residual(target, observation) -> tuple[str, float] | None`
(with a `D102` docstring) returning the dimension label and magnitude of the
largest absolute difference between `target` and the observation's
proprioceptive state. It lives on the toolset because the toolset owns
`_state_key` and `_labels`.

It must never raise, because it runs on the delivery path where §3e forbids
killing a trial over a degraded observation. It returns `None` when any of
these hold: `self._state_key is None`; the key is absent from
`observation.state`; the value does not coerce to a float array; its shape does
not match `len(self._labels)`; or any element of `target - state` is
non-finite. `_current_state` is not reusable here — it indexes
`observation.state[...]` directly and would raise `KeyError` on a dropped
field, which `rollout.py:225` turns into a `PolicyError`. A NaN state would
otherwise print `nan on <first label>` as if it were a measurement.

### 3c. The call walk

`act()` stops executing only the first tool call. It walks the turn's calls in
order and appends **every** tool result before anything else, because both
wires require the results to sit immediately after the assistant message that
requested them: the Messages API rejects a `tool_use` whose `tool_result` is
not in the very next message, and `_translate_messages` flushes its pending
results the moment any non-tool message arrives. Slipping an image message
between two tool results is a 400, not a style question.

The walk collects results for every call and only then decides what `act()`
does. It is a two-state machine, not a list of cases. The state is *open* until
some call produces a chunk or an immediate capture, and *closed* after.

Every call reaches `toolset.execute` **while the walk is open**. Once closed,
the only call still dispatched is an on-demand `take_pic` behind a non-stop
motion chunk, which needs argument and camera-name validation to queue.
Anything answered `ignored` is never dispatched — `_move` raises on a
non-finite proprioceptive reference (`_tools.py:232`) and both `_move` and
`_stop` index `observation.state[...]` directly (`_tools.py:211`), so
dispatching an extra on a degraded observation would turn today's correctable
error loop into a fatal `PolicyError` (`rollout.py:225`).

**While open**, each call is dispatched normally:

- an on-demand `take_pic` closes the walk **whatever its outcome**. On success
  it is an *immediate* capture (§3d): the model asked to look before acting, so
  it re-decides with the frames in hand rather than executing a motion it chose
  blind. On a `toolset.execute` error it is answered with the error text and
  increments `failures`. On a revealed-set rejection it is answered with the
  rejection text and leaves `failures` alone (§3d). All three close, so a move
  behind a rejected `take_pic` is answered `ignored`, not executed.
- every other call — including an unknown tool name, and including `take_pic`
  in `always` mode — goes to `toolset.execute` exactly as today. A chunk closes
  the walk. An error is answered with its text, increments `failures`, and
  closes the walk with no chunk.

**Once closed**, nothing further reaches the embodiment and no further chunk is
produced. The one remaining live path is queuing, and it exists only when the
walk was closed by a **non-stop motion chunk** — that is the only close that
guarantees a next observation and a `chunk_len` to count against:

- on-demand `take_pic`, walk closed by a motion chunk: still passed to
  `toolset.execute` for argument and camera-name validation. On success it is
  *queued* (§3e) and answered `queued: frames arrive with the next observation,
  once the motion has finished playing`. On error it is answered with the error
  text, nothing is queued, the queue slot stays free for a later valid
  `take_pic` in the same turn, and `failures` is untouched — `act()` is
  returning the chunk regardless, so counting it would leak into the next
  observation's budget. Only one capture is queued per turn; once the slot is
  filled a second `take_pic` is answered `ignored: one tool call per turn`.
- on-demand `take_pic`, walk closed by a **stop** chunk: answered `ignored: the
  trial ends with this call`. There is no next observation to deliver into. The
  policy recognises a stop by reading `chunk.actions[0].meta.get("request_stop")`,
  which `_stop` already sets (`_tools.py:216-219`), rather than forking the
  `("done", "give_up")` literal out of `_tools.py:200`.
- on-demand `take_pic`, walk closed **without a chunk** (an immediate capture, a
  revealed-set rejection, or an error): answered `ignored: one tool call per
  turn`. No motion was emitted, so "once the motion has finished playing" would
  be false and `chunk_len` would be undefined. `[take_pic, take_pic]` resolves
  here.
- everything else is answered `ignored: one tool call per turn`, the string
  used today.

Nothing after a chunk can cancel it. An error there comes from a call that
never ran, and discarding the chunk would leave the move's own tool result
(`executing move_joints over 12 steps`) in the transcript describing a motion
that never reached the embodiment.

**After the walk**, `act()` returns the chunk if one was produced. Otherwise it
continues the `while True` loop for another LLM turn — the same path a
no-tool-call turn already takes. That covers an immediate capture, a
revealed-set rejection, an errored call, and a turn of nothing but `ignored`.
Only an error increments `failures`; an immediate capture resets it to `0`; a
revealed-set rejection leaves it untouched (§3d). `max_llm_calls` bounds every
one of these paths, since each loop iteration calls `complete()` and increments
`_calls_used`.

**This changes `always` mode too, in one respect: ordering.** Today every
extra's `ignored` result is appended *before* the executed call's result
(`policy.py:421-436`). The walk emits results in call order instead, which is
the correct mirror of the request and keeps a single dispatch path rather than
two divergent ones. Two existing tests pin the old order and must be updated:
`test_transcript_echo_marks_extra_tool_calls_before_executed_result` and
`test_extra_tool_calls_are_answered_but_not_executed` (`test_policy_e2e.py:417`
and `:832`). The `ignored: one tool call per turn` wording is deliberately
unchanged so only the order moves. CHANGELOG records it.

### 3d. Immediate capture

After the walk, an immediate capture appends one `user` message whose parts are
the labelled image parts, built by the same helper `_observation_content` uses
so the `camera 'top_cam' (step 480):` label stays **byte-identical**. That
label is not cosmetic: `_html.py` matches it with `_FRAME_LABEL_RE.fullmatch`
to pair a transcript label with its stored frame in `inspect-robots view`, so
any decoration inside the parentheses silently drops the frame from the report.
Narration never goes in the label; it goes in a neighbouring text part.

The loop then asks for another LLM call. The capture spends one
`max_llm_calls` unit; budget exhaustion already forces `give_up`.

Two consecutive `user` messages (tool results, then frames) is the shape the
Anthropic wire already produces between turns today, so no wire client changes.

**Revealed-set resolution (immediate path only).** Frames cannot change without
stepping the robot, so each camera is revealed at most once per observation.
The policy keeps a revealed set **scoped to the current observation**, cleared
every time `act()` appends a new observation message — not just in `__init__`
and `reset()`, or every camera would be one-shot for the whole trial. It
resolves the request against that set:

- `cameras` omitted → the observation's cameras minus the revealed set
- `cameras` named → the named cameras minus the revealed set

If the remainder is non-empty, those frames are revealed and the tool result is
`captured 1 frame(s): 'wrist' (already shown: 'top')`, with the parenthetical
omitted when nothing was skipped. If the remainder is empty, the result is
`already shown for this observation: 'top'; the view cannot change until the
robot moves` — and **that rejection does not increment `failures`**. It is a
refusal of a well-formed call, not a malformed call, and counting it would let
three bare `take_pic()` calls error a trial that is otherwise healthy. The
on-demand system prompt states the rule so the model can avoid it in the first
place.

This subtraction applies only here. The queued path (§3e) stores the request
unresolved.

**Failure counter.** A successful immediate capture resets `failures` to `0`.
Without that, the counter initialised once per `act()` (`policy.py:393`) stops
meaning "consecutive" the moment a success stops returning, and the two
messages that call it consecutive (`policy.py:415`, `:441`) become wrong.

### 3e. Queued capture, playout, and measured residual

`_PendingCapture` records the requested camera names (or `None` for "all"),
`issued_step` (`observation.extra["env_step"]` when the motion was issued),
`chunk_len` (the number of actions in the emitted chunk), and `target` (the
`ToolResult.target` vector, `None` outside absolute modes).

At the top of the next `act()` the policy consumes it. Requested names are
**intersected with the new observation's `images`**: present cameras are
revealed, absent ones are named in the narration, and nothing raises — a
dropped frame between the request and the arrival must not kill the trial via
`PolicyError`. The revealed cameras are entered into the new observation's
revealed set so the model cannot immediately re-request them.

The image parts go **into the observation message itself**, via
`_observation_content(observation, state_labels, reveal=cameras)`. `reveal` is
keyword-only and defaults to `None` meaning "every camera", so the existing
call sites that pass one or two positional arguments
(`test_policy_e2e.py:321-370`) keep rendering every frame; on-demand mode
passes an explicit empty tuple for an unrevealed observation. That holds
the message count and delta-stream shape identical to `always` mode, and is
semantically right: the frames *are* that observation's.

The leading text part gains one narration line. With
`advanced = new_env_step - issued_step`:

- both steps are `int` and `advanced > 0`:
  - `advanced >= chunk_len` → `the motion finished playing (12 of 12 steps).`
  - otherwise → `the motion played 3 of 12 steps before this observation; it
    did not run to the end.`
- `env_step` missing, not an `int`, or `advanced <= 0` → `these frames follow
  the motion.` with no arithmetic. The `<= 0` case is the direct-`policy.act()`
  pattern the existing suite uses (`test_policy_e2e.py:436`), where every call
  passes `env_step=0`.

When `target` is set **and** `Toolset.residual` returns a value, a second
sentence follows: `largest remaining offset from the requested target is 0.004
on j3.` The magnitude is formatted `:.4g`, matching `bounds_text`
(`_tools.py:493-497`). When residual returns `None` the sentence is omitted
entirely and the playout line stands alone. This is
what actually answers the user's requirement, and it is a measurement rather
than a promise. The policy cannot make the rollout wait for arrival — the
rollout owns the loop — but it can tell the model how far off the arm ended up.
That matters precisely where step counting is blind: `rollout.py:267-278` hands
every action to the approver chain and never reports a rewrite, the CLI wires
Clamp and DeltaLimit by default, a tight `--max-action-delta` truncates
absolute interpolants, and `SmoothingController` EMA-blends every action so the
commanded value is never the interpolant's endpoint. In all of those a step
count reads as success while the residual reads as short.

Displacement modes have no state-space target, so they keep the playout-only
line.

**Degradation under other controllers, stated plainly in the README.**
`DefaultController` buffers `list(chunk.actions)[:replan_interval]`, so the
advance is `min(replan_interval, chunk_len)`: a `replan_interval` shorter than
the interpolation reports a partial playout every time, while a chunk shorter
than the interval still reports finished. `EnsemblingController` re-queries
every control step, so the advance is always `1` (and it rebuilds actions from
chunk meta, so `done`/`give_up` do not terminate there either — an existing
limitation noted in `rollout.py:259-263`).

**A trial that ends drops the queued capture.** `rollout()` breaks on
`terminated`, `truncated`, or a policy-requested stop, and the `while/else`
ends on `max_steps`; in each case there is no next `act()`. The transcript's
last tool message then reads `queued: ...` with nothing following it, which is
accurate — the trial ended first. Cross-trial leakage is prevented by `reset()`
clearing the queue and the revealed set, not by the consume path.

### 3f. System prompt and nudge

On-demand mode swaps two sentences of `_SYSTEM_TEMPLATE`:

- perception: camera images are not attached automatically; call `take_pic` to
  see them, and a camera already shown for the current observation cannot be
  re-taken until the robot moves
- turn shape: exactly one motion per turn, and `take_pic` may be chained in the
  same turn — placed after a motion it returns frames once the motion has
  finished playing, placed alone it looks before deciding

The no-tool-call nudge (`policy.py:417-418`, `"Respond with exactly one tool
call."`) becomes mode-dependent for the same reason; in on-demand mode it must
not contradict the turn shape the system prompt just taught. `always` mode
keeps both strings verbatim.

### 3g. Echo, state, and transcript

`transcript_echo` gains lines for captures (`[agent] -- captured 2 frame(s)`,
`[agent] -- queued capture: 'top_cam'`). The observation echo summary
(`policy.py:384`, `f"{len(observation.images)} camera(s)"`) becomes
`"1 camera(s) available"` in on-demand mode so it stops implying frames were
sent.

`_pending` and the revealed set are initialised in `__init__` next to
`self._calls_used = 0` as well as in `reset()`: `act()` is reachable without
`reset()` (`policy.py:373` guards only `bind()`), and `mypy --strict` requires
the attributes to exist. The revealed set is additionally cleared on every new
observation message (§3d); `_pending` is cleared when consumed.

`_sanitize` and `transcript_delta` need no change: the new image parts are
`image_url` dicts inside list content, which `_sanitize` already replaces with
omission markers, and the rollout pulls the delta once per inference.

## 4. Files

```
plugins/inspect-robots-agent/
├── pyproject.toml         # version 0.13.0 -> 0.14.0
├── README.md              # images mode, take_pic, chaining, playout + residual
│                          # semantics, controller degradation, the -P knob list
├── src/inspect_robots_agent/
│   ├── _tools.py          # take_pic schema + validation, ToolResult.capture and
│   │                      # .target and its docstring, Toolset.residual,
│   │                      # build_toolset(images=...) bind refusal
│   └── policy.py          # images knob, call walk, revealed-set resolution,
│                          # _PendingCapture, playout + residual narration,
│                          # on-demand prompt, nudge, and echo
└── tests/
    ├── test_package.py    # pinned __version__
    ├── test_tools_motion.py   # schema exposure, validation, bind refusal, residual
    ├── test_policy_e2e.py     # on-demand flow, chaining, narration,
    │                          # the two reordered extras tests
    ├── test_anthropic.py      # translated Messages body for a capture history
    └── test_responses.py      # item order for a capture history
plans/0027-agent-on-demand-vision.md
CHANGELOG.md
```

`inspect_robots_agent.__all__` and the core `tests/test_api_snapshot.py` are
unchanged: `_PendingCapture` and the `take_pic` plumbing are private and no
core API moves. `__version__` reads `importlib.metadata`, so bumping the plugin
pyproject alone is enough.

## 5. Testing

Unit (`test_tools_motion.py`):

- `take_pic` absent from `schemas()` in `always` mode, present in `on_demand`;
  a `take_pic` call in `always` mode returns the unknown-tool error
- `build_toolset(images="on_demand")` raises `ToolsetError` naming the fix when
  the observation space declares no cameras
- unknown name, declared-but-absent name, non-list `cameras`, empty `cameras`,
  blank `note`, and an imageless observation each return a structured error
  rather than raising
- a successful `_move_absolute` carries the clipped `target`; `residual` returns
  the labelled largest offset, and `None` when there is no matching state field

Multi-camera cases need a two-camera fixture: `CubePickEmbodiment` declares one
camera (`mock/cubepick.py:57`), so these build a custom `EmbodimentInfo` and
`Observation` the way `test_policy_e2e.py` already does elsewhere.

**The e2e suite needs a new embodiment fixture, and neither existing one
works.** `CubePickEmbodiment` is `eef_delta_pos`, so `state_key` stays `None`
and `residual` can never fire. `_AbsoluteEmbodiment` (`test_policy_e2e.py:185`)
is `joint_pos` but declares no `CameraSpec` and emits no images, so
`build_toolset` refuses on-demand at bind. Add one modelled on
`_AbsoluteEmbodiment`: `joint_pos` semantics, two `CameraSpec`s, and matching
`images` on `reset()`/`step()`. Every on-demand e2e test below runs against it.

End-to-end (`test_policy_e2e.py`, scripted `httpx.MockTransport` as today):

- on-demand observation messages carry no image parts and name the cameras
- an immediate `take_pic` appends its tool result, then a user message of image
  parts, and the loop continues to a second LLM call
- `take_pic` before a move short-circuits: the move is answered `ignored`, no
  chunk is returned from that turn, and every `tool_call_id` is answered exactly
  once with the results contiguous
- an erroring `take_pic` *after* a move leaves the chunk intact and returns it,
  queues nothing, and leaves the slot free: `[move, take_pic(invalid),
  take_pic(valid)]` still queues the third call's capture
- `[take_pic, take_pic]` captures once; the second is answered `ignored: one
  tool call per turn` and no second image message is appended
- the unknown-tool error's `available:` list names `take_pic` in on-demand mode
- a bare repeat `take_pic()` is rejected without incrementing the failure
  counter: three in a row do not error the trial
- with two cameras, a `take_pic` naming both after one was already shown reveals
  only the unshown one and says so
- a move chained with `take_pic` returns the chunk, answers both calls in order,
  and attaches frames to the *next* observation message with a byte-identical
  `camera 'x' (step N):` label
- the narration reads "finished playing" at full advance, reports observed
  counts under `DefaultController(replan_interval=1)`, degrades to the neutral
  sentence when `advanced <= 0`, and appends the residual sentence in an
  absolute mode
- a camera that disappears between request and arrival is named as missing and
  does not raise
- a capture queued behind a chunk on a step that terminates the trial is never
  delivered and leaves no residue in the next trial
- `take_pic` chained after `done` is answered `ignored: the trial ends with this
  call`
- the two reordered extras tests assert the new call-order results
- `images` appears in `AgentPolicyConfig`; an invalid value raises `ConfigError`
  carrying a `fix:` line

Wire tests (`test_anthropic.py`, `test_responses.py`): the e2e suite runs the
chat wire, which passes `self._messages` through verbatim (`_llm.py:199`) and
so proves nothing about the adjacency constraint §3c is built around. Add one
test per wire feeding a capture history (assistant with two tool calls, both
results, then the image user message) and asserting the translated body keeps
the `tool_result` blocks contiguous in the message immediately following the
assistant turn, and one `function_call_output` per `function_call` in order.

Gates: `ruff check`, `ruff format --check`, `mypy --strict` over
`plugins/inspect-robots-agent/src/inspect_robots_agent`, and the plugin test
suite. Core coverage is untouched; plugin coverage stays report-only.

## 6. Risks

**A blind model.** In on-demand mode a model that never calls `take_pic` drives
on proprioception alone and will score worse. This is the point of the knob and
the reason `always` stays the default; the on-demand system prompt says plainly
that images exist and how to get them.

**Capture spam.** A model could burn its budget looking. The revealed-set
resolution removes the only useless case (the same camera twice with no motion
between) without erroring the trial, and `max_llm_calls` bounds the rest.

**Chaining is a latency win, not a correctness win.** It saves one LLM
round-trip per look-after-move; it does not make the motion more accurate, and
under `replan_interval` or ensembling it degrades to reporting a partial
playout. The residual sentence is what keeps the transcript honest in those
cases, and the README says so rather than leaving users to infer it from a
surprising transcript.
