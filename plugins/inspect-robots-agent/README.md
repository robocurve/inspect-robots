# inspect-robots-agent

LLM agent policy for [Inspect Robots](https://github.com/robocurve/inspect-robots):
frontier LLMs (Claude, GPT, anything behind an OpenAI-compatible API) drive any
registered embodiment through tool calls, as a first-class `Policy` named
`agent`. The same policy runs ad-hoc instructions and scores on registered
tasks next to fine-tuned VLAs.

## Install

```bash
pip install inspect-robots inspect-robots-agent
```

## Quickstart (no hardware)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-fable-5 -P effort=low --embodiment cubepick
```

Model strings are OpenRouter-style `provider/model`, resolved from
`-P model=...` or `$INSPECT_ROBOTS_MODEL`. API keys come from the environment:

1. `-P base_url=...` (with `-P api_key_env=NAME`): any supported compatible endpoint
2. A known provider prefix with that provider's key set: the provider's own
   endpoint, with the prefix stripped unless that endpoint requires the full id
3. `OPENROUTER_API_KEY`: OpenRouter, any model string. Ids ending in a known
   OpenRouter variant suffix (`:free`, `:nitro`, `:floor`, `:extended`,
   `:online`, `:thinking`) always route here, since the variant means nothing
   to a provider's own API; other colons (fine-tune ids like `openai/ft:...`)
   still resolve directly.

Providers resolved directly by prefix:

| Prefix | Key | Endpoint |
|---|---|---|
| `anthropic/*` | `ANTHROPIC_API_KEY` | Anthropic (OpenAI-compat, or native with `-P wire=messages`) |
| `openai/*` | `OPENAI_API_KEY` | OpenAI |
| `google/*` | `GEMINI_API_KEY` | Google Gemini (OpenAI-compat) |
| `x-ai/*` or `xai/*` | `XAI_API_KEY` | xAI |
| `groq/*` | `GROQ_API_KEY` | Groq (rest of the id passed through, slashes and all) |
| `mistralai/*` | `MISTRAL_API_KEY` | Mistral |
| `deepseek/*` | `DEEPSEEK_API_KEY` | DeepSeek |
| `thinkingmachines/*` | `TINKER_API_KEY` | Tinker Messages API (full model id passed through) |

### Gemini Robotics ER 2

Google serves two Gemini Robotics ER 2 model ids. Use
`google/gemini-robotics-er-2-preview` on the default `chat` wire. The
latency-oriented `google/gemini-robotics-er-2-streaming-preview` id requires
the stateful Live API wire:

```bash
inspect-robots "pick up the cube" --policy agent \
    -P model=google/gemini-robotics-er-2-streaming-preview \
    -P wire=gemini-live --embodiment cubepick
```

The Live wire does not accept `effort` and does not use `image_horizon`.
Leave both unset. Google's own Live context-window compression is the
equivalent history mechanism because frames already streamed into a session
cannot be evicted by the client. Live usage counts include the empty resumed
generation and the observation-triggered generation on a normal step, so
input token totals cover two generations per step.

The wire format defaults to Chat Completions for broad OpenAI-compatible
endpoint support:

| `-P wire=` | Endpoint | Use it when |
|---|---|---|
| `chat` (default) | `/chat/completions` | Anything OpenAI-compatible: OpenRouter, vLLM, Ollama, the Anthropic and Gemini compat endpoints |
| `responses` | `/responses` | A direct OpenAI or compatible endpoint requires the Responses API |
| `messages` (`anthropic` alias) | `/messages` | Anthropic, Tinker, or a compatible Messages endpoint |
| `gemini-live` | `BidiGenerateContent` (WSS) | Google's Live API: required for the `-streaming-` robotics model ids |

## How it works

Motion tool calls state where to go, not how long to move. For absolute modes,
the move tool (`move_joints` for joint spaces, `move_to` for Cartesian pose
modes) interpolates named partial targets from the observed state at a fixed
safe speed. The default `max_speed_frac=0.1` allows a tenth of each
dimension's range per second, subject to a 5%-of-range per-step ceiling that
matches the core's default delta backstop. At that default a near-full-range
move exceeds the 10 s per-call playout cap, so the agent receives a
split-the-move error and issues it as two smaller motions; raise the fraction
(up to `0.5` before the ceiling binds at 10 Hz) for faster arms. The tool
result reports the computed step count and, when the embodiment declares
`control_hz`, the corresponding playout time. `duration_s` is not part of either motion tool.

Every move tool call also requires a `note` with one or two plain sentences
describing the current observation and why the agent chose that motion. The
user reads these notes live and in the saved transcript to follow what the
agent sees and decides.

Camera images are attached to every observation by default
(`-P images=always`). Set `-P images=on_demand` to send state without image
payloads and give the model a `take_pic` tool instead:

```bash
inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-fable-5 -P images=on_demand \
    --embodiment cubepick
```

`take_pic` requires a human-readable `note` and accepts an optional `cameras`
list. Omitting the list requests every camera available in that observation.
A camera can be revealed only once per observation because its view cannot
change until the robot moves. If a runtime camera dropout leaves the
observation with no images, the well-formed call is rejected with `no camera
images are available in this observation` instead of counting as a malformed
tool call. The first such world-state rejection in one policy decision is
free; repeated rejections escalate to the normal three-strike guard.

A standalone `take_pic` shows the current frame and lets the model decide
again before moving. A `take_pic` placed after one motion in the same assistant
turn is queued with `queued: frames arrive with the next observation, after
the motion plays`: the motion chunk is returned immediately, and the requested
frames are attached to the next observation after the controller has played
the available part of the chunk. Two motions still cannot be chained.

Queued-capture narration reports what the rollout actually observed. It says
whether all requested chunk steps played or only a prefix did, names camera
frames missing on arrival, and, for absolute control modes with matching
proprioception, reports the largest measured offset from the requested target.
The residual is the arrival check: a full step count alone does not prove the
arm reached its target when an approver rewrote actions or a smoothing
controller blended them.

Controller choice affects that report. `DefaultController` buffers
`min(replan_interval, chunk_len)` actions, so a `replan_interval` shorter than
the interpolation always reports partial playout; a chunk shorter than the
interval reports finished. `EnsemblingController` re-queries every control
step, so the observed advance is always one step. It also rebuilds actions
from chunk metadata, which means `done` and `give_up` do not terminate under
ensembling (an existing core limitation). A trial that terminates or reaches
its step limit before the next policy call drops any queued capture.

For displacement modes, `move_by` splits the requested total so every action
fits the box side in that direction. The action box is the embodiment author's
per-step speed statement, so `max_speed_frac` does not apply to displacement
modes. `done` and `give_up` end the trial through the core's policy-stop
channel. Both tools ask for a required `hindsight` argument: what the agent
knows now that it wishes it had known at the start of the episode, as
concrete transferable rig and task facts. The system prompt announces the
question up front so the model tracks learnings during the rollout. The
answer persists twice deliberately (the transcript naturally carries the tool
call as well): as `stop_hindsight` in the stop action's meta, and as
`trial_metadata["hindsight"]` in the JSON log next to `llm_usage`. Missing
hindsight never fails execution (the budget-exhausted forced `give_up`
cannot answer). Harvested hindsight is written to be usable as
`prior_learnings` input on later runs, which closes the relearning loop.

When `control_hz` is `None`, the plugin uses a 10 Hz fallback to compute step
counts and the per-call playout cap, but leaves the emitted chunk rate unset.
The embodiment then plays the chunk at its native rate. In this case the speed
and playout caps are step-count constructs, not wall-clock guarantees, and the
tool result does not report seconds.

When the embodiment publishes operating notes via `EmbodimentInfo.docs`
(joint layout, sign conventions, gripper polarity), the policy appends them
to the system prompt as an `Embodiment notes:` section. The per-step
observation also labels the proprioceptive state vector with the action
dimension names (`left_j0=0.01 ...`) whenever the mapping is unambiguous.

Every action still passes the CLI's default safety approvers (bounds clamp plus
per-step delta limit); the plugin contains no safety-critical code path of its
own. An explicit `--max-action-delta` tighter than 5% of range can truncate
absolute interpolants. In displacement modes, a value tighter than the action
box can truncate each `move_by` step. Either setting can make the executed
motion fall short of the tool's requested total.

### Operator feedback

`LLMAgentPolicy` opts into the framework's live operator channel on attended
runs. Feedback typed during a trial is included in the next observation sent
to the model, labeled with the environment step when it was received. The
model treats these lines as trusted guidance from the human supervising the
robot. The framework also saves the feedback in the eval log.

## Motion pre-check

Python callers can pass `pre_check=` to `LLMAgentPolicy` to inspect one
absolute motion before its chunk is emitted. The callable receives a
read-only float64 array with shape `(steps, dim)`. It contains the exact
already-clipped action waypoints for one move call. The first row is the first
commanded waypoint and the last row is the final target. Return `None` to
allow the motion. Return a nonempty human-readable string to reject it. The
agent receives `pre-check rejected this motion: <reason>` and can choose a
different target on the next turn.

Here is an adapter for the collision checker from
[`inspect-robots-yam`](https://github.com/robocurve/inspect-robots-yam):

```python
import numpy as np
import numpy.typing as npt

from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_yam.collision import CollisionChecker


def make_yam_collision_pre_check(checker: CollisionChecker):
    """Reject the first emitted YAM waypoint whose geometry penetrates."""

    def check_yam_waypoints(
        waypoints: npt.NDArray[np.float64],
    ) -> str | None:
        for index, waypoint in enumerate(waypoints):
            report = checker.check(waypoint)
            if report.collided:
                return f"{report.geom1}:{report.geom2} at waypoint {index}"
        return None

    return check_yam_waypoints


checker = CollisionChecker()
policy = LLMAgentPolicy(pre_check=make_yam_collision_pre_check(checker))
```

This hook is programmatic-only. `-P` CLI flags carry serialized values and
cannot carry callables. Displacement control modes are refused at bind time
when a pre-check is configured because their emitted vectors are per-step
deltas, not absolute configurations.

**Layering:** The pre-check supplies model feedback. The framework approver
chain remains the enforcement backstop. Passing the pre-check does not imply
that an approver will pass the motion. In particular, the YAM collision
approver sweeps interpolated substeps finer than the emitted waypoint spacing
at low control rates. An adapter that needs parity should interpolate and
check between emitted waypoints itself.

**Exceptions:** Exceptions from the callable propagate. The rollout converts
a generic exception into `PolicyError`, so the trial fails and
`fail_on_error` applies. A typed `SafetyAbort` keeps its own meaning and halts
the eval. A crashing or hard-vetoing adapter must fail visibly. Silently
allowing the motion is never acceptable.

**Retry budget:** A rejection is a normal tool error. Three rejected moves in
a row raise a policy error instead of reaching `give_up`. Verify rig
measurements such as `table_height` and base offsets so an over-conservative
checker does not consume the budget.

**Recorded identity:** Eval configuration records only the adapter code
identity as `module.qualname`. Two runs using the same adapter with differently
configured checkers record the same string. When checker configuration must
be distinguishable, encode it in a named factory's qualname, for example
`make_lab_a_table_742mm_pre_check`.

> [!WARNING]
> Guardrails are on by default at the CLI. **Never pass `--disable-guardrails`
> on real hardware** unless you fully trust the policy and the rig.

Configuration knobs (all `-P key=value`): `model`, `base_url`, `api_key_env`,
`wire`, `speed`, `max_output_tokens`, `max_llm_calls` (default `100`),
`temperature`, `effort`, `max_speed_frac`, `transcript_echo`, `images`
(default `always`; use `on_demand` for model-requested frames),
`image_horizon`, `depth` (default `render`; use `off` to omit depth
renders), and `prior_learnings`.
`speed` and `max_output_tokens` apply to `-P wire=messages` only, and passing
either on another wire is an error. `speed=fast` is meaningful only for Claude
on Anthropic's API; Tinker accepts and silently ignores it.

| Image option | Default | Behavior |
|---|---|---|
| `-P images=` | `always` | Attach every observation's frames; use `on_demand` for model-requested frames |
| `-P image_horizon=` | `2` on HTTP wires | Keep frames from the newest two image-bearing messages in each outgoing request; unset on `gemini-live` |

On the HTTP wires, set `-P image_horizon=none` to send the full image history.
Do not use a bare `-P image_horizon=`: the CLI parses it as an empty string,
which the policy rejects. Full history grows request bodies by about 420 KB
per observation with three cameras and can reach a 413 response around 85
observations. The HTTP default replaces older outgoing camera parts with
deterministic text stubs; the saved conversation, transcript, and separately
stored frames remain complete and unchanged.

Set `-P prior_learnings=path/to/learnings.md` to append a nonempty UTF-8 notes
file to the system prompt after any embodiment notes. The file is read once
when the policy is constructed, and its resolved path and content hash are
recorded in the eval configuration. The `hindsight` answers that `done` and
`give_up` collect into `trial_metadata` are the natural source material for
this file: harvest them across runs, distill, and feed them back here.
Set `-P transcript_echo=true` to print live `[agent]` conversation lines to
stderr, including goals, observation summaries, assistant output, tool calls,
and tool results.
Move notes appear inside the echoed tool-call arguments.
The speed fraction defaults to `0.1` and applies only to absolute modes.

`LLMAgentPolicy.transcript()` returns the current conversation as a deep copy with streamed camera frames replaced by omission markers, ready for core eval-log persistence.
Camera labels such as `camera 'top_cam' (step 480):` provide the join key from a transcript observation to its stored frame.
Live Rerun transcript streaming happens automatically when a Rerun sink is attached.

Wire capture is on by default (`-P wire_capture=false` to disable): every
request attempt each wire client sends (tool schemas, evicted view, depth
composites, and cache breakpoints) and every response land in
`wire/<run_id>/<trial_id>/calls.jsonl` under the log directory, with image
payloads deduplicated as `$blob:<sha256>` references into
`wire/<run_id>/blobs/`. The format contract lives in the
`inspect_robots_agent._capture` module docstring; browse captures with
`inspect-robots view` (Wire section) or `inspect-robots inspect --wire`.
Requires a core with the `on_trial_start` policy hook; on older cores the
policy prints one notice and captures nothing.
At trial end, `record.metadata["llm_usage"]` records `llm_calls` and the summed
integer token counters returned by the wire. The Messages wire
includes input, output, cache-creation, and cache-read tokens; other wires
currently record `llm_calls` only. Trials with no LLM calls omit the key.

Like `temperature`, reasoning effort is omitted when `-P effort=` is unset, so
the provider's own default applies. Explicit named levels (`minimal`, `low`,
`medium`, `high`, `xhigh`, and `max`) pass through unchanged. A bare
`-P effort=none` now requests the true minimum on every HTTP wire:

| Wire | Request field |
| --- | --- |
| `chat` | `reasoning_effort: "none"` |
| `responses` | `reasoning: {"effort": "none"}` |
| `messages` | `thinking: {"type": "disabled"}` (no `output_config`) |

The older quoted spelling, `-P effort="'none'"`, remains valid but is no longer
needed. In Python, both `effort=None` and `effort="none"` request the `none`
level; omit the argument to inherit the provider default. Gemini Live has no
effort field and rejects any explicit effort, so leave it unset on that wire.
To pin the behavior from before version 0.23, add `-P effort=low`.

## Depth rendering

For each camera, the policy looks for metric depth in
`observation.extra[f"{cam}_depth"]`. When present, it renders the depth as a
grayscale image immediately after that camera's RGB image: near is bright,
far is dim, and invalid pixels are black. Depth follows RGB in both
`images=always` observations and `take_pic` reveals under
`images=on_demand`.

Each render is preceded by a metric label:

```text
depth 'left_cam' (step 3): bright 0.09 m -> dim 1.41 m (2nd-98th pctl), 87% valid, center 0.31 m:
```

The bright and dim distances anchor the grayscale window at the 2nd and 98th
percentiles of valid depth. The valid percentage is an integer, and the
center depth appears only when the center pixel is valid. As with RGB camera
labels, the `(step N)` suffix is present only when the observation carries an
integer environment step; otherwise the label starts
`depth 'left_cam': bright ...`.

Depth rendering defaults to `-P depth=render`. Set `-P depth=off` to restore
RGB-only observation payloads. Each rendered depth camera adds another image
to an observation or reveal, so this kill-switch is useful when input payload
cost matters.

A camera with no `{cam}_depth` key is unchanged. If a depth thunk fails, its
value is non-numeric or not two-dimensional, or fewer than 1% of its pixels
are valid, the policy emits a descriptive text line and no depth image.

Saved transcripts retain the metric depth label but replace the depth image
with the standard `[image omitted: streamed camera frame]` placeholder. The
HTML viewer shows that placeholder text verbatim below the depth label because
the frame store has no saved frame for rendered depth. This is a known
cosmetic artifact; the metric label remains available in the report.

## Inkling on Tinker

Tinker serves Inkling and Inkling-Small directly through the Messages API.
Set its key and select the model; the provider prefix infers the wire and keeps
the full model id required by the endpoint:

```bash
export TINKER_API_KEY=tk-...

inspect-robots "pick up the cube" --policy agent \
    -P model=thinkingmachines/Inkling -P effort=low \
    --embodiment cubepick
```

With effort unset, Inkling inherits Tinker's own default, documented as high in
the thinking-effort cookbook. That can increase control latency because the arm
stands still while the model thinks; pass `-P effort=low` for latency-sensitive
runs or to pin the plugin's pre-0.23 behavior. The endpoint accepts `low`,
`medium`, `high`, `xhigh`, and `max`; `effort=none` is translated to disabled
thinking, while `minimal` remains unsupported.

Tinker currently reports `input_tokens: 0` because input usage appears in its
cache-creation and cache-read counters. EvalLog input-token statistics and the
live `in=0` transcript line therefore undercount input even though requests are
processed normally. Tinker is a beta service. `-P speed=fast` is a
Claude-on-Anthropic-API option and Tinker silently ignores it, returning HTTP
200 at normal speed. Extended-context model ids ending in `:peft:262144` have
not been tested with this plugin.

## Fast mode on Claude

`-P wire=messages` drives Claude through the native Messages API instead of
the OpenAI-compat endpoint. That is the only way to reach fast mode, which
serves the same model at up to 2.5x higher output tokens per second:

```bash
inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-opus-5 -P wire=messages -P speed=fast \
    --embodiment cubepick
```

Direct-provider model-id handling follows the provider table: Anthropic takes
the bare Claude id, while Tinker keeps `thinkingmachines/`. A Messages run is
refused up front when its model resolves to an endpoint that does not serve
`/v1/messages`, with a fix for a missing prefix, provider key, or an OpenRouter
`:variant` suffix. Pass `-P base_url=...` to point at a compatible Messages
gateway yourself.

> [!NOTE]
> With `-P base_url=...` and no `-P api_key_env=`, the Messages wire sends
> `$ANTHROPIC_API_KEY` to that host. The other wires default to
> `$OPENROUTER_API_KEY` instead. Name the variable explicitly
> (`-P api_key_env=MYGW_KEY`) when the gateway takes its own credential, and
> point it at an unset variable to send no key at all.

Fast mode costs roughly double the standard price on both input and output
(see [Anthropic's pricing](https://www.anthropic.com/pricing)), and it draws on
a rate limit separate from standard capacity, so a fast-mode run can hit a 429
while standard quota sits idle. It is available on Claude Opus 5 and Opus 4.8,
on the Claude API only: not Bedrock, Vertex, Foundry, or Claude Platform on
AWS. A rejection that names fast mode is turned into an error naming the fix.

With effort unset or set to a named level, this wire requests adaptive
thinking. Pre-4.6 models such as Sonnet 4.5 and Haiku 4.5 do not support
adaptive thinking; pass `-P effort=none` to disable thinking and use them on
`wire=messages`, or use `-P wire=chat`.

The Messages API requires an output cap, so `-P max_output_tokens=` defaults to
`16000` here. Thinking bills against that same cap, and a response truncated at
the limit is an error naming the knob rather than a silently missing tool call.
On Anthropic's endpoint, keep `-P effort=` at `high` or below: `xhigh` and
`max` want a cap of 64000 or more, which needs streaming this client does not
implement yet. Tinker accepts `xhigh` and `max` with the plugin's non-streaming
request shape.
The read timeout scales with the cap and tops out at 600 s per attempt, so a
large cap plus retries can sit for several minutes before failing.

Prompt caching is automatic on this wire. Requests use up to three ephemeral
breakpoints: the system prompt, the newest elided-image anchor when one exists,
and the final message. Check
`record.metadata["llm_usage"]["cache_read_input_tokens"]` to verify cache hits;
it should become positive after the first ordinary call.
Anthropic searches only 20 blocks behind a breakpoint, so a cycle with heavy
retry or on-demand rejection churn can cause one silent full-prefix rewrite
and a temporary zero cache-read count. A final nudge also changes wire shape
once it is superseded. Both are cost blips rather than errors, and the anchor
normally restores the hit on the next cycle.

## Reasoning effort on OpenAI models

Recent OpenAI reasoning models can reject function tools on the Chat
Completions wire with an error like this:

```text
Function tools with reasoning_effort are not supported. To use function
tools, use /v1/responses or set reasoning_effort to 'none'.
```

This is a Chat Completions API restriction, not an inspect-robots bug. Use a
direct OpenAI endpoint and select the Responses wire to keep reasoning enabled:

```bash
inspect-robots "pick up the cube" --policy agent \
    -P model=openai/gpt-5.6-sol -P wire=responses -P effort=medium \
    --embodiment cubepick
```

To stay on Chat Completions and disable reasoning instead, pass
`-P effort=none`. It sends the literal `reasoning_effort: "none"`; no nested
quoting is required.
