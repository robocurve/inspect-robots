# inspect-robots-capx:

Code-as-policy manipulation for
[Inspect Robots](https://github.com/robocurve/inspect-robots), backed by the
SAM3, Contact-GraspNet, and Pyroki servers from
[CaP-X](https://github.com/capgym/cap-x). The installed policy name is `capx`.

## Install:

Install the core and plugin in the environment that drives the embodiment:

```bash
pip install inspect-robots inspect-robots-capx
```

CaP-X is not a Python dependency of this package. Its research checkout and
GPU dependencies stay in the separate server environment.

## Start the CaP-X servers:

From an installed CaP-X checkout, start one process for each model:

```bash
# terminal 1
uv run capx/serving/launch_sam3_server.py --port 8114

# terminal 2
uv run capx/serving/launch_contact_graspnet_server.py --port 8115

# terminal 3
uv run python -c 'from capx.serving.launch_pyroki_server import main; main(robot="panda_description", port=8116)'
```

The Pyroki launcher is different from the other two. Upstream
`launch_pyroki_server.py` calls a bare `main()` and does not parse command-line
flags, so invoking the file with `--robot` or `--port` silently ignores them.
Use the `python -c` form above and select the robot description that matches
the embodiment. The server's full actuated configuration may include a
gripper joint; the client strips the returned vector to the bound arm size and
keeps the full vector only as the next IK warm start.

## Run a policy:

The v1 profile needs a joint-space embodiment. The core `cubepick` mock uses
2-D end-effector deltas and is intentionally incompatible.

```bash
export ANTHROPIC_API_KEY=sk-ant-...

inspect-robots "pick up the red cube" --policy capx \
    --embodiment <joint-space-embodiment> \
    -P model=anthropic/claude-fable-5 \
    -P sam3_url=http://gpu-box:8114 \
    -P graspnet_url=http://gpu-box:8115 \
    -P pyroki_url=http://gpu-box:8116
```

Model routing, API key selection, and the `chat` versus `responses` wire follow
the [inspect-robots-agent](../inspect-robots-agent/) plugin. Model strings come
from `-P model=...` or `$INSPECT_ROBOTS_MODEL`.

## Required embodiment profile:

`bind()` checks the complete profile before a rollout begins:

- The action space is a one-dimensional `Box` with at least two dimensions.
- `ActionSemantics.control_mode` is `joint_pos` and `gripper` is not `none`.
- A `dim_labels` entry named `gripper` locates the gripper. If labels are
  absent, the final action dimension is the fallback.
- The action box has finite low and high bounds. The selected gripper bound is
  high for open and low for closed by default. Set
  `gripper_open_is_high=false` for an inverted embodiment.
- Exactly one `StateSpec` field has the full action-vector shape. It is the
  proprioceptive reference for motion and stop holds.
- `control_hz` is finite and positive.
- `camera=None` selects the sole declared camera. Set `camera=NAME` when the
  embodiment declares several cameras.

Pyroki solves in its URDF base frame. This plugin treats world and robot base
as the same frame. Match the server's robot model, the embodiment's action
dimensions, and the task's coordinates.

## Depth and camera metadata:

Core observations carry RGB images, but do not reserve a depth slot. An
embodiment can opt into grasp planning with entries in `Observation.extra`:

```python
Observation(
    images={"front": rgb},
    state={"joint_pos": joint_config},
    extra={
        "depth": lambda: camera.read_depth(),
        "intrinsics": lambda: camera.intrinsics(),
        "extrinsics": lambda: camera.camera_to_base(),
    },
)
```

Each value may be the array itself or a zero-argument callable returning it.
The callable form is recommended for real embodiments. Trial records retain
`Observation.extra` at every control step, so raw per-step depth arrays can
otherwise become an in-memory depth-video buffer. The defaults expect depth
shape `(H, W)`, intrinsics shape `(3, 3)`, and camera-to-base extrinsics shape
`(4, 4)`. The `depth_key`, `intrinsics_key`, and `extrinsics_key` arguments can
rename them.

Missing depth does not block segmentation or IK. `plan_grasp()` raises inside
the code namespace with a message naming the missing key, so the model sees the
error on stderr and can choose another route. Contact-GraspNet poses are in the
camera frame; model code transforms them with `extrinsics @ pose` before IK.

## Execution model:

Each LLM turn returns raw Python, optionally wrapped in one Markdown code
fence. The namespace persists for the trial, including variables and imports.
The helpers expose the current observation as `obs` and provide:

- `segment(text)` for SAM3 text-prompt masks
- `plan_grasp(mask)` for camera-frame grasp poses and scores
- `solve_ik(position, quaternion_wxyz)` for arm joints
- `move_to_joints(joints)` for a speed-limited joint target
- `open_gripper()` and `close_gripper()` for speed-limited gripper ramps

Motion helpers queue full joint targets. The complete queue becomes one open
loop `ActionChunk`, so perception in that Python turn uses the initial
observation. The next policy turn observes the executed result. Within a turn,
the motion cursor chains across every arm and gripper call. At the next turn it
is seeded again from observed proprioception.

Interpolation uses `max_speed_frac / control_hz` of each action-box range per
step, capped by the same native-dtype 5 percent backstop as the core
`DeltaLimitApprover`. This prevents default guardrails from silently rewriting
steps within a chunk. A lagging arm can still make the first target of the next
turn differ from the last approved action, so the approver may clamp that turn
boundary.

After execution, the LLM receives its code, captured stdout, and captured
stderr with any traceback. Execution reports keep their final 16,000
characters when truncation is needed, since exceptions appear at the tail.
The report is appended to the transcript before `act()` returns. The full,
untruncated code is also stored in the first queued action's `meta["code"]`.

`FINISH` and `GIVE_UP` return a one-action hold with the core policy-stop
metadata. A clean turn with no queued motion continues inside the same
`act()` call. `max_llm_calls` bounds all turns in a trial.
`max_code_failures` counts consecutive Python exceptions across `act()` calls.
A clean Python turn resets it. Code that queues motion and then raises still
returns that motion chunk, while the error remains counted for the next turn.

## Trust model:

> [!WARNING]
> Model-generated Python executes in-process with the evaluator's user
> privileges. It can import installed packages, read files, make network
> requests, and mutate process state. This integration follows CaP-X; it is not
> a security sandbox. Run untrusted models inside a container or another
> external isolation boundary.

Every queued robot action still passes through the rollout approver. CLI runs
install `ClampApprover` plus `DeltaLimitApprover` by default. The Python
`eval()` API defaults to the permissive `AutoApprover`, so programmatic real
robot runs must wire guardrails explicitly:

```python
from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_capx import CapxPolicy

# `embodiment` is your constructed joint-space adapter.
space = embodiment.info.action_space
guardrails = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
policy = CapxPolicy(model="anthropic/claude-fable-5")

eval(task, policy, embodiment, approver=guardrails)
```

Guardrails constrain robot actions. They do not constrain what executed Python
can do to the evaluator process.

## Configuration:

CLI policy arguments use `-P key=value`.

| Argument | Default | Contract |
|---|---|---|
| `model` | `$INSPECT_ROBOTS_MODEL` | OpenRouter-style model id or provider-prefixed direct model. |
| `base_url` | provider routing | Custom OpenAI-compatible endpoint. |
| `api_key_env` | `OPENROUTER_API_KEY` with custom URL | Environment variable holding a custom endpoint key. |
| `wire` | `chat` | `chat` or `responses`. |
| `temperature` | omitted | Sampling temperature sent when set. |
| `effort` | `low` | Reasoning effort, or null to omit the field. |
| `sam3_url` | `http://127.0.0.1:8114` | SAM3 server base URL. |
| `graspnet_url` | `http://127.0.0.1:8115` | Contact-GraspNet server base URL. |
| `pyroki_url` | `http://127.0.0.1:8116` | Pyroki server base URL. |
| `camera` | sole camera | RGB camera used by perception. |
| `depth_key` | `depth` | `Observation.extra` depth entry. |
| `intrinsics_key` | `intrinsics` | `Observation.extra` camera intrinsics entry. |
| `extrinsics_key` | `extrinsics` | `Observation.extra` camera-to-base transform entry. |
| `max_llm_calls` | `100` | Total LLM calls per trial. |
| `max_code_failures` | `3` | Consecutive exception turns before policy failure. |
| `max_speed_frac` | `0.1` | Action-range fraction per second before the per-step backstop. |
| `request_timeout_s` | `120` | Total wall-clock budget for one helper HTTP call, retries included. |
| `gripper_open_is_high` | `true` | Use the gripper high bound as open. |
| `transcript_echo` | `false` | Echo code and execution feedback to stderr. |
| `transport` | `None` | Programmatic `httpx` transport injection for tests. |
| `env` | process environment | Programmatic provider-environment override for tests. |

Each helper request retries transient transport errors, HTTP 429, and server
5xx responses with exponential backoff. The total retry loop stays within
`request_timeout_s`; one attempt receives `min(remaining_budget, 30 seconds)`.

## Transcripts:

`CapxPolicy.transcript()` returns the current conversation as an isolated copy.
Camera data URLs are replaced with omission markers, while code, stdout,
stderr, and stop words remain. Inspect a saved run with:

```bash
inspect-robots inspect LOG.json --transcript
inspect-robots view LOG.json
```

## Troubleshooting:

- A 503 during initial use usually means a model server is still loading.
  Requests retry within `request_timeout_s`; increase that value for slow GPU
  cold starts.
- A connection error names the failed URL and the exact launch command for that
  service. Check host routing and firewall access from the evaluator machine.
- A Pyroki response shorter than the arm size means the server robot and
  embodiment do not match. Restart Pyroki with the correct robot description.
- Five-degree-of-freedom arms may not reach arbitrary six-degree-of-freedom
  grasp poses. Filter for top-down grasps in the task prompt or use a matching
  IK target convention.
- Direct OpenAI reasoning models may require `-P wire=responses`. Chat-only
  endpoints should keep `-P wire=chat`; `-P effort=none` can disable reasoning
  where function or code-generation requests reject it.
