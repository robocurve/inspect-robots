# inspect-robots-xpolicylab

Drive any [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)-served policy
from [Inspect Robots](https://github.com/robocurve/inspect-robots).

XPolicyLab unifies **40+ VLA and imitation-learning policies** (π0 / π0.5,
GR00T N1.7, OpenVLA-OFT, RDT-1B, SmolVLA, ACT, Diffusion Policy, …) behind one
policy-server contract: each policy runs in its own conda/uv environment and
serves inference over a websocket protocol. This plugin is the Inspect Robots
side of that wire: a `Policy` adapter that connects to a running XPolicyLab
policy server, so the whole policy zoo becomes evaluable with any Inspect
Robots embodiment and task.

The adapter speaks XPolicyLab's msgpack-over-websocket protocol directly — it
does **not** depend on the `xpolicylab` package (which is not on PyPI). Only
the policy *server* needs an XPolicyLab checkout.

## Install

```bash
pip install inspect-robots-xpolicylab
```

The `xpolicylab` policy then appears in `inspect-robots list policies`.

## Quickstart

**Terminal 1 — serve a policy from your XPolicyLab checkout** (its own
environment, possibly another machine; see the
[XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab#readme) for
per-policy install/checkpoint details):

```bash
cd XPolicyLab/policy/Pi_0
bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> \
    <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_env> 19000 0.0.0.0
```

**Terminal 2 — evaluate it with Inspect Robots:**

```bash
inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
    -P url=ws://gpu-box:19000 -P cameras=cam_head:base_rgb
```

or programmatically:

```python
from inspect_robots import eval
from inspect_robots_xpolicylab import XPolicyLabPolicy

with XPolicyLabPolicy(url="ws://gpu-box:19000", cameras="cam_head:base_rgb") as policy:
    log = eval("my-task", policy, "isaacsim")
```

Constructing the policy and reading `.info` never touch the network — the
websocket connects on first `reset()`/`act()`, with retries while the policy
server cold-starts (loading VLA weights can take minutes).

## Configuration

| Arg | Default | Meaning |
| --- | --- | --- |
| `url` | `ws://localhost:19000` | policy server websocket URL |
| `action_type` | `joint` | `joint` → `*arm_joint_state` actions; `ee` → `*ee_pose` |
| `arms` | `1` | `1` or `2` (dual-arm uses `left_*`/`right_*` keys) |
| `arm_dim` / `ee_dim` | `7` / `1` | per-arm joint dims / gripper dims |
| `cameras` | `cam_head:cam_head` | XPolicyLab camera slot → Inspect Robots camera name |
| `state_map` | see below | XPolicyLab state key → Inspect Robots state key |
| `required_state_keys` | joint: `joint_pos,gripper`; ee: none | state keys enforced by compatibility checking |
| `action_keys` + `action_dim` | derived | explicit ordered action-dict keys for exotic setups |
| `camera_height` / `camera_width` | unset | declare camera resolution for compatibility checks |
| `control_hz` | unset | chunk playback rate; also sent as `additional_info.frequency` |
| `name` | `xpolicylab` | policy name in logs (tag runs e.g. `xpolicylab:pi0`) |
| `connect_timeout_s` / `request_timeout_s` | `30` / `120` | client timeouts |
| `connect_attempts` / `connect_retry_delay_s` | `10` / `5` | cold-start retry budget |

Mapping-valued args accept a compact string form for the CLI:
`-P cameras=cam_head:base_rgb,cam_wrist:wrist_rgb`.

Default `state_map` (all XPolicyLab state fields are optional; mapped keys
missing from an observation are simply not sent):

| XPolicyLab key | Inspect Robots key |
| --- | --- |
| `arm_joint_state` | `joint_pos` |
| `ee_joint_state` | `gripper` |
| `ee_pose` | `eef_pose` |

## Action mapping

Each `infer` reply is an action chunk — one dict per future control step.
Dicts flatten to vectors in a fixed order: for each arm (left then right), the
arm key then the end-effector key. The default single-arm joint profile is
`concat(arm_joint_state, ee_joint_state)` → dim 8, matching the
`inspect-robots-isaacsim` default Franka profile.

## Lifecycle notes

- `reset(scene)` ends the previous trial server-side (`trial_end`) and starts
  a fresh `trial_id`; `close()` ends any open trial and says goodbye.
- `eval()` closes embodiments it resolves, **not** policies — use `with`
  / `close()` in programmatic runs. A best-effort `atexit` hook covers
  registry-resolved CLI runs.
- If the socket drops mid-eval, the trial in flight errors and the next
  `reset()`/`act()` reconnects (replaying the `hello` handshake).

## Protocol compatibility

Validated against XPolicyLab commit
`fe71eb54675cef495fea817a637386a4f4529153`. The protocol carries no version
field; if upstream changes the wire format, `_protocol.py` is the single small
module to diff and update.
