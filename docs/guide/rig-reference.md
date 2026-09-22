# Rig reference

Set up a physical rig, check its cameras and motors, and run policies with
Inspect Robots. The hardware examples use the `inspect-robots-yam` plugin;
substitute your embodiment's plugin and setup instructions for other robots.
For model selection, reasoning effort, task generation, grading, voice, and
evaluation sets, see [example commands](examples.md).

Replace `robot.example.com`, `robot-user`, and `gpu-box` with your own hosts
and account. `9XXX` is a placeholder: replace it with a free port in the
9000s, using the same value at both ends of each connection. The commands
below assume `inspect-robots setup` has selected your default embodiment.
Keep an e-stop within reach whenever the arms can move.

## Install and update

On the robot host, create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -U --refresh "inspect-robots[rerun]" inspect-robots-yam \
    inspect-robots-agent inspect-robots-voice
```

For updates, activate your existing environment and repeat the install
command. `--refresh` bypasses uv's cache so it sees recent releases. Do not
run `uv sync` afterwards in this pip-managed environment: it reconciles
packages with a project's lockfile and can remove the packages just added.

## Select and configure a rig

Use a separate config file for each rig. Set an absolute path before running
the setup wizard, for example:

```bash
export INSPECT_ROBOTS_CONFIG="$HOME/.config/inspect-robots/my-rig.ini"
inspect-robots setup
```

The wizard picks the default policy and embodiment, finds cameras, and asks
about the behavior settings exposed by the embodiment plugin. Run it again
after rewiring cameras. Keep the same `INSPECT_ROBOTS_CONFIG` for later
commands, or pass `--config` explicitly. A rig's working directory can also
have a `.env` that sets this variable; use an absolute path in that file
because `~` is not expanded there. See
[several rigs on one host](cli.md#several-rigs-on-one-host).

## Live view over SSH

Install `inspect-robots[rerun]` in an environment on your laptop too. Add a
host entry to the laptop's `~/.ssh/config` to forward the viewer port from
the robot host back to your laptop:

```text
Host my-robot
  HostName robot.example.com
  User robot-user
  RemoteForward 9XXX localhost:9XXX
  ExitOnForwardFailure yes
```

Start the viewer on the laptop, then connect:

```bash
rerun --port 9XXX
ssh my-robot
```

In the remote shell, activate the robot's environment, select its config,
and run:

```bash
inspect-robots "place the fork on the plate" \
    --rerun-connect rerun+http://127.0.0.1:9XXX/proxy
```

A bare `--rerun-connect` defaults to port 9876, so pass the URL to keep your
chosen port. See [logging and Rerun](logging-and-rerun.md) for stream
shedding and `.rrd` capture details.

## YAM reference geometry

The public [MolmoAct2 calibration thread](https://github.com/allenai/molmoact2/issues/2)
contains camera intrinsics, a mounting reference, and the arm-spacing photo
for its bimanual YAM setup:

- **Top camera:** a D435 mounted 88 cm above the top face of the 2060
  extrusion.
- **Midpoint frame:** the point midway between the arm bases on the wooden
  table, beside the aluminum extrusion. The reported camera position is
  `[0.013927, -0.007471, 0.889581]` m in this frame.
- **Arm bases:** symmetric about the midpoint, with no relative rotation;
  use the spacing photo in the thread.
- **Intrinsics:** calibrate each camera. The thread includes sample D435
  and D405 readouts and discusses the wrist D405 mount.

The published transform is:

```text
T_midpoint_from_camera =
[
  [ 0.999986716, -0.004210873, -0.002972762,  0.013927329],
  [-0.003082845, -0.950802809,  0.309781399, -0.007471385],
  [-0.004130961, -0.309768119, -0.950803159,  0.889581466],
  [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]
]
```

These values describe that reference setup. Measure and calibrate your own
rig before using a transform for control or collision checking.

## YAM health checks

Check an idle rig before an eval: all three cameras should deliver fresh,
non-uniform frames and both arms should report finite joint positions within
their configured limits. The check writes a labeled montage to `health.jpg`:

```bash
inspect-robots-yam-health
```

Run this only while the rig is idle, with both arms at rest or supported
and an e-stop in hand. Connecting and then closing the motor driver drops
motor torque. Do not run it concurrently with an eval. Pass `--json` for a
machine-readable report, or `--skip-cameras` / `--skip-motors` to run one
section.

To aim the cameras, use watch mode. It never touches the motors, so the
torque warning does not apply:

```bash
inspect-robots-yam-health --watch --bind 127.0.0.1 --port 9XXX
```

Choose a watch port different from the Rerun viewer port. Open
`http://127.0.0.1:9XXX/` on the robot host, or forward the watch port from
your laptop in another terminal:

```bash
ssh -N -L 9XXX:localhost:9XXX robot-user@robot.example.com
```

Use the actual user and hostname here, rather than the `my-robot` alias:
that alias would request the viewer's `RemoteForward` again and fail while
the first SSH connection holds its port. Keep this tunnel open and visit
`http://127.0.0.1:9XXX/` on your laptop, using the watch port throughout.
Press Ctrl-C in each terminal to stop the watch server and tunnel.
The stream is unauthenticated; the explicit loopback
bind above limits access to the host and your SSH tunnel. Without `--bind`,
watch mode listens on `0.0.0.0` (port 8807 unless overridden).

## YAM control interfaces and advice

Select joint-space actions with `-E control_interface=joints` (the default)
or 14-D Cartesian end-effector actions with `-E control_interface=eef_pos`.
The policy and embodiment must agree on the resulting control mode;
compatibility checking rejects mismatches before anything moves.

The embodiment supplies built-in advice for its selected interface. To add
facts about your own rig, create local Markdown files such as
`rig-notes/facts.md`, `rig-notes/advice-joints.md`, and
`rig-notes/advice-eef.md`. These are user-authored files, not files shipped
with Inspect Robots. Match your additional advice to the interface:

```bash
inspect-robots "stack the red cube on the blue cube" --policy agent \
    -P model=anthropic/claude-fable-5 \
    -E "docs_extra=$(cat rig-notes/facts.md rig-notes/advice-joints.md)" \
    -E control_interface=joints
inspect-robots "stack the red cube on the blue cube" --policy agent \
    -P model=anthropic/claude-fable-5 \
    -E "docs_extra=$(cat rig-notes/facts.md rig-notes/advice-eef.md)" \
    -E control_interface=eef_pos
```

`docs_extra` is included in the policy's prompt. Only include material you
intend to send to the configured model provider. Omit the argument if you
have no additional advice. To cap an agent's requested speed while checking
a setup, add `-P max_speed_frac=0.1`.

## Worked example: giant Jenga

This combines the SSH viewer, end-effector control, estimated joint-effort
observations, and your local advice files. It uses the messages wire and
fast output with an agent policy.

This example disables the collision guardrail for a task close to the
tower. With that guardrail off, keep the e-stop in hand and supervise every
motion; do not treat the example as safe for unattended operation. Only use
it after checking your rig's workspace and limits.

```bash
inspect-robots run --policy agent \
    --rerun-connect rerun+http://127.0.0.1:9XXX/proxy \
    -E collision_guardrail=false \
    -E report_joint_eff=true \
    -P wire=messages \
    -P model=anthropic/claude-opus-5 \
    -P speed=fast \
    -E "docs_extra=$(cat rig-notes/facts.md rig-notes/advice-eef.md)" \
    -E control_interface=eef_pos \
    --instruction "Pull out the protruded Jenga block and put it on the top of the tower with your left arm. Be careful to not topple the tower with your arm. This is a Giant Jenga. You can use your right arm for visuals but be careful not to hit the tower. Ideally the left arm should engage in this sequence once it is ready: move left arm into position (A) at the far left with the gripper facing the block, approach the block so the block is between grippers without touching the tower face (B), grip the block, retract the block to A, move smoothly vertically upwards to reach the top (vertical offset from A), move left arm rightwards to position the block on top of the tower, then release the gripper."
```

## Pi 0.5 on YAM through OpenPI

The YAM plugin's `molmoact2` policy is a generic `/act` client and can also
connect to Pi 0.5 through a compatible server. This recipe uses the public
[YAM checkpoint](https://huggingface.co/robocurve/pi0.5-yam) and
[OpenPI serving script](https://gist.github.com/jeqcho/60716eef6c2aa7706e2ecd575a7a7a7e).

On a GPU machine with more than 16 GB VRAM, download OpenPI, the checkpoint,
and the script. Review the script before running it, then leave the server
running in its own terminal:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 15a9616
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install json_numpy
uvx --from 'huggingface_hub[cli]' hf download robocurve/pi0.5-yam \
    --local-dir ../pi05-yam
curl -fLO https://gist.githubusercontent.com/jeqcho/60716eef6c2aa7706e2ecd575a7a7a7e/raw/e3ee89434aa2c7056411c8e05697f00784e82876/serve_pi05_yam.py
uv run --no-sync python serve_pi05_yam.py --ckpt ../pi05-yam --port 9XXX
```

Wait for the `serving pi05-yam` message after model loading and JIT warmup.
The server binds `0.0.0.0` by default and has no authentication; use it on a
trusted network, or bind it to loopback with `--host 127.0.0.1` and connect
through an SSH tunnel.

On the robot host, with the YAM config selected:

```bash
inspect-robots run --policy molmoact2 \
    -P name=pi05 -P server_url=http://gpu-box:9XXX \
    -P cam_height=360 -P cam_width=640 \
    -E cam_height=360 -E cam_width=640 -E control_hz=30 \
    --max-steps 3600 \
    --instruction "stack the red block on the blue block"
```

The camera dimensions and control rate match the training setup. At 30 Hz,
3,600 steps represent two minutes of control, excluding inference and other
overhead. `name` only labels the run in logs; keep it aligned with the served
checkpoint. If JAX reports "no kernel image" on a new GPU, the source recipe
uses `uv pip install -U "jax[cuda12]"`; choose a build compatible with your
GPU and driver. Keep an e-stop within reach: this checkpoint can move quickly
and is not intended for unattended operation.

For MolmoAct 2 serving, see [running VLA policies](examples.md#running-vla-policies).
For GR00T, see the [GR00T on YAM cookbook](../cookbooks/gr00t-on-yam.md).
