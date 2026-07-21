# Quickstart

## Install

In a fresh directory (or your existing project), create a virtual environment
and install Inspect Robots with Rerun visualization:

```bash
uv venv && uv pip install "inspect-robots[rerun]"
```

For the NumPy-only core:

```bash
uv venv && uv pip install inspect-robots
```

Any virtual environment workflow works. Activate it once (`source
.venv/bin/activate`; `.venv\Scripts\activate` on Windows) and call
`inspect-robots` directly.

:::note
Invoke the CLI as plain `inspect-robots`, not `uv run inspect-robots`. Inside a
uv project, `uv run` re-syncs to the lockfile and can uninstall what `uv pip
install` just added.
:::

## Run your first evaluation

The dependency-free `CubePick` mock world lets you exercise the whole stack with
no hardware or simulator:

```python
from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

task = Task(
    name="cubepick-reach",
    scenes=[Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(5)],
    scorer=success_at_end(),
    max_steps=80,
)

(log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment())
print(log.status)                    # "success"
print(log.results.metrics)           # {"success_at_end": 1.0}
```

`eval()` returns a list of [`EvalLog`](/api/#inspect_robots.log.EvalLog) (one per task, mirroring
Inspect AI). Each log is immutable, schema-versioned, and written to `log_dir`.

## Use registry names

`task`, `policy`, and `embodiment` may also be registry names, the same
mechanism the CLI uses:

```python
from inspect_robots import eval

(log,) = eval("cubepick-reach", "scripted", "cubepick")
```

## From the command line

The CLI resolves any registered task, policy, or embodiment (builtins plus
installed plugins):

```bash
inspect-robots list                                          # all registered components
inspect-robots list policies                                 # just policies
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick -P chunk_size=6
inspect-robots inspect logs/cubepick-reach_*.json            # print a saved log
inspect-robots view logs/cubepick-reach_*.json               # render an HTML report
inspect-robots video logs/cubepick-reach_*.json              # camera frames to MP4
```

`view` writes a self-contained HTML page. `video` needs a run captured with
`--store-frames` and the `ffmpeg` binary on PATH. Every command is covered in
[the CLI guide](cli.md).

## On a real robot

Install the plugin for your rig and set your defaults once:

```bash
source .venv/bin/activate
uv pip install inspect-robots-yam   # provides the molmoact2 policy + yam_arms rig
inspect-robots setup
```

The wizard picks your defaults and finds your cameras, then writes
`~/.config/inspect-robots/config.ini`. On a different rig, install its plugin
instead and type its component names at the prompts; to write the config file
by hand, see [the CLI guide](cli.md).

The `molmoact2` policy is only a client: nothing moves until the MolmoAct2
server is listening, and the server does not start itself or survive a reboot
(full setup in the
[yam plugin README](https://github.com/robocurve/inspect-robots-yam#install-on-the-robotgpu-machine)):

```bash
# On the GPU machine, from the MolmoAct2 repo. Leave it running, e.g. in tmux:
python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202
curl http://127.0.0.1:8202/act      # 200 means the server is ready
```

On a different rig, start whatever serves your policy instead; in-process
policies (such as `agent` or the mock `scripted`) need no server.

Then tell the robot what to do:

```bash
inspect-robots "place the fork on the plate"
```

Every run opens a live Rerun viewer streaming the cameras, proprioception,
and actions straight from the eval pipeline, so you watch exactly what the
policy sees while the robot moves. CLI flags override any default
(`--no-rerun`, `--no-store-frames`, `--max-steps 300`, ...).

Safety guardrails (a bounds clamp plus a per-step delta limit derived from
the embodiment's action space) are wired into every CLI run by default, for
every policy. Turning them off requires an explicit `--disable-guardrails`.

## Drive the robot with an LLM

The policy slot is not limited to VLAs. With the
[inspect-robots-agent](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent)
plugin, a frontier LLM drives the same rig through tool calls, one
approver-checked motion chunk per call.

Put a `.env` with your API key in the working directory (the CLI loads it
automatically):

```ini
ANTHROPIC_API_KEY=sk-ant-...
```

Install the add-on:

```bash
uv pip install inspect-robots-agent
```

Run the LLM on the robot:

```bash
inspect-robots "place the fork on the plate" --policy agent \
    -P model=anthropic/claude-fable-5 -P effort=low
```

No rig? The same policy drives the mock world, no hardware or GPU required
(an exported key works in place of `.env`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-fable-5 -P effort=low --embodiment cubepick
```

Read the recorded agent conversation with
`inspect-robots inspect LOG.json --transcript`, or open the HTML report with
`inspect-robots view LOG.json`; for `--store-frames` runs it includes the
camera frames the model saw.

## Generate robot policy code with CaP-X

The
[inspect-robots-capx](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-capx)
plugin evaluates a code-as-policy agent in the same policy slot. The LLM
writes Python against SAM3 segmentation, Contact-GraspNet planning, Pyroki
IK, and speed-limited joint-motion helpers.

```bash
uv pip install inspect-robots-capx

inspect-robots "place the fork on the plate" --policy capx \
    --embodiment <joint-space-embodiment> \
    -P model=anthropic/claude-fable-5 -P sam3_url=http://gpu-box:8114
```

See the plugin README for embodiment requirements, model-server bringup, and
the model-code trust boundary.

## Run in simulation

The same instruction runs on your configured simulator instead of the real
robot (the mapping is explained in [the CLI guide](cli.md)):

```bash
inspect-robots "place the fork on the plate" --sim
```

## Next steps

- [Concepts](concepts.md): the core abstractions.
- [Writing a benchmark](writing-a-benchmark.md): define your own `Task`.
- [Policies and embodiments](policies-and-embodiments.md): plug in a real VLA or robot/sim.
- [Plugins](plugins.md): the first-party adapters for ROS, Isaac Lab, XPolicyLab, LLM agents, and CaP-X.
- [The CLI](cli.md): configuration, `--sim` mapping, scoring, and every command.
