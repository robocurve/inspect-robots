<div align="center">

<img src="https://raw.githubusercontent.com/robocurve/inspect-robots/main/docs/assets/inspect-robots-logo.svg" alt="Inspect Robots logo — a line-art robot inspecting a dot through a magnifying lens" width="160">

# Inspect Robots

### An open-source evaluation framework for benchmarking AI and robots in the physical world

Define a robotics benchmark once, then run any policy against any compatible
embodiment (a real robot or a simulator) with reproducible logs and first-class
[Rerun](https://github.com/rerun-io/rerun) visualization.

If you know [Inspect AI](https://inspect.aisi.org.uk/), this is that for robotics.

![Status: alpha](https://img.shields.io/badge/status-alpha-blue)
[![CI](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml)
[![Docs](https://github.com/robocurve/inspect-robots/actions/workflows/docs.yml/badge.svg)](https://inspectrobots.org/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://github.com/robocurve/inspect-robots)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-blue)](https://github.com/robocurve/inspect-robots)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml)
[![Docs coverage](https://img.shields.io/badge/public%20docstrings-100%25-brightgreen)](https://github.com/robocurve/inspect-robots/actions/workflows/ci.yml)

[**Documentation**](https://inspectrobots.org/) ·
[Quickstart](https://inspectrobots.org/guide/quickstart/) ·
[Concepts](https://inspectrobots.org/guide/concepts/) ·
[For LLMs](https://inspectrobots.org/llms.txt)

</div>

> [!NOTE]
> This project is in early development. The API may change between releases, so pin a version before depending on it.

---

## Install

In a fresh directory (or your existing project), create a virtual environment
and install:

```bash
uv venv && uv pip install "inspect-robots[rerun]"
```

The `rerun` extra powers the live run viewer. For the numpy-only core:

```bash
uv venv && uv pip install inspect-robots
```

Any venv workflow works. Activate it once (`source .venv/bin/activate`;
`.venv\Scripts\activate` on Windows) and call `inspect-robots` directly,
as shown below.

> [!NOTE]
> Invoke the CLI as plain `inspect-robots`, not `uv run inspect-robots` —
> inside a uv project, `uv run` re-syncs to the lockfile and silently
> uninstalls what `uv pip install` just added.

## Quickstart

Install the plugin for your rig and set your defaults once:

```bash
source .venv/bin/activate
uv pip install inspect-robots-yam   # provides the molmoact2 policy + yam_arms rig
inspect-robots setup
```

The wizard picks your defaults and finds your cameras, then writes
`~/.config/inspect-robots/config.ini`. On a different rig, install its plugin
instead and type its component names at the prompts; to write the config file
by hand, see [the CLI guide](https://inspectrobots.org/guide/cli/).

The `molmoact2` policy is only a client: nothing moves until the MolmoAct2
server is listening, and the server does not start itself or survive a
reboot (full setup in the
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

### Drive the robot with an LLM

The policy slot is not limited to VLAs. With the
[inspect-robots-agent](plugins/inspect-robots-agent/) plugin, a frontier LLM
drives the same rig through tool calls, one approver-checked motion chunk
per call.

Put a `.env` with your API key in the working directory (the CLI loads it
automatically; [.env.example](.env.example) is a template):

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

Read the recorded agent conversation with
`inspect-robots inspect LOG.json --transcript`, or open the HTML report with
`inspect-robots view LOG.json` — for `--store-frames` runs it includes the
camera frames the model saw.

### Generate robot policy code with CaP-X

The [inspect-robots-capx](plugins/inspect-robots-capx/) plugin evaluates a
code-as-policy agent in the same policy slot. The LLM writes Python against
SAM3 segmentation, Contact-GraspNet planning, Pyroki IK, and speed-limited
joint-motion helpers.

```bash
uv pip install inspect-robots-capx

inspect-robots "place the fork on the plate" --policy capx \
    --embodiment <joint-space-embodiment> \
    -P model=anthropic/claude-fable-5 -P sam3_url=http://gpu-box:8114
```

See the [plugin README](plugins/inspect-robots-capx/) for embodiment
requirements, model-server bringup, and the model-code trust boundary.

### Run in simulation

The same instruction runs on your configured simulator instead of the
real robot:

```bash
inspect-robots "place the fork on the plate" --sim
```

### More CLI commands

The full command line resolves any registered task/policy/embodiment
(builtins + installed plugins). List what is registered:

```bash
inspect-robots list
```

Run a registered task with explicit components:

```bash
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
```

Pretty-print a saved eval log:

```bash
inspect-robots inspect logs/cubepick-reach_*.json
```

Render a saved eval log as a self-contained HTML report:

```bash
inspect-robots view logs/cubepick-reach_*.json
```

Render a `--store-frames` run's camera frames to MP4 videos (needs the
`ffmpeg` binary on PATH):

```bash
inspect-robots video logs/cubepick-reach_*.json
```

### Python API

Everything is a Python API. No hardware or simulator needed: the
dependency-free `CubePick` mock world exercises the whole stack:

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

# The two swappable inputs: a policy (VLA) and an embodiment (robot/sim).
(log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment())
print(log.status, log.results.metrics)   # success {'success_at_end': 1.0}
```

## Why Inspect Robots

- **Real-world first.** Interfaces assume real-robot reality: human-in-the-loop
  reset, no privileged success oracle, wall-clock control rate. Simulators just
  offer more (seeding, privileged success, rendering) via opt-in capabilities.
- **Compatibility checked up front.** Before any rollout, the
  `(policy, embodiment)` pair is validated — action/observation spaces,
  semantics, control rate, scene realizability — and fails fast if not.
- **Reproducible.** Every run yields an immutable, schema-versioned `EvalLog`
  with the resolved config, git revision, and package versions. It is re-readable
  across releases and re-scorable offline.
- **Light core.** Depends only on NumPy. Rerun and simulator/VLA backends are
  optional extras and separately installable plugins.
- **Safe unattended.** An explicit error taxonomy separates "record and continue"
  from "halt and require a human", so a faulted robot never auto-advances overnight.
- **Rerun visualization.** Stream camera images, 3D poses, joint/action
  time-series, and success markers to a `.rrd` recording. Logging is non-blocking:
  a slow viewer connection drops camera frames first (whole steps only under
  sustained stall) instead of delaying the robot control loop, and camera
  streams are JPEG-compressed by default.
- **Pluggable.** Backends ship as separate packages — the first-party plugins
  below, and rig plugins like `inspect-robots-yam`. Entry points make them
  appear in `inspect-robots list` automatically.
- **VLA-native.** Action chunking, open-loop execution, and ACT/ALOHA temporal
  ensembling are built in, with action *semantics* (control mode, rotation
  representation, gripper, frame) that make compatibility and ensembling correct.

## First-party plugins

Both halves of an eval (the "body" and the "brain") have a ready-made
adapter shipped from this repo as separate packages:

- **[inspect-robots-ros](plugins/inspect-robots-ros/)**: run evals on ROS 1 or
  ROS 2 arms through rosbridge, with no ROS installation on the eval machine
  (`--embodiment ros`).
- **[inspect-robots-isaacsim](plugins/inspect-robots-isaacsim/)**: run evals
  against an [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) simulation
  (`--embodiment isaacsim`).
- **[inspect-robots-xpolicylab](plugins/inspect-robots-xpolicylab/)**: drive
  any [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)-served policy.
  One adapter puts its zoo of 40+ VLAs (π0/π0.5, GR00T, OpenVLA-OFT, RDT-1B,
  SmolVLA, ACT, …) behind `--policy xpolicylab -P url=ws://gpu-box:19000`.
- **[inspect-robots-agent](plugins/inspect-robots-agent/)**: let a frontier
  LLM (Claude, GPT, anything behind an OpenAI-compatible API) drive any
  embodiment through tool calls, as a first-class policy. The same
  `--policy agent` runs ad-hoc instructions and scores on registered tasks
  next to fine-tuned VLAs.
- **[inspect-robots-capx](plugins/inspect-robots-capx/)**: evaluate CaP-X-style
  code-as-policy agents against a joint-space embodiment. Model-generated
  Python calls separately served SAM3, Contact-GraspNet, and Pyroki helpers,
  then queues approver-checked joint targets behind `--policy capx`.

```bash
# Isaac Lab world + a π0 checkpoint served by XPolicyLab, evaluated end to end:
inspect-robots run --task my-task --embodiment isaacsim \
    --policy xpolicylab -P url=ws://gpu-box:19000 -P cameras=cam_head:base_rgb

# Claude driving the mock world, no hardware or GPU required:
export ANTHROPIC_API_KEY=sk-ant-...
inspect-robots "pick up the cube" --policy agent \
    -P model=anthropic/claude-fable-5 -P effort=low --embodiment cubepick
```

### Real robots via ROS

The ROS embodiment connects to any ROS 1 or ROS 2 arm that exposes standard
joint, compressed-image, and optional pose topics through `rosbridge_server`.
It publishes joint-position commands at a configured control rate and works
with every compatible policy, including `agent` and XPolicyLab-served VLAs.

```bash
uv pip install inspect-robots-ros

inspect-robots run --task my-task --policy agent --embodiment ros \
    -E url=ws://robot:9090 \
    -E joints=joint1,joint2,joint3,joint4,joint5,joint6 \
    -E command_topic=/joint_trajectory_controller/joint_trajectory \
    -E action_low=-3.1,-2.2,-2.9,-3.1,-2.9,-3.1 \
    -E action_high=3.1,2.2,2.9,3.1,2.9,3.1
```

Swap `--policy agent` for `--policy xpolicylab -P url=ws://gpu-box:19000` to
evaluate any XPolicyLab-served VLA on the same arm; the `-E` robot arguments
stay unchanged. Robot bringup, controller mappings, safety requirements,
camera configuration, and reset behavior are documented in the
[ROS plugin README](plugins/inspect-robots-ros/).

Safety guardrails (a bounds clamp plus a per-step delta limit derived from
the embodiment's action space) are wired into every CLI run by default, for
every policy. Turning them off requires an explicit `--disable-guardrails`.
Persist your usual setup once with `inspect-robots config set embodiment NAME`
and `inspect-robots config set policy NAME`, then a bare
`inspect-robots "wipe the table"` does the rest.

## How it maps to Inspect AI

If you know [Inspect AI](https://inspect.aisi.org.uk/), you already know Inspect Robots.

| Inspect AI | Inspect Robots |
|---|---|
| `Model` | `Policy` (VLA) **+** `Embodiment` *(two inputs)* |
| `Task = dataset + solver + scorer` | `Task = scenes + controller + scorer` |
| `Sample` | `Scene` |
| `Solver` chain | `Controller` middleware (chunking, ensembling, smoothing) |
| `eval()` → `EvalLog` | `eval()` → `EvalLog` |
| `@task` / `@solver` / `@scorer` + registry | `@task` / `@policy` / `@embodiment` / `@scorer` + entry points |

This repository is the framework. Concrete benchmarks live in
[WorldEvals](https://github.com/robocurve/worldevals), the benchmark catalog,
and backend adapters live in separate plugin packages.

## Documentation

Full guides and an auto-generated API reference live at
**[inspectrobots.org](https://inspectrobots.org/)**.
LLM-friendly versions: [`llms.txt`](https://inspectrobots.org/llms.txt)
and [`llms-full.txt`](https://inspectrobots.org/llms-full.txt).

## Development

> **Dependency changes:** after editing dependencies in `pyproject.toml`, run
> `uv lock` and commit the updated lockfile. CI installs with
> `uv sync --locked` and fails with "the lockfile needs to be updated" if you
> forget. Day-to-day conventions (PR-only `main`, the required `ci-ok` check,
> one-click releases) are documented in [`CLAUDE.md`](CLAUDE.md).

```bash
uv venv && uv pip install -e ".[dev]"
uv run pre-commit install          # ruff + mypy on commit, 100% coverage on push
uv run pytest --cov                 # 100% coverage required
uv run ruff check . && uv run mypy
```

To build the Docusaurus documentation site, generate its ignored API page and
then build from `website/`:

```bash
uv run python scripts/gen_api_docs.py
cd website && npm ci && npm run build
```

Pre-commit hooks and a blocking CI coverage gate keep `main` green. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the design docs in [`plans/`](plans/).

## Citation

If you use Inspect Robots in your research, please cite it:

```bibtex
@software{inspect-robots,
  author  = {Robocurve},
  title   = {Inspect Robots: The open-source evaluation framework for physical AI},
  year    = {2026},
  url     = {https://github.com/robocurve/inspect-robots},
  version = {0.3.0},
  license = {MIT}
}
```

## License

[MIT](LICENSE)
