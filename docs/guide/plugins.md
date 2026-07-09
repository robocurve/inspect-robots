# Plugins & the registry

Inspect Robots components register by name and resolve from strings — the mechanism the
CLI and `eval("...", "...", "...")` use. In-tree builtins register via decorators;
out-of-tree packages publish **entry points**, so an installed plugin appears in
`inspect-robots list` without being imported first.

## Decorators

```python
from inspect_robots.registry import embodiment, policy, scorer, task

@policy("my-vla")
class MyVLA: ...

@embodiment("my-arm")
class MyArm: ...

@scorer("smooth")
def smooth(): ...

@task("my-bench")
def my_bench(): ...
```

## Resolving

```python
from inspect_robots.registry import registered, resolve

registered("policy")          # {"scripted": ..., "random": ..., "my-vla": ...}
policy = resolve("policy", "my-vla", checkpoint="...")   # constructor kwargs forwarded
```

## Shipping an out-of-tree plugin

Publish entry points from your package's `pyproject.toml`:

```toml
[project.entry-points."inspect_robots.embodiments"]
maniskill = "inspect_robots_maniskill:ManiSkillEmbodiment"

[project.entry-points."inspect_robots.policies"]
openvla = "inspect_robots_openvla:OpenVLAPolicy"
```

Groups: `inspect_robots.tasks`, `inspect_robots.policies`, `inspect_robots.embodiments`,
`inspect_robots.scorers`, `inspect_robots.sinks`. After `pip install inspect-robots-maniskill`, it
shows up in `inspect-robots list` and resolves by name in `eval()` and the CLI.

This is how the ecosystem stays decoupled: this repository is the **framework**;
specific simulators, VLA weights, and benchmarks live in their own packages.

## First-party plugins

Two adapters ship from the Inspect Robots repo itself (as separate packages
under `plugins/*`), covering both halves of an eval:

### `inspect-robots-isaacsim` — the body

Wraps an [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) simulation as an
embodiment. Installing it makes `isaacsim` resolvable; only `reset()`/`step()`
need a working Isaac install (listing and compatibility checks run anywhere):

```bash
pip install inspect-robots-isaacsim
inspect-robots run --task my-task --policy my-vla --embodiment isaacsim \
    -E task_id=Isaac-Lift-Cube-Franka-v0
```

### `inspect-robots-xpolicylab` — the brain

Drives any [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)-served
policy. XPolicyLab wraps 40+ VLA / imitation-learning policies (π0/π0.5,
GR00T, OpenVLA-OFT, RDT-1B, SmolVLA, ACT, …) behind one websocket
policy-server contract; this adapter speaks that protocol directly, so the
whole zoo becomes evaluable without installing any model dependencies locally:

```bash
pip install inspect-robots-xpolicylab

# terminal 1 — serve a policy from an XPolicyLab checkout (its own env/machine)
cd XPolicyLab/policy/Pi_0 && bash setup_eval_policy_server.sh ... 19000 0.0.0.0

# terminal 2 — evaluate it
inspect-robots run --task my-task --policy xpolicylab --embodiment isaacsim \
    -P url=ws://gpu-box:19000 -P cameras=cam_head:base_rgb
```

See each plugin's README (`plugins/inspect-robots-isaacsim`,
`plugins/inspect-robots-xpolicylab`) for the full configuration reference.
