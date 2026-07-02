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
