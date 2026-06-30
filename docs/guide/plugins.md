# Plugins & the registry

RoboInspect components register by name and resolve from strings — the mechanism the
CLI and `eval("...", "...", "...")` use. In-tree builtins register via decorators;
out-of-tree packages publish **entry points**, so an installed plugin appears in
`roboinspect list` without being imported first.

## Decorators

```python
from roboinspect.registry import embodiment, policy, scorer, task

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
from roboinspect.registry import registered, resolve

registered("policy")          # {"scripted": ..., "random": ..., "my-vla": ...}
policy = resolve("policy", "my-vla", checkpoint="...")   # constructor kwargs forwarded
```

## Shipping an out-of-tree plugin

Publish entry points from your package's `pyproject.toml`:

```toml
[project.entry-points."roboinspect.embodiments"]
maniskill = "roboinspect_maniskill:ManiSkillEmbodiment"

[project.entry-points."roboinspect.policies"]
openvla = "roboinspect_openvla:OpenVLAPolicy"
```

Groups: `roboinspect.tasks`, `roboinspect.policies`, `roboinspect.embodiments`,
`roboinspect.scorers`, `roboinspect.sinks`. After `pip install roboinspect-maniskill`, it
shows up in `roboinspect list` and resolves by name in `eval()` and the CLI.

This is how the ecosystem stays decoupled: this repository is the **framework**;
specific simulators, VLA weights, and benchmarks live in their own packages.
