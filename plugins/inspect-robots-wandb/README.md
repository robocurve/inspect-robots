# inspect-robots-wandb

The `inspect-robots-wandb` plugin logs Inspect Robots evaluation summaries to
[Weights & Biases](https://wandb.ai/).

## Install

```bash
pip install inspect-robots-wandb
wandb login
```

The plugin registers the `wandb` sink in the `inspect_robots.sinks` entry-point
group.

## Use from Python

Passing `sinks=` replaces the default JSON sink, so include both sinks when you
want the local immutable log and the W&B summary:

```python
from inspect_robots import eval
from inspect_robots.logging import JsonLogSink
from inspect_robots_wandb import WandbSink

logs = eval(
    "cubepick-reach",
    "scripted",
    "cubepick",
    sinks=[
        JsonLogSink("logs"),
        WandbSink(project="robot-evals", tags=("cubepick", "nightly")),
    ],
)
```

`WandbSink` creates one W&B run for each evaluation lifecycle. It stores the
evaluation specification as run config, then logs final status, scene and trial
counts, errored trials, total steps, duration, and every aggregate result metric.
The same sink instance can be reused by `eval_set()`; each task receives its own
W&B run.

Use `mode="offline"` to write a local W&B run for later synchronization, or
`mode="disabled"` to exercise the integration without recording data.

## Configuration

| Argument | Default | Meaning |
| --- | --- | --- |
| `project` | `inspect-robots` | W&B project name. |
| `entity` | unset | W&B team or user name. |
| `name` | unset | Display name for the run. |
| `group` | unset | W&B run group. |
| `tags` | unset | Sequence of W&B tags. |
| `mode` | unset | W&B mode such as `online`, `offline`, or `disabled`. |
| `dir` | unset | Local directory used by W&B. |
