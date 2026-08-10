# inspect-robots-wandb

Weights & Biases (W&B) logging sink plugin for **Inspect Robots**.

Streams evaluation runs, trial metrics, step counts, duration, and success rates directly to W&B dashboards.

## Installation

```bash
pip install "inspect-robots-wandb[wandb]"
```

## Usage

```python
from inspect_robots import eval
from inspect_robots_wandb import WandbSink

sink = WandbSink(project="my-vla-evals")
eval("cube_pick", "scripted", "cubepick", sinks=[sink])
```
