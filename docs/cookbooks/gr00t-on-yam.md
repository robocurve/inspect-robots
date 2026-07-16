# Evaluate a GR00T fine-tune on real YAM arms

This cookbook takes you from a GR00T checkpoint on the Hugging Face Hub to a
scored evaluation on real [I2RT YAM](https://i2rt.com/products/yam-6-dof-arm)
bimanual arms, with every step before first motion verifiable off-robot. The
worked example is
[robocurve/gr00t-n1.7-yam-molmoact2](https://huggingface.co/robocurve/gr00t-n1.7-yam-molmoact2),
a GR00T N1.7 fine-tune on the AllenAI MolmoAct2 bimanual-YAM dataset; the same
recipe applies to any GR00T YAM fine-tune that keeps the 14-D joint contract.

Three processes cooperate:

| Process | Where | Role |
|---|---|---|
| GR00T `/act` server | GPU machine | Owns the weights; turns observations into action chunks |
| `inspect-robots` CLI | Rig host (often the same machine) | Runs the eval loop, scoring, logging, Rerun |
| `yam_arms` embodiment | Rig host | Drives the arms over CAN, reads the three cameras |

The policy and the embodiment are decoupled by design: the `gr00t` policy from
[inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam) is a thin
HTTP client for the `/act` protocol, and the serving shim adapts an
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) checkpoint to that wire
format. Compatibility (action dims, semantics, cameras, state keys) is checked
before any motion.

## Prerequisites

- A CUDA GPU with roughly 8 GB of free VRAM for the 3B checkpoint (measured:
  7.1 GiB during inference on an RTX 5090). Blackwell GPUs (`sm_120`) work
  with the PyTorch build Isaac-GR00T pins (torch 2.7.1 cu128 wheels).
- A Hugging Face account with access to
  [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
  GR00T N1.7 loads its vision-language backbone from that gated repository
  even when the fine-tune itself is public, so accept the license there once
  (approval is automatic).
- The YAM rig set up per the
  [inspect-robots-yam README](https://github.com/robocurve/inspect-robots-yam#run-on-hardware):
  CAN channels, three cameras, `config.ini` written by `inspect-robots setup`.
- `inspect-robots-yam` 0.13.0 or newer (the release that ships the `gr00t`
  policy and the serving shim).

## 1. GPU side: environment and checkpoint

Clone Isaac-GR00T and let `uv sync` build the environment. The pinned
`flash-attn` comes as a prebuilt wheel (cu12, torch 2.7), so there is no
source build:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
uv sync
source .venv/bin/activate
```

Add the shim's serving dependencies (they are not part of Isaac-GR00T), log
in to Hugging Face (required for the gated backbone, see prerequisites), and
download the checkpoint (about 7 GB):

```bash
uv pip install fastapi uvicorn json_numpy
hf auth login
hf download robocurve/gr00t-n1.7-yam-molmoact2
```

## 2. Start the `/act` server

The shim ships with the inspect-robots-yam repository (it is a script, not
part of the installed package, because it needs the Isaac-GR00T environment):

```bash
curl -LO https://raw.githubusercontent.com/robocurve/inspect-robots-yam/main/scripts/serve_gr00t_act.py
python serve_gr00t_act.py --model robocurve/gr00t-n1.7-yam-molmoact2
```

The default port is 8203, so a MolmoAct2 server on 8202 can keep running next
to it. Startup is the first safety gate: the shim hard-fails unless the
checkpoint's state and action keys exactly match the packed 14-D YAM layout
(`left_arm`, `left_gripper`, `right_arm`, `right_gripper`), the per-key widths
from `dataset_statistics.json` match their slices, arm statistics stay within
radians range, and gripper statistics stay within normalized range. A
checkpoint trained in degrees, with swapped or missing keys, or with frame
history is rejected before it can ever move metal.

Two things startup checks cannot prove: joint polarity and
absolute-versus-delta action semantics. Those stay human checks (section 5).

Health check from any machine:

```bash
curl http://gpu-host:8203/act
# {"status": "ok", "model": "robocurve/gr00t-n1.7-yam-molmoact2"}
```

## 3. Rig side: install the plugin

On the rig host, in your inspect-robots environment:

```bash
uv pip install "inspect-robots-yam>=0.13"
inspect-robots list policies
# ... gr00t ... molmoact2 ...
```

The `gr00t` policy defaults to `http://127.0.0.1:8203`. If the GPU box is a
different machine, point at it per run with `-P server_url=http://gpu-host:8203`
or persist it in `config.ini` under `[policy.args]`.

## 4. Prove the loop off-robot

Run one inference through the exact client the eval will use, with a synthetic
observation, before any hardware is involved:

```python
import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_yam import gr00t_policy

policy = gr00t_policy()  # or gr00t_policy(server_url="http://gpu-host:8203")
policy.reset(Scene(id="smoke", instruction="stack the red block on the blue block"))
img = np.zeros((480, 640, 3), dtype=np.uint8)
obs = Observation(
    images={"top_cam": img, "left_cam": img, "right_cam": img},
    state={"joint_pos": np.zeros(14)},
)
chunk = policy.act(obs)
print(len(chunk), chunk.actions[0].data.round(3))
```

Expect a 16-step chunk of plausible joint radians. The client rejects
non-finite values and wrong shapes loudly, so a clean print here means the
wire format, camera mapping, and action packing all hold. For scale: on an
RTX 5090 the first request takes about 1.2 s (warmup) and warm requests
about 90 ms per 16-step chunk.

Then run the plugin's preflight, which checks the full
`(policy, embodiment)` pair without motion:

```bash
inspect-robots-yam-preflight --policy gr00t
inspect-robots-yam-preflight --policy gr00t --dry-run   # affirms no motion will occur
```

## 5. First motion, human at the e-stop

!!! warning
    First runs with a new checkpoint family are hardware-verification runs,
    not evals. Keep a hand on the e-stop, leave guardrails on (they are on by
    default; disabling requires an explicit `--disable-guardrails`), and clear
    the workspace.

Verify the arms hold position while the policy computes (seconds can pass
between action chunks):

```bash
inspect-robots-yam-holdcheck
```

Then run a short, simple instruction from the checkpoint's trained task
families. For the worked example those are block manipulation, box packing,
and cable charging; VLA policies do not transfer to untrained task families,
so pick accordingly:

```bash
inspect-robots "stack the red block on the blue block" --policy gr00t --max-steps 300
```

Watch the first chunk in the Rerun viewer before the arms move far: joint
directions that mirror the intended motion indicate a polarity mismatch, and
runaway drift from the start indicates delta actions being interpreted as
absolute (or vice versa). Stop immediately in either case; both are
checkpoint-contract problems, not tuning problems.

## 6. Scored evaluations

Once the rig behaves, evals are ordinary Inspect Robots runs. With
[KitchenBench](https://github.com/robocurve/kitchenbench) installed:

```bash
inspect-robots run --task kitchenbench/pour_pasta --policy gr00t --embodiment yam_arms
```

At each episode end the operator answers y/N; the score lands in the
`EvalLog` together with the resolved config, so a GR00T run is labeled
`gr00t` (not `molmoact2`) in every log:

```bash
inspect-robots inspect logs/pour_pasta_*.json
```

For a different GR00T fine-tune, pass `-P action_horizon=<its chunk length>`
so the recorded metadata matches that checkpoint (the rollout itself always
uses the returned chunk length; this field is log metadata).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv sync` fails with "Invalid zip file structure" on a flash-attn or torchcodec wheel | The Isaac-GR00T checkout vendors aarch64 wheels via git-lfs; without lfs they are pointer files | `git lfs install --local && git lfs pull --include "scripts/deployment/dgpu/wheels/*"`, then re-run `uv sync` |
| Shim exits with `401 ... gated repo` for `nvidia/Cosmos-Reason2-2B` | GPU host is not logged in to Hugging Face, or the account has not accepted the backbone license | `hf auth login`, accept the license on the [model page](https://huggingface.co/nvidia/Cosmos-Reason2-2B), restart the shim |
| Shim exits with "keys must equal" or width errors | Checkpoint does not follow the bimanual-YAM layout | Wrong checkpoint for this rig; do not bypass the check |
| Shim exits with radians/normalized range errors | Checkpoint trained in degrees or unnormalized grippers | Same: incompatible checkpoint |
| `no policy named 'gr00t'` | Plugin older than 0.13.0, or `uv run` re-synced the venv and removed it | `uv pip install -U inspect-robots-yam`, and invoke plain `inspect-robots`, not `uv run inspect-robots` |
| Client `ValueError: /act returned non-finite actions` | Server-side numerical problem | Check the shim log; restart the server; if reproducible, file an issue with the request |
| CUDA errors on RTX 50-series | PyTorch without `sm_120` support | Use Isaac-GR00T's pinned torch 2.7.1 (cu128); avoid overriding it |
| Slow first inference | CUDA graph and cache warmup | Expected; time the second request |
