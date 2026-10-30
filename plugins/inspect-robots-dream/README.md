# inspect-robots-dream

Run Dream Labs' [Dream-1](https://dreamlabs.ai/research/dream-1/) on a bimanual YAM as an
[Inspect Robots](https://github.com/robocurve/inspect-robots) policy, in simulation (`yam_sim`) or
on the real arms (`yam_arms`), and evaluate it like any other policy. Not affiliated with Dream Labs.

```bash
inspect-robots run --embodiment yam_sim --policy dream -E control_hz=20 \
  --instruction "put the red cube on the plate"
```

## How it works

Every control step, the `dream` policy hands Dream the three camera frames and the 14 joint values,
and plays Dream's 48-row chunks one row per step:

```text
embodiment ──frames + joints──▶ dream policy ──▶ Dream client ──▶ Dream-1 (hosted)
     ▲                                                               │
     └─────────────── one row per step ◀── 48-row chunk ◀────────────┘
```

- The arms hold still for the first 1.6 s, while Dream's window of nine frames per camera fills.
- The rollout waits while Dream answers, so a slow request pauses the arms rather than skipping rows,
  and each chunk takes over at its first row: the same seed plays the same trajectory however long
  Dream takes to answer.
- Requests use Dream's YAM station keys, as Dream's own simulated-YAM example does, so Dream computes
  each hand's pose in the frame the model was trained in.
- Rows play at the embodiment's control rate. Dream-1's rows are 30 Hz actions, and Dream's YAM
  station plays them at 20 Hz, so run the embodiment at 20 Hz.
- Each request passes the last row played, so Dream's seam smoothing blends into the new chunk from
  where the arms are headed.

## Install

Put your API key from [platform.dreamlabs.ai](https://platform.dreamlabs.ai/api-keys) where the
Dream client looks for it:

```bash
mkdir -p ~/.dreamlabs && echo 'DREAM_API_KEY=dl_live_...' > ~/.dreamlabs/credentials
chmod 600 ~/.dreamlabs/credentials
```

Install the policy with the Dream client:

```bash
pip install "inspect-robots-dream[client]"
```

Under Isaac Sim, which pins websockets 12 while the Dream client needs 13 or newer, install the
policy alone next to Isaac Sim and the client in its own venv:

```bash
uv pip install --python /data/env_isaaclab/bin/python inspect-robots-dream
uv venv --python 3.11 ~/dream-venv
uv pip install --python ~/dream-venv/bin/python "inspect-robots-dream[client]"
```

## Run

```bash
inspect-robots run --embodiment yam_sim --policy dream \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put the red cube on the plate" \
  --max-steps 1600 --store-frames --log-dir ~/yam-eval/dream
inspect-robots view ~/yam-eval/dream
```

Under Isaac Sim, start the bridge first and point the policy at it:

```bash
~/dream-venv/bin/inspect-robots-dream-bridge --warm
inspect-robots run --embodiment yam_sim --policy dream -P server_url=http://127.0.0.1:8210 ...
```

`--warm` asks for one chunk before serving, because a model that is not running can take many
minutes to start. The bridge listens on loopback and has no authentication. It serves one run at a
time: if a second run starts on the same bridge, the first one stops with an error rather than
sharing Dream's frame window.

Before the embodiment starts, the policy checks that Dream can serve the model: a missing client, a
bridge that is down, a refused key or an unknown model stops the run there.

To check Dream's replies before they move real arms, add `-P dry_run=true`: Dream is asked as usual,
the arms hold still, and each request is logged in the policy transcript.

## Options

| `-P` option | Default | Meaning |
|---|---|---|
| `model` | `dreamlabs/dream-1-beta` | Checkpoint; `dreamlabs/dream-1-beta-flash` is the faster one |
| `server_url` | none | Bridge URL; without it the Dream client runs in-process |
| `rows_per_request` | `48` | Rows of each chunk played before asking again (1 to 48) |
| `preroll_steps` | 32 at 20 Hz | Steps that hold still while the frame window fills |
| `dry_run` | `false` | Ask Dream but hold the arms still |
| `record` | `false` | Let the Dream client upload ticks to your Dream account |
| `timeout_s` | `300` | Longest wait for one chunk, retries included |

The policy accepts any embodiment with the bimanual YAM layout: 14 absolute joint targets
(`left_j0` to `left_j5`, `left_gripper`, then the right arm), cameras `top_cam`, `left_cam` and
`right_cam`, and state `joint_pos`. It warns when the control rate or camera shape differs from
what Dream expects. Each trial's transcript lists every Dream request (round trip, inference time,
retries, rows beyond the joint limits) and how closely the arms tracked the rows they were sent.

## Status

Dream-1 beta was run on `yam_sim` (Isaac Sim 5.1, L40S) on 2026-10-30 with a pre-release build that
used the same request protocol. An 80 s trial made 33 requests of about 0.4 s each with no errors,
and the arms tracked Dream's rows to within 0.03 rad. Dream reached the cube with the right arm and
kept trying to grasp it, but did not lift it. `yam_sim`'s overhead camera sits lower and looks less
steeply than the station camera Dream was trained with, which may matter.

## Development

From the repository root:

```bash
uv sync --all-packages --extra dev
uv run pytest plugins/inspect-robots-dream/tests --cov=inspect_robots_dream
```

The tests run offline at 100% coverage: fakes stand in for Dream, a real loopback bridge carries
bridged episodes, and the real Dream client runs with only its HTTP transport stubbed. The design
is in [plan 0087](../../plans/0087-dream-policy-plugin.md).
