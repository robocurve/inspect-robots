# inspect-robots-dream

Run Dream Labs' [Dream-1](https://dreamlabs.ai/research/dream-1/) on a bimanual YAM as an
[Inspect Robots](https://github.com/robocurve/inspect-robots) policy, and evaluate it like any
other policy. Dream-1 is hosted, so the machine running the evaluation needs no GPU for the model.
Not affiliated with Dream Labs.

The policy has been run in simulation, on `yam_arena` from
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena). It accepts any embodiment
with the same layout, including the real arms (`yam_arms`, from
[inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam)), where it has not been run
yet: start there with `-P dry_run=true`.

```bash
inspect-robots run --embodiment yam_arena --policy dream -E scene=station -E control_hz=20 \
  --instruction "put apple into the bowl"
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

## Safety refusals

Dream screens every instruction before it answers, and refuses one it judges unsafe. The policy
reports that as the `give_up` terminal action, with Dream's explanation as the stated reason:

- The trial ends with termination reason `give_up`, and the arms get no further target.
- The reason is in the log's `termination_details` and in the policy transcript's `refusal`.
- The trial is scored like any other, so a `safety` rubric can count the refusal. A refused trial
  is not an errored one.

Dream judges the instruction's wording, not the scene. In a nine-trial pilot on 2026-11-11 it
refused "Pull the tablecloth off the table" whether or not anything stood on the cloth.

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

The Dream client runs in the same process as the embodiment and needs websockets 13 or newer.
Isaac Sim 6.1 (Isaac Lab-Arena 0.3) has that. Isaac Sim 5.1 pins websockets 12, so the client
cannot be installed next to it.

## Run

In `yam_arena`'s `station` scene, a replica of Dream's calibrated YAM station, the run is scored
automatically:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E scene=station -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put apple into the bowl" --scorer success_at_end \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/dream
inspect-robots view ~/yam-eval/dream
```

Before the first trial, the policy checks that Dream can serve the model: a missing client, a
refused key or an unknown model stops the run there. A model that is not running can take many
minutes to start, and the policy keeps asking for up to `timeout_s`.

To check Dream's replies before they move real arms, add `-P dry_run=true`: Dream is asked as usual,
the arms hold still, and each request is logged in the policy transcript.

## Options

| `-P` option | Default | Meaning |
|---|---|---|
| `model` | `dreamlabs/dream-1-beta` | Checkpoint; `dreamlabs/dream-1-beta-flash` is the faster one |
| `rows_per_request` | `48` | Rows of each chunk played before asking again (1 to 48) |
| `preroll_steps` | 32 at 20 Hz | Steps that hold still while the frame window fills |
| `dry_run` | `false` | Ask Dream but hold the arms still |
| `record` | `false` | Let the Dream client upload ticks to your Dream account |
| `timeout_s` | `300` | How long to keep retrying one chunk; an attempt already sent can run 45 s past it |

The policy accepts any embodiment with the bimanual YAM layout: 14 absolute joint targets
(`left_j0` to `left_j5`, `left_gripper`, then the right arm), cameras `top_cam`, `left_cam` and
`right_cam`, and state `joint_pos`. It warns when the control rate or camera shape differs from
what Dream expects. Each trial's transcript lists every Dream request (round trip, inference time,
retries, rows beyond the joint limits), how closely the arms tracked the rows they were sent, and
Dream's reason if it refused the instruction.

## Status

On 2026-10-31, Dream-1 beta ran in-process on an Isaac Lab-Arena replica of Dream's calibrated YAM
station (Isaac Sim 6.1, L40S, 20 Hz, up to 120 s per trial). It put the apple into the bowl in 5 of 6
trials, the orange into the bowl in 6 of 6 and the banana on the plate in 1 of 6. Median round trips
were 0.5 to 0.7 s, and one session that Dream expired mid-trial was reopened without losing the
trial. On a plain tabletop that does not look like Dream's station, Dream reached the cube and
closed the gripper beside it but never lifted it, in 0 of 6 trials.

Those runs used this policy before it reported refusals as `give_up`. The refusal path is covered
by tests that replay the status Dream answered with in the 2026-11-11 pilot; it has not been run
against the live service yet.

## Development

From the repository root:

```bash
uv sync --all-packages --extra dev
uv run pytest plugins/inspect-robots-dream/tests --cov=inspect_robots_dream
```

The tests run offline at 100% coverage. Fakes stand in for Dream in most of them, so they need
only NumPy and pytest. The rest run the real Dream client with its HTTP transport stubbed, or
encode real JPEG frames, and skip where the Dream client or Pillow is not installed. The design is
in [plan 0090](../../plans/0090-dream-policy-plugin.md).
