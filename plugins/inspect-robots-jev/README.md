<div align="center">

# inspect-robots-jev

TypeSafe's **Jev** structured-decision model as an
[Inspect Robots](https://github.com/robocurve/inspect-robots) policy.

</div>

Jev reads text and picks one option from a list you give it. It cannot see
images and cannot write free text or numbers. This plugin turns a robot scene
into text Jev can act on: AprilTags on the objects give their positions, the
plugin describes where the target is relative to the gripper in plain words
("3.5 cm FORWARD, 4 cm BELOW"), offers a menu of small moves with a hint on
each ("use when the cube is 2 to 5 cm away"), and executes whichever one Jev
picks. Code owns the task phases (approach, descend, grasp, lift, carry,
lower, release, retreat, verify); Jev steers inside each phase. Descents also
end on contact (the measured height stops dropping), and the trial only ends
once the released cube's tag has been seen again from above. Registered as the
policy `jev` and the scorer `jev_cube_in_bowl`.

Design, measurements, and the decisions behind them:
[`plans/0076-jev-decision-policy.md`](../../plans/0076-jev-decision-policy.md).

## Install

```bash
pip install inspect-robots inspect-robots-jev
```

Jev is reached through OpenRouter's Decisions endpoint (default, needs
`OPENROUTER_API_KEY`) or TypeSafe directly (`-P api=typesafe`, needs
`TYPESAFE_API_KEY`). Keys are read from the environment or a `.env` file in
the working directory. Every request is one `state` plus one Choice
question; the answer's full probability table and confidence land in the
transcript (`inspect-robots inspect --transcript`) and the HTML report.

## Quickstart (no hardware)

The plugin's end-to-end test drives a toy Cartesian rig through the whole
task with a scripted Jev. Run it to see the loop work:

```bash
uv run --no-sync python -m pytest plugins/inspect-robots-jev/tests/test_end_to_end.py -q
```

## Setting up a rig

### 1. Print the tags

Family `tag36h11`. A tag is an 8 x 8 black square with a white margin of at
least one cell all round (10 x 10 cells total). The size that matters is the
**black square's edge**, so print at 100 % and measure it.

| Where | Black square | Footprint incl. margin | IDs |
|---|---|---|---|
| cube, one per face except the bottom | 20 mm (22 mm on a 28 mm cube) | 25 mm (27.5 mm) | 0-4 |
| bowl, flat outer sides or rim, plus the inside bottom | 40 mm | 50 mm | 10-14 |
| table reference tag for calibration | 80 mm | 100 mm | 20 |

Glue each tag flat on rigid backing. Fix the table tag near the front edge
of the workspace, inside the top camera's view, where both grippers can
touch its corners. Mark its printed **top-left** corner with a pen.

### 2. Describe the tags

`tags.json` lists every tag's object, black-square size in metres, and the
object centre in the tag's own frame. AprilTag's tag frame is x right, y
down, **z into the tag away from the camera**, so a face tag on a 25.4 mm
cube has the centre at `+0.0127` along z:

```json
{
  "table_tag_id": 20,
  "tags": [
    {"id": 0, "object": "cube", "size_m": 0.020, "offset_m": [0, 0, 0.0127]},
    {"id": 1, "object": "cube", "size_m": 0.020, "offset_m": [0, 0, 0.0127]},
    {"id": 14, "object": "bowl", "size_m": 0.040, "offset_m": [0, 0, 0]},
    {"id": 20, "object": "table", "size_m": 0.080, "offset_m": [0, 0, 0]}
  ]
}
```

Several tags on one object are averaged as a rigid body. From a top-down
camera the side faces are edge-on, so once the gripper covers the top tag the
cube is effectively unseen; the policy tolerates that in `descend` (the goal
is the remembered pose) and pins the cube to the gripper while it is held.

### 3. Calibrate the camera once

The policy needs to know where the camera is relative to each arm's base.
Instead of measuring by hand, touch the table tag's four corners with each
gripper tip and read the grasp-point position the embodiment reports
(`inspect-robots-yam`'s pose CLI prints it). Order: the printed tag's
top-left, top-right, bottom-right, bottom-left. Save them as

```json
{"left":  [[x, y, z], [x, y, z], [x, y, z], [x, y, z]],
 "right": [[x, y, z], [x, y, z], [x, y, z], [x, y, z]]}
```

Grab one top-camera frame and its intrinsics as `.npy` files. From Python,
with an observation from the embodiment:

```python
from pathlib import Path
from inspect_robots_jev.calibrate import observation_to_npy
frame, intr = observation_to_npy(observation, "top_cam", Path("~/jev/cap").expanduser())
```

Then solve and save:

```bash
python -m inspect_robots_jev.calibrate --corners corners.json \
  --frame $HOME/jev/cap/top_cam_frame.npy --intrinsics $HOME/jev/cap/top_cam_intrinsics.npy \
  --tags tags.json --out calibration.json
```

It prints the camera's position in each arm frame; sanity-check the height.

### 4. Running on the YAM rig

The tags need every pixel. Both YAM camera paths capture at 640 x 480 by
default, which makes a 20 mm tag about 10 px wide from a 1.2 m mount, so Jev
runs need the RealSense path (intrinsics and depth) at full capture
resolution. In `~/.config/inspect-robots/config.ini`:

```ini
[embodiment.args]
control_interface = eef_pos
top_depth_serial = <D435 serial>      # hands the top slot to librealsense: intrinsics + depth
capture_width = 1920                  # requires inspect-robots-yam with configurable capture size
capture_height = 1080
cam_width = 1920                      # no downscale
cam_height = 1080
gripper_max_step = 0.25               # a full close/open in 4 ticks instead of 20
```

Then:

```bash
inspect-robots "put the cube in the bowl" --policy jev \
  -P tags=$HOME/jev/tags.json -P calibration=$HOME/jev/calibration.json \
  --embodiment yam --scorer jev_cube_in_bowl --max-steps 600
```

`--max-steps` counts control ticks, not Jev decisions: one pick becomes 1 to
4 ticks of motion, and a nominal run is about 150 ticks.

## Options (`-P key=value`)

| Key | Default | Meaning |
|---|---|---|
| `api` | `openrouter` | `openrouter` (Decisions alpha endpoint) or `typesafe` (direct) |
| `model` | per backend | `typesafe/jev-1.13` on OpenRouter, `jev-latest` direct |
| `steps_cm` | `0.5,2,5` | menu step sizes; each option says when to use it |
| `camera` | `top_cam` | observation image key of the fixed camera |
| `tags` | `tags.json` | tag layout file |
| `calibration` | `calibration.json` | camera-to-arm transforms |
| `history` | `5` | recent moves shown to Jev (it has no memory of its own) |
| `stale_after` | `10` | in `approach`/`carry`, hold and log a stall when the target tag has been unseen for this many Jev decisions (not control ticks) |
| `cube`, `bowl` | `cube`, `bowl` | object names as they appear in `tags.json` |

## What the state looks like

```json
{
  "task": "Move the left gripper to the target point 5 cm above the cube. Move toward that point, never away.",
  "target_point_relative_to_left_gripper": ["3.5 cm FORWARD", "aligned on y", "4 cm BELOW"],
  "straight_line_distance_cm": 5.3,
  "left_gripper": "open",
  "other_objects_on_table_ignore_them": ["bowl"],
  "recent_moves_oldest_first": ["move the left gripper FORWARD by 5 cm"]
}
```

Only the current target's geometry is present; other objects are names. The
throwaway probe that fixed this design scored 30/30 with this wording versus
12/30 with signed numbers and 6/30 when every object's geometry was listed.
`examples/jev_textsim.py` reruns that probe against the live API; run it
before changing any wording.

## Scoring

`jev_cube_in_bowl` reads the last recorded action's `meta["jev"]["world"]`
and succeeds when the cube's tag ended within 4 cm horizontally of the bowl
tag and no more than 3 cm above it, **and** the cube was actually re-detected
within the last 3 decisions after release (a cube stuck in the jaws, or a pose
inferred from the gripper, is not credited).
