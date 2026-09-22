# Jev decision policy: a text-only harness for the robot (design spec)

**Status:** design reviewed 2026-09-19; open questions answered (§9). The
step-by-step build list gets added next (in this repo, the plan file is the
spec).

**The task for version 1:** one arm picks up a cube (about 1 inch on a
side) and places it in a bowl (about 4 inches across). Target selection is
done by code in this version so we measure Jev's steering on its own first;
letting Jev pick the target with a second call is the planned follow-up
(§10).

## 0. Terms used in this document

Read these first. Each term is used in the sense given here.

- **Policy** — the "brain" in an eval: the thing that looks at the world and
  decides what the robot should do next. Usually a vision model; here, Jev.
- **Embodiment** — the "body": the code that talks to the real arms and
  cameras. Ours is the bimanual YAM adapter (`inspect-robots-yam`).
- **Observation** — what the embodiment hands the policy each step: camera
  images, the arms' current positions, and the task text.
- **Action / ActionChunk** — what the policy hands back: one or more small
  motion commands, played one per control tick.
- **Rollout / trial** — one attempt at the task, from reset to done or
  timeout. **`max_steps`** caps the number of *control ticks* (one per
  action played), not Jev decisions: one Jev pick becomes several ticks (a
  5 cm move is 4 ticks, a full gripper close about 20 at the default 5 %
  per-tick limit), so a nominal cube-into-bowl run needs roughly 150 ticks
  and budgets below are set to 600.
- **Scorer** — code that decides whether a trial succeeded and writes the
  score into the log.
- **Guardrails** — safety filters that already sit between the policy and
  the arms. They clamp commands to allowed ranges and cap how far anything
  may move in one tick. This design does not change them.
- **Plugin** — a separate installable package that adds a policy,
  embodiment, or scorer to the framework. The core package stays free of
  heavy dependencies; anything needing a camera library goes in a plugin.
- **Jev** — TypeSafe's "structured decision model". It reads text and picks
  from options you list. It cannot see images and cannot write free text or
  numbers. Details in §1.
- **State** (Jev's word) — the text or JSON you send Jev describing the
  situation. **Question** — one thing you ask about that state. Three kinds:
  **Choice** (pick one from a list you give, up to 255 entries), **Score**
  (place the situation on a scale of 2 to 10 levels you describe), **Noul**
  (yes or no, returned as a probability). Every answer also carries a
  **confidence** number from 0 to 1.
- **OpenRouter** — a service that fronts many models behind one API key.
  It offers Jev through a special "Decisions" endpoint, not the usual chat
  endpoint.
- **AprilTag** — a black-and-white square marker, like a simplified QR
  code, printed and stuck on an object. A detector finds it in a camera
  image and reads its ID. Because the tag's real size is known, the
  detector can also compute how far away it is and how it is turned.
  "36h11" is the standard tag family we will use.
- **Pose** — an object's position (x, y, z) plus its rotation.
- **Camera intrinsics (K)** — the numbers describing a camera's lens
  (focal length, image centre). Needed to turn pixels into metres. The YAM
  camera reader already provides them.
- **Extrinsics / calibration** — the fixed relationship between where the
  camera is and where the robot base is. Found once by looking at a
  reference tag whose position relative to the robot is measured.
- **Depth image** — a second image from the RealSense camera giving
  distance per pixel. Optional here; it sharpens the distance estimate.
- **Grasp point** — the spot between the gripper fingertips. The YAM
  embodiment reports it directly in metres, so we never have to compute it
  from joint angles.
- **Phase machine** — a small piece of ordinary code that tracks which stage
  of the task we are in (approach, grasp, carry, release) and moves between
  stages on simple rules.
- **Transcript** — the per-step record of what the policy saw and decided,
  saved with the eval log and shown in the HTML report.

## 1. What Jev is and how we reach it

| Fact | Value | Where it comes from |
|---|---|---|
| Model name on OpenRouter | `typesafe/jev-1.13` | OpenRouter model list |
| Endpoint via OpenRouter | `POST https://openrouter.ai/api/alpha/decisions`, using `OPENROUTER_API_KEY` | OpenRouter API reference |
| Endpoint direct from TypeSafe | `POST https://api.typesafe.ai/v1/systemone`, using `TYPESAFE_API_KEY`, model `jev-latest` | docs.typesafe.ai |
| Request body | `{"model", "state", "questions": {id: question}}`, same on both endpoints | both |
| Answer | one entry per question id: the pick, a probability for every option, and confidence | both |
| Input limits | text only; about 32k tokens of state | docs.typesafe.ai/models |
| Speed | about 0.2 s per call, measured from this machine | our test |
| Cost | $0.042 per million input tokens, output free. Our whole test run (about 6,000 calls) cost under one cent | our test |
| Memory | none. Each call is independent. If Jev needs to know what happened before, we must put it in the state | docs |

**The normal chat API does not work for Jev.** OpenRouter's page says it
"runs on the OpenRouter Decisions API rather than the OpenAI-compatible chat
endpoint". Our existing LLM policy plugin (`inspect-robots-agent`) speaks
the chat API, so its network code cannot be reused. We need a small new
client, written like the repo's existing `_chatwire.py` with only the
Python standard library.

## 2. What we learned from the throwaway test (2026-09-19)

We built a text-only simulation with no robot: a "gripper" as a point in a
40 × 40 × 30 cm box, a target at a random spot, and a menu of moves
(each axis, plus or minus, in 0.5 / 2 / 5 cm steps, plus "hold"). Each turn
we described the situation to the real Jev and applied whatever move it
picked. Success meant getting within 0.75 cm. 30 episodes per variant,
capped at 80 moves each.

| Variant | How the state was worded | Episodes solved | Moves that got closer | Back-and-forth reversals per episode |
|---|---|---|---|---|
| v1 | signed numbers ("x: +3.5, z: -4.0") plus a sentence explaining the sign convention | 12 of 30 | 47 % | 7.7 |
| v2 | words ("3.5 cm RIGHT, 4 cm BELOW") and a hint on every menu option saying when to use that step size | **30 of 30** | 98 % | 0.0 |
| v3 | two arms, one target, two other cubes; full geometry listed for every cube | 6 of 30 | 17 % | — |
| v3c | same as v3, but the target cube's block labelled "THE TARGET" | 7 of 30 | 15 % | — |
| v3b | geometry given only for the target; the other cubes just named | **30 of 30** | 83 % | — |

What went wrong in v1: Jev chose the right axis 91 % of the time but the
right *direction* along it only 19 % of the time, and it chose the 5 cm step
92 % of the time even when 1 cm away. In plain terms, it reads signed
numbers backwards and does not weigh magnitudes. In v3 it chose the closer
arm correctly 93–100 % of the time; the failure was steering toward the
wrong cube's numbers.

Smaller observations: "hold" was chosen about 1 % of the time. A yes/no
question "is the gripper already on the cube" answered 0.38 on average when
true and 0.01 when false, so it is a weak signal at best. Confidence did
not tell good moves from bad ones (99 % vs 96 % improvement in v2).

### Rules this gives us

- **R1. Code decides what Jev sees.** The state holds only the geometry
  needed for the current decision, described relative to the gripper being
  moved. Other objects appear by name only.
- **R2. Never ask Jev to do arithmetic.** Directions are words, distances
  are rounded to 0.5 cm, and each axis gets its own line.
- **R3. Every menu option says when to use it.** That is what fixed the
  step-size problem.
- **R4. Code decides task progress.** Jev picks the next small move. Code
  decides when the gripper is close enough to grasp, when the grasp worked,
  and when the task is done.
- **R5. Log confidence, don't act on it yet.** The test gave no evidence
  that pausing on low confidence helps. Keep "hold" in the menu and revisit
  once we have data from the real rig.

## 3. How the pieces fit

A new plugin, `plugins/inspect-robots-jev/`, that registers a policy called
`jev`. Nothing changes in the core, the YAM embodiment, the guardrails, the
rollout loop, or the report. Tag detection lives inside the policy because
the policy already receives the camera images, and the YAM camera reader
already supplies the intrinsics and depth in the observation's `extra`
field.

```
plugins/inspect-robots-jev/
  pyproject.toml                # deps: numpy, an AprilTag detector; optional depth extra
  src/inspect_robots_jev/
    __init__.py                 # registers the `jev` policy and the pose-based scorer
    _decisions.py               # the network client (OpenRouter or TypeSafe), testable without a network
    perceiver.py                # images + K (+ depth) -> where each tagged object is, in each arm's frame
    calibration.py              # find and store the camera-to-robot relationship from the table tag
    serializer.py               # turn the curated world state into Jev's `state` text, in words
    menu.py                     # build the Choice options for the current phase
    phase.py                    # the phase machine (approach / descend / grasp / lift / carry / release / done)
    policy.py                   # JevPolicy: wires the above together; keeps the transcript
    scorer.py                   # decides success from tag poses, no human or model judgement
    py.typed
  tests/                        # 100 % coverage; fake perceiver; recorded Jev answers
  README.md, CLAUDE.md
```

### 3.1 What happens on every step

1. The rollout loop calls `JevPolicy.act(observation)`.
2. **Look.** The perceiver finds the tags in the fixed camera's image, works
   out each tag's pose from its size and the camera intrinsics, converts
   that into each arm's coordinate frame using the stored calibration, and
   (if depth is available) tightens the distance using the depth image at
   the tag centre. Several tags on one object are merged so the object
   stays visible when one face is hidden. Objects not seen this step keep
   their last pose and a "last seen N moves ago" note. Gripper positions
   are read straight from the observation.
3. **Which stage are we in?** The phase machine checks simple facts (is the
   gripper above the cube, is it closed, is the cube lifted) and returns the
   current phase, for example "approach the red cube with the left arm".
4. **Write the state and the menu.** The serializer writes the state for
   this phase following R1 and R2. The menu builder lists the legal moves
   for this phase following R3: for an approach phase, six directions times
   three step sizes for the active arm, plus "hold"; for grasp or release,
   just two options.
5. **Ask Jev.** One request, one Choice question. About 0.2 s.
6. **Move.** The chosen option becomes a small position change for the
   active arm in metres, split into per-tick actions the same way the
   existing agent plugin splits its moves, or a gripper open/close. Returned
   as one ActionChunk. The chunk's metadata records Jev's pick, the whole
   probability table, the confidence, and the phase.
7. The guardrails check every tick as they do today.

### 3.1a The phases for cube-into-bowl

| Phase | Active arm does | Jev's menu | Code moves on when |
|---|---|---|---|
| approach | steer the open gripper to hover above the cube | 6 directions × 3 step sizes + hold | gripper within 1 cm of the point 5 cm above the cube |
| descend | lower onto the cube | UP/DOWN steps + hold | grasp point within 1 cm of cube centre height |
| grasp | close | close / hold | gripper reads closed on something (opening ≤ `closed_on_object` but > `closed_on_air`); closed on air (≤ `closed_on_air`) → `reopen` |
| reopen | open again after closing on nothing | open / hold | gripper reads open (≥ `open_threshold`); then back to `approach` |
| lift | raise | UP steps + hold | cube tag (or gripper) 8 cm above table |
| carry | steer to hover above the bowl centre | 6 directions × 3 step sizes + hold | within 1.5 cm above the bowl centre |
| lower | descend into the bowl | UP/DOWN steps + hold | grasp point below bowl rim height |
| release | open | open / hold | gripper reads open |
| retreat | rise clear | UP steps + hold | 8 cm above the bowl tag; then `verify` |
| verify | keep rising until the camera sees the cube again | UP/DOWN steps + hold | the cube's tag is re-detected this decision (not inferred from the gripper), or `verify_decisions` (8) elapse; then `done` |

Contact rule: `descend` and `lower` also complete when the gripper's measured
height has not dropped by more than `z_tol_m` over `contact_decisions` (3)
consecutive decisions, because the fingers or the carried cube are resting on
something (a bowl shallower than `bowl_depth_m`, the table under a low cube).
Without it the computed goal can be unreachable and the arm would press down
until the horizon. The policy also restarts its *position* interpolation from
the measured pose whenever the active arm's commanded and measured x/y/z
differ by more than 2 cm; gripper and orientation keep their commanded values
(a jaw stopped on the cube at 0.3 must stay commanded closed), so a blocked
arm is never driven from a phantom pose and a held grasp never relaxes. The
contact rule counts only decisions that commanded a DOWN move.

The arm is the one whose base is closer to the cube at reset. The
scorer marks success when the cube tag's final position is inside the
bowl's footprint and below its rim height, read from the tags.

### 3.2 Example state, approach phase

```json
{
  "task": "Move the left gripper onto the red cube. Move toward it, never away.",
  "red_cube_relative_to_left_gripper": ["3.5 cm FORWARD", "aligned on y", "4 cm BELOW"],
  "straight_line_distance_cm": 5.3,
  "other_objects_on_table_ignore_them": ["blue plate", "green cube"],
  "recent_moves_oldest_first": ["move the left gripper FORWARD by 5 cm",
                                "move the left gripper DOWN by 5 cm"]
}
```

Direction words follow the YAM arm's own frame: FORWARD/BACK along x,
LEFT/RIGHT along y, UP/DOWN along z, seen from the active arm's base. (The
test used different x/y words; the rig wording is what ships.)

### 3.3 Example menu, approach phase, left arm

19 options: FORWARD, BACK, LEFT, RIGHT, UP, DOWN, each in 0.5, 2 and 5 cm,
plus "hold". Each description reads like "move the left gripper FORWARD by
2 cm; use when the cube is 2 to 5 cm away in that direction". Step sizes
are configurable.

## 4. Decisions and why

- **D1. A plugin, not core code.** The tag detector is a new dependency and
  the core must stay NumPy-only.
- **D2. Perception inside the policy.** We considered wrapping the
  embodiment so it adds object positions to every observation instead.
  Rejected for now: it would need a new registry kind, and this policy is
  the only consumer. Revisit if a second text-only policy appears.
- **D3. Code runs the phases; Jev works inside a phase (R4).** Tests v3 and
  v3c show Jev cannot ignore geometry that does not matter, so it must not
  be shown the whole table and asked what to do. The agreed follow-up is a
  *second, separate* Jev call at the start of each phase that picks the
  target object from a list of names given the task sentence, with no
  geometry in that state. It must be a separate call because the move
  question needs geometry for the chosen target only, and Jev answers all
  questions in one request against the same state. It runs once per phase,
  not every step, so the target cannot flip back and forth. Version 1 ships
  without it to get a clean steering baseline.
- **D4. One Choice per request, with the arm baked into the option name.**
  Asking "which arm" and "which move" as two questions is worse: Jev
  answers each question independently and could pick an arm and a move that
  do not go together.
- **D5. Reuse the agent plugin's move-splitting code.** Import it if it is
  public; otherwise copy the roughly 40 lines with a comment, because the two
  plugins must install independently.
- **D6. Two backends, one client.** `-P api=openrouter` (default) or
  `-P api=typesafe`. Request and answer bodies are identical; only the URL,
  the header, and a few bookkeeping fields differ.
- **D7. Keep a full transcript.** Each step records the phase, the state
  sent, the question, and the full answer including every option's
  probability. This is the main scientific output and it will show up in
  `inspect --transcript` and the HTML report.
- **D8. Code decides "done".** The phase machine's `done` state ends the
  trial. The scorer checks poses. Neither Jev nor a human operator is asked.
- **D9. Hidden tags.** An object unseen for 1 to 10 steps keeps its last
  pose and the state says "last seen 3 moves ago". Unseen for more than 10
  steps (configurable, `-P stale_after=10`) during an approach or carry
  phase: code forces "hold" and logs a perception stall. While grasping and
  lifting, the held object is expected to be hidden, and its pose is taken
  as the gripper position plus the known grasp offset.

## 5. Perception details and physical setup

- **Camera.** The top camera is the fixed context camera; its mount is
  rigid, so calibrating once per run is fine. Today the rig opens it through
  the plain webcam path, which gives no lens intrinsics and no depth, and
  the policy receives 224 × 224 pixel images. Both are showstoppers for tag
  reading: a 20 mm tag would be about 4 pixels wide. Worse, *both* YAM camera
  paths capture at a fixed 640 × 480 and then resize to `cam_width ×
  cam_height` (`_capture_proc.py` `REALSENSE_CAPTURE_WIDTH/HEIGHT`; the
  OpenCV path sets 640 too), so a larger frame size only upsamples. From the
  measured 1.2 m camera height a 20 mm tag is about 10 px wide at 640, which
  no detector reads. Version 1 therefore needs a small **companion change in
  `inspect-robots-yam`** (its own PR): make the capture resolution
  configurable (`capture_width`, `capture_height`, defaults unchanged at
  640 × 480) on both paths, with intrinsics scaled from the capture size.
  Jev runs then set `top_depth_serial = <serial>` (RealSense path:
  intrinsics + depth), `capture_width = 1920`, `capture_height = 1080`,
  `cam_width = 1920`, `cam_height = 1080`. Non-square sizes are accepted
  today (`YamConfig` has no square check). Fallbacks if 1080p capture is too
  slow: 1280 × 720 with the camera lowered to about 0.8 m, or larger objects.
  The wrist cameras above each gripper move with the arms and are not used
  in version 1.
- **Tag family** 36h11, the default every detector supports. A 36h11 tag is
  a black square of 8 × 8 cells and needs a white margin of at least one
  cell all round, so its full footprint is 10 × 10 cells.
- **Tag sizes** (the "size" that matters is the black square's edge, and we
  must know it to within half a millimetre, so we print at exact scale and
  measure). Rule: footprint = the face it sits on, black square = 0.8 × that.

  | Where | Black square | Footprint incl. white | Count and IDs |
  |---|---|---|---|
  | cube, one per face except the bottom | 20 mm (22 mm if the cube measures 28 mm) | 25 mm (27.5 mm) | 5, IDs 0–4 |
  | bowl, on flat outer sides or rim, plus one on the inside bottom | 40 mm | 50 mm | up to 5, IDs 10–14 |
  | table reference tag for calibration | 80 mm | 100 mm | 1, ID 20 |

  Pixel check for the cube tags: a RealSense D435 colour stream at 1280 px
  wide has a focal length of roughly 930 px. A 20 mm tag 70 cm from the
  camera is then about 27 px wide, which is near the low end of reliable
  detection. If the camera sits higher than about 80 cm, or detection is
  flaky, the fallbacks are: 1920 × 1080 frames, mounting the camera lower,
  or using the wrist cameras during approach (version 2).
- **Several tags per object** are merged as one rigid body from a config
  file mapping tag ID → object name, black-square size, and the object
  centre's offset in the *tag's* frame, so the cube stays visible when the
  gripper hides its top face. Tag frame convention (AprilTag 3, as returned
  by `pupil-apriltags`): x to the tag's right, y down, **z into the tag, away
  from the camera**. A face tag on a 25.4 mm cube has the cube centre at
  `(0, 0, +0.0127)`.
- **Calibration** (finding where the camera is relative to each arm base).
  The table tag stays fixed. Instead of measuring its position by hand, we
  touch its four corners with the gripper tip and read the grasp-point
  position the embodiment reports, once per arm, in a fixed order: the
  *printed tag's* top-left, top-right, bottom-right, bottom-left (mark the
  printed top-left corner with a pen; the order follows the tag's own
  orientation, not the camera image). That gives the tag's pose
  in each arm's frame to a few millimetres. At the start of each run the
  code sees the tag, solves the camera pose, and caches it. The `doctor`
  command gains a check that the table tag is visible.
- **Distance refinement.** With depth available, the tag detector's distance
  is replaced by the median depth in a small window around the tag centre.
  The detector alone is enough to run; depth just improves it.
- **Rounding** to 0.5 cm in the serializer hides frame-to-frame jitter (R2).

## 6. Safety

Guardrails unchanged. On top of that: the largest menu step (5 cm) bounds
any single decision; the phase machine only allows a grasp when the gripper
is within tolerance; every positional goal is clamped into the arm's
configured workspace bounds before the tolerance test, so a goal below the
`z` floor (default 0.03 m) cannot deadlock a phase; a gripper that closes on
nothing goes through a `reopen` phase before the approach is retried, so it
cannot loop closed; and `max_steps` on the task (control ticks, set to 600)
stops a trial that stalls.
In the test, a successful approach took about 11 to 12 decisions.

## 7. Settings

`-P model=typesafe/jev-1.13`, `-P api=openrouter|typesafe` (default
`openrouter`),
`-P steps_cm=0.5,2,5`, `-P tags=path.json`, `-P camera=<name>`,
`-P history=5`, `-P calibration=path.json`, `-P stale_after=10`. API keys
come from the environment or the `.env` file.

## 8. Testing

- The repo's usual gates: ruff, mypy strict, pytest at 100 % coverage.
- The network client is tested through a swappable "send" function, so
  tests never touch the network: both backends, error mapping, and retry on
  rate limits honouring the `retry-after` header.
- The perceiver is tested on synthetic images drawn with a known camera and
  tag pose. Calibration is tested as a round trip.
- The phase machine and serializer are pure functions over a small
  `WorldState` data class, so they get table-driven tests, including hidden
  tags and the exact wording rules from §2. One test pins the direction
  words and the 0.5 cm rounding, because the test showed wording decides
  success.
- End to end: the framework's built-in mock world with a fake perceiver and
  recorded Jev answers runs a full eval and checks the log and transcript.
- Live check, not in CI: a cleaned-up copy of test v3b kept under
  `examples/`, rerun against the real API before touching hardware, to catch
  wording regressions.

## 9. Review answers (2026-09-19)

1. **Task:** pick up a cube and place it in a bowl, one arm active. Cube
   about 1 inch on a side, bowl about 4 inches across. Exact millimetre
   sizes to be measured (see next steps); tag sizes above assume 25 mm
   faces.
2. **Tags:** as in §5. Cube tags 20 mm, bowl tags 40 mm, table tag 80 mm.
3. **Camera:** top camera, rigid mount, once-per-run calibration is fine.
   Wrist cameras exist but are unused in version 1.
4. **Backend:** OpenRouter.
5. **Requests:** one Choice per request, no extra diagnostic question.
6. **Target selection:** by code in version 1; Jev-picks-the-target as a
   second call is the first follow-up (D3).

## 10. Not in the first version

Jev choosing the target object (second call per phase, D3); wrist cameras;
learned or segmentation-based perception; using Score for continuous
offsets; acting on confidence; the Isaac Sim embodiment (nothing blocks it later, since the perceiver only
needs images and intrinsics).

---

# Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ship `plugins/inspect-robots-jev/`, a policy named `jev` that reads AprilTag poses from the top camera, writes a curated text state, asks Jev one Choice question per step over the OpenRouter Decisions API, and turns the pick into a bounded Cartesian move on the YAM `eef_pos` interface, plus a pose-based scorer for cube-into-bowl.

**Architecture:** pure-function core (world model → phase → state text + menu → move) wrapped by a thin `JevPolicy` that owns I/O (tag detector, HTTP client). Every I/O edge is injectable so the whole policy runs in tests with no network, camera, or robot. Core, YAM embodiment, guardrails, rollout, sinks, and report are untouched.

**Tech stack:** Python ≥ 3.10, numpy, `pupil-apriltags` (AprilTag 3 bindings; gives 6-DoF pose when handed camera intrinsics and tag size, so no OpenCV needed), stdlib `urllib` for HTTP. Dev: pytest, pytest-cov, mypy strict, ruff (extends root config).

**Spec:** the design sections above (§0–§10 of this file).

## Global constraints

- Core package stays NumPy-only; every new dependency lives in this plugin's `pyproject.toml`.
- Plugin gates: `ruff check`, `ruff format --check`, `mypy --strict` over `src` and `tests`, `pytest --cov=inspect_robots_jev --cov-fail-under=100`.
- Ruff D1: docstring on every public module, class, function; state the contract, not the name. Tests are exempt (`"tests/**" = ["D1"]`).
- No `git` commands are run by the implementer if it is Codex; the main session commits. Commit steps below are for the main session.
- Direction words are fixed: `+x FORWARD`, `-x BACK`, `+y LEFT`, `-y RIGHT`, `+z UP`, `-z DOWN`, in the active arm's base frame (YAM `eef_pos` convention). Distances in the state are rounded to 0.5 cm.
- Jev request: exactly one Choice question per step, id `move`. No Score, no Noul.
- Backends: `openrouter` (default; `POST https://openrouter.ai/api/alpha/decisions`, `OPENROUTER_API_KEY`, model `typesafe/jev-1.13`) and `typesafe` (`POST https://api.typesafe.ai/v1/systemone`, `TYPESAFE_API_KEY`, model `jev-latest`).
- Package layout, naming, CI wiring, and README/PyPI-readme boilerplate mirror
  `plugins/inspect-robots-capx/`. The new job is added to **both** `ci-ok`
  `needs` lists (root CLAUDE.md: a job not in `ci-ok` does not gate merges);
  `uv.lock` is regenerated and committed with the scaffold (CI uses
  `uv sync --locked`). The `pypi-jev` GitHub environment and PyPI trusted
  publisher must be created by a maintainer before the release job can
  succeed — a human-only step, listed in the PR.
- Scoring order: `eval()` runs scorers **before** `policy.on_trial_end`
  (`eval.py` ~565–585), so anything a scorer needs must ride on
  `Action.meta` of the recorded steps, never on `record.metadata`.
- `request_stop` is read from `Action.meta` per action (`rollout.py` ~410),
  not from `ActionChunk.meta`. A stopped trial ends `truncated=True` with
  `status == "success"` and is scored.
- Scorers belong to `Task(scorer=[...])`; `eval()` has no scorer argument.
- Real `pupil_apriltags.Detector.detect` requires a contiguous 2-D `uint8`
  array.

## File structure

| File | Responsibility |
|---|---|
| `plugins/inspect-robots-jev/pyproject.toml` | packaging, deps, entry points `inspect_robots.policies: jev`, `inspect_robots.scorers: jev_cube_in_bowl` |
| `src/inspect_robots_jev/__init__.py` | public re-exports, `__version__` |
| `src/inspect_robots_jev/world.py` | `Vec3`, `ObjectView`, `GripperView`, `WorldState`, direction vocabulary, `describe_offset` |
| `src/inspect_robots_jev/_decisions.py` | `DecisionsClient`, `ChoiceAnswer`, `DecisionsError`; injectable `http_post` |
| `src/inspect_robots_jev/menu.py` | `Move`, `build_menu(arm, kind, target_name, steps_cm)`, `parse_option(option_id)` |
| `src/inspect_robots_jev/phase.py` | `Phase`, `TaskConfig`, `PhaseMachine` (cube-into-bowl transitions) |
| `src/inspect_robots_jev/serializer.py` | `build_state(world, phase, history)` → JSON-able dict |
| `src/inspect_robots_jev/motion.py` | `MotionMapper`: `Move` + current `eef_state` → `ActionChunk` on an absolute Cartesian `Box` |
| `src/inspect_robots_jev/calibration.py` | `Calibration` (camera→arm transforms), `solve_camera_pose`, JSON load/save |
| `src/inspect_robots_jev/perceiver.py` | `TagSpec`, `TagLayout`, `Perceiver.update(observation, step)` → `WorldState` |
| `src/inspect_robots_jev/policy.py` | `JevPolicy(PolicyBase)`, `jev_policy(**kwargs)` registry entry |
| `src/inspect_robots_jev/scorer.py` | `cube_in_bowl(...)` scorer reading the last step's `action.meta["jev"]["world"]` |
| `tests/` | one test module per source module + `test_end_to_end.py` |
| `examples/jev_textsim.py` (repo root `examples/`) | live wording-regression probe (spike v3b cleaned up), not in CI |
| `README.md`, `CLAUDE.md`, root `CHANGELOG.md`, `.github/workflows/ci.yml`, `release.yml` | docs and CI wiring |

Shared vocabulary used by several tasks (defined in Task 2, `world.py`):

```python
Vec3 = tuple[float, float, float]            # metres, (x, y, z)
Arm = Literal["left", "right"]
AXES: tuple[str, str, str] = ("x", "y", "z")
DIRECTION: dict[tuple[str, int], str] = {("x", 1): "FORWARD", ("x", -1): "BACK",
    ("y", 1): "LEFT", ("y", -1): "RIGHT", ("z", 1): "UP", ("z", -1): "DOWN"}
```

---

### Task 1: Plugin scaffold, packaging, CI wiring

**Files:**
- Create: `plugins/inspect-robots-jev/pyproject.toml`, `README.md`, `CLAUDE.md`, `src/inspect_robots_jev/__init__.py`, `src/inspect_robots_jev/py.typed`, `tests/__init__.py`, `tests/test_package.py`
- Modify: `.github/workflows/ci.yml` (copy the capx job block, lines ~385–404, renaming to jev), `.github/workflows/release.yml` (copy the capx publish job, lines ~203–217)

**Interfaces:**
- Produces: importable package `inspect_robots_jev` with `__version__`; entry points `inspect_robots.policies` → `jev = inspect_robots_jev.policy:jev_policy`, `inspect_robots.scorers` → `jev_cube_in_bowl = inspect_robots_jev.scorer:cube_in_bowl`. (Targets are created in Tasks 10 and 11; until then `__init__` must not import them, so the package imports cleanly.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
from importlib.metadata import entry_points

import inspect_robots_jev


def test_version_is_string() -> None:
    assert isinstance(inspect_robots_jev.__version__, str)


def test_entry_points_declared() -> None:
    policies = {ep.name: ep.value for ep in entry_points(group="inspect_robots.policies")}
    scorers = {ep.name: ep.value for ep in entry_points(group="inspect_robots.scorers")}
    assert policies["jev"] == "inspect_robots_jev.policy:jev_policy"
    assert scorers["jev_cube_in_bowl"] == "inspect_robots_jev.scorer:cube_in_bowl"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-jev/tests/test_package.py -q`
Expected: FAIL, `ModuleNotFoundError: inspect_robots_jev`.

- [ ] **Step 3: Write pyproject.toml**

Copy `plugins/inspect-robots-capx/pyproject.toml` and change: `name = "inspect-robots-jev"`, `version = "0.1.0"`, description `"TypeSafe Jev structured-decision model as an Inspect Robots policy over AprilTag world state."`, keywords add `"apriltag", "jev", "typesafe"`, dependencies:

```toml
dependencies = [
    "inspect-robots>=0.30",
    "numpy>=1.24",
    "pupil-apriltags>=1.0.4",
]
```

Entry points:

```toml
[project.entry-points."inspect_robots.policies"]
jev = "inspect_robots_jev.policy:jev_policy"

[project.entry-points."inspect_robots.scorers"]
jev_cube_in_bowl = "inspect_robots_jev.scorer:cube_in_bowl"
```

`[tool.uv.sources]` keeps only `inspect-robots = { workspace = true }`. `known-first-party = ["inspect_robots", "inspect_robots_jev"]`. Wheel/sdist paths use `src/inspect_robots_jev`. Keep the fancy-pypi-readme block verbatim.

- [ ] **Step 4: Write `__init__.py`, `py.typed`, README stub, CLAUDE.md stub**

```python
"""inspect-robots-jev: TypeSafe's Jev decision model as an Inspect Robots policy.

Jev reads text and picks one option from a list; it never sees images. This
plugin reads AprilTag poses from the fixed top camera, writes a curated text
state in directional words, asks Jev one Choice question per step, and turns
the pick into a bounded Cartesian move. Registered as the policy ``jev`` and
the scorer ``jev_cube_in_bowl``.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("inspect-robots-jev")
```

`README.md`: title, one-paragraph description (from the spec §0 Jev entry), install line `pip install inspect-robots inspect-robots-jev`, and a "Status: under construction" note; fuller docs land in Task 13. `CLAUDE.md`: the file table from "File structure" above and the Global constraints list.

- [ ] **Step 5: Install into the workspace and run the test**

Run: `uv lock && uv sync --all-packages --extra dev && uv run --no-sync python -m pytest plugins/inspect-robots-jev/tests/test_package.py -q`
Expected: PASS (entry points resolve from metadata even though targets do not exist yet). `uv.lock` changes and is committed in Step 7.

- [ ] **Step 6: CI wiring**

In `.github/workflows/ci.yml` duplicate the capx plugin job with every `capx` → `jev`, `inspect_robots_capx` → `inspect_robots_jev`; set `--cov-fail-under=100`; add `plugin-jev` to both `ci-ok` `needs` lists (~lines 443 and 456). In `release.yml` duplicate the capx publish job for `inspect-robots-jev`. Run `uv run --no-sync ruff check plugins/inspect-robots-jev && uv run --no-sync ruff format --check plugins/inspect-robots-jev`; expected clean.

- [ ] **Step 7: Commit**

```bash
git add plugins/inspect-robots-jev .github/workflows/ci.yml .github/workflows/release.yml uv.lock
git commit -m "feat(jev): scaffold inspect-robots-jev plugin package and CI"
```

---

### Task 2: World model and direction vocabulary (`world.py`)

**Files:**
- Create: `src/inspect_robots_jev/world.py`, `tests/test_world.py`

**Interfaces:**
- Produces:

```python
Vec3 = tuple[float, float, float]
Arm = Literal["left", "right"]
ARMS: tuple[Arm, Arm] = ("left", "right")
AXES = ("x", "y", "z")
DIRECTION: dict[tuple[str, int], str]      # see shared vocabulary
ROUND_M = 0.005                            # 0.5 cm

def round_half_cm(metres: float) -> float
def describe_offset(target: Vec3, origin: Vec3) -> list[str]
    # e.g. ["3.5 cm FORWARD", "aligned on y", "4 cm BELOW"]  -- BELOW/ABOVE for z
def distance(a: Vec3, b: Vec3) -> float

@dataclass(frozen=True)
class ObjectView:
    name: str
    in_frame: Mapping[Arm, Vec3]           # object centre in each arm's base frame
    last_seen_step: int

@dataclass(frozen=True)
class GripperView:
    position: Vec3                         # grasp point, own base frame
    opening: float                         # 0 closed .. 1 open

@dataclass(frozen=True)
class WorldState:
    step: int
    objects: Mapping[str, ObjectView]
    grippers: Mapping[Arm, GripperView]
    def offset(self, arm: Arm, obj: str) -> Vec3       # target - gripper, arm frame
    def steps_since_seen(self, obj: str) -> int
```

Note on z wording: `describe_offset` uses ABOVE/BELOW for z (where the target is relative to the gripper); `DIRECTION` uses UP/DOWN for *moves*. Both are pinned by tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_world.py
import pytest

from inspect_robots_jev.world import (
    DIRECTION, GripperView, ObjectView, WorldState, describe_offset, distance, round_half_cm,
)


@pytest.mark.parametrize(("m", "expected"), [(0.0349, 0.035), (0.0374, 0.035), (0.0376, 0.04), (-0.0126, -0.015), (0.002, 0.0)])
def test_round_half_cm(m: float, expected: float) -> None:
    assert round_half_cm(m) == pytest.approx(expected)


def test_describe_offset_words() -> None:
    words = describe_offset(target=(0.135, 0.20, 0.06), origin=(0.10, 0.20, 0.10))
    assert words == ["3.5 cm FORWARD", "aligned on y", "4 cm BELOW"]


def test_describe_offset_other_signs() -> None:
    words = describe_offset(target=(0.08, 0.15, 0.20), origin=(0.10, 0.20, 0.10))
    assert words == ["2 cm BACK", "5 cm RIGHT", "10 cm ABOVE"]


def test_direction_table_is_complete() -> None:
    assert set(DIRECTION.values()) == {"FORWARD", "BACK", "LEFT", "RIGHT", "UP", "DOWN"}


def test_world_offset_and_seen() -> None:
    world = WorldState(
        step=7,
        objects={"cube": ObjectView("cube", {"left": (0.3, 0.0, 0.02), "right": (0.3, 0.4, 0.02)}, last_seen_step=4)},
        grippers={"left": GripperView((0.2, 0.0, 0.1), 1.0), "right": GripperView((0.2, 0.0, 0.1), 1.0)},
    )
    assert world.offset("left", "cube") == pytest.approx((0.1, 0.0, -0.08))
    assert world.steps_since_seen("cube") == 3
    assert distance((0.0, 0.0, 0.0), (0.0, 0.03, 0.04)) == pytest.approx(0.05)
```

- [ ] **Step 2: Run, expect ImportError.**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-jev/tests/test_world.py -q`

- [ ] **Step 3: Implement**

```python
"""World model shared by the perceiver, phase machine, serializer, and scorer.

Positions are metres in an arm's base frame (YAM ``eef_pos`` convention:
+x forward, +y left, +z up). Wording helpers turn geometry into the
directional phrases the spike showed Jev reads correctly; do not change the
words without rerunning ``examples/jev_textsim.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Vec3 = tuple[float, float, float]
Arm = Literal["left", "right"]
ARMS: tuple[Arm, Arm] = ("left", "right")
AXES: tuple[str, str, str] = ("x", "y", "z")
#: Move words per (axis, sign).
DIRECTION: dict[tuple[str, int], str] = {
    ("x", 1): "FORWARD", ("x", -1): "BACK",
    ("y", 1): "LEFT", ("y", -1): "RIGHT",
    ("z", 1): "UP", ("z", -1): "DOWN",
}
#: Where-the-target-is words per (axis, sign); z differs from move words.
_WHERE: dict[tuple[str, int], str] = {**DIRECTION, ("z", 1): "ABOVE", ("z", -1): "BELOW"}
ROUND_M = 0.005


def round_half_cm(metres: float) -> float:
    """Round to the nearest 0.5 cm so tag jitter never changes the wording."""
    return round(metres / ROUND_M) * ROUND_M


def _cm_text(metres: float) -> str:
    cm = abs(metres) * 100.0
    return f"{cm:g} cm"


def describe_offset(target: Vec3, origin: Vec3) -> list[str]:
    """Describe where ``target`` is relative to ``origin``, one phrase per axis."""
    words: list[str] = []
    for axis, t, o in zip(AXES, target, origin, strict=True):
        d = round_half_cm(t - o)
        if d == 0.0:
            words.append(f"aligned on {axis}")
        else:
            words.append(f"{_cm_text(d)} {_WHERE[(axis, 1 if d > 0 else -1)]}")
    return words


def distance(a: Vec3, b: Vec3) -> float:
    """Euclidean distance in metres."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


@dataclass(frozen=True)
class ObjectView:
    """One tagged object's centre in every arm frame and when it was last seen."""

    name: str
    in_frame: Mapping[Arm, Vec3]
    last_seen_step: int


@dataclass(frozen=True)
class GripperView:
    """One gripper's grasp point in its own base frame and its opening (0 closed, 1 open)."""

    position: Vec3
    opening: float


@dataclass(frozen=True)
class WorldState:
    """Everything the phase machine and serializer may look at for one step."""

    step: int
    objects: Mapping[str, ObjectView]
    grippers: Mapping[Arm, GripperView]

    def offset(self, arm: Arm, obj: str) -> Vec3:
        """Return ``object - gripper`` in ``arm``'s frame."""
        target = self.objects[obj].in_frame[arm]
        origin = self.grippers[arm].position
        return (target[0] - origin[0], target[1] - origin[1], target[2] - origin[2])

    def steps_since_seen(self, obj: str) -> int:
        """Steps elapsed since the object's tag was last detected."""
        return self.step - self.objects[obj].last_seen_step
```

- [ ] **Step 4: Run tests; expect PASS. Run `ruff` + `mypy --strict` on the plugin; expect clean.**

- [ ] **Step 5: Commit** — `git commit -am "feat(jev): world model and directional wording"` (after `git add` of the two files).

---

### Task 3: Decisions API client (`_decisions.py`)

**Files:**
- Create: `src/inspect_robots_jev/_decisions.py`, `tests/test_decisions.py`

**Interfaces:**
- Produces:

```python
HttpPost = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]
#             (url, headers, body, timeout_s) -> (status, response_headers_lowercased, body)

class DecisionsError(RuntimeError): ...        # message carries status and body excerpt

@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    model: str
    usage: Mapping[str, Any]
    raw: Mapping[str, Any]

class DecisionsClient:
    def __init__(self, *, api: str = "openrouter", model: str | None = None,
                 api_key: str | None = None, http_post: HttpPost | None = None,
                 timeout_s: float = 30.0, max_attempts: int = 4, sleep: Callable[[float], None] = time.sleep) -> None
    def choose(self, *, state: Any, instructions: str, criteria: Mapping[str, str],
               question_id: str = "move") -> ChoiceAnswer
```

Backend table: `openrouter` → url `https://openrouter.ai/api/alpha/decisions`, key env `OPENROUTER_API_KEY`, default model `typesafe/jev-1.13`; `typesafe` → `https://api.typesafe.ai/v1/systemone`, `TYPESAFE_API_KEY`, `jev-latest`. Unknown `api` → `ValueError`. Missing key → `DecisionsError` naming the env var. Retries on 429 and 5xx up to `max_attempts` using `retry-after` seconds when present else `1.5 * attempt`; other 4xx raise immediately. Response validation: `answers[question_id]["type"] == "choice"` and `choice in criteria`, else `DecisionsError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_decisions.py
import json
from typing import Any

import pytest

from inspect_robots_jev._decisions import ChoiceAnswer, DecisionsClient, DecisionsError

OK = {"model": "typesafe/jev-1.13-20260917", "answers": {"move": {"type": "choice", "choice": "a",
      "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.4}}, "usage": {"input_tokens": 10, "output_tokens": 2}}


class FakePost:
    def __init__(self, responses: list[tuple[int, dict[str, str], Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((url, headers, json.loads(body)))
        status, hdrs, payload = self.responses.pop(0)
        return status, hdrs, json.dumps(payload).encode()


def test_openrouter_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = FakePost([(200, {}, OK)])
    client = DecisionsClient(http_post=post)
    ans = client.choose(state={"x": 1}, instructions="pick", criteria={"a": "A", "b": "B"})
    url, headers, body = post.calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer k"
    assert body == {"model": "typesafe/jev-1.13", "state": {"x": 1},
                    "questions": {"move": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}}
    assert ans == ChoiceAnswer(choice="a", probabilities={"a": 0.7, "b": 0.3}, confidence=0.4,
                               model="typesafe/jev-1.13-20260917", usage={"input_tokens": 10, "output_tokens": 2}, raw=OK)


def test_typesafe_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    post = FakePost([(200, {}, OK)])
    DecisionsClient(api="typesafe", http_post=post).choose(state="s", instructions="i", criteria={"a": "A"})
    url, headers, body = post.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone" and body["model"] == "jev-latest"


def test_unknown_api() -> None:
    with pytest.raises(ValueError, match="api"):
        DecisionsClient(api="nope", api_key="k")


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(DecisionsError, match="OPENROUTER_API_KEY"):
        DecisionsClient(http_post=FakePost([]))


def test_retry_then_success() -> None:
    slept: list[float] = []
    post = FakePost([(429, {"retry-after": "2"}, {"error": "slow"}), (503, {}, {"error": "down"}), (200, {}, OK)])
    client = DecisionsClient(api_key="k", http_post=post, sleep=slept.append)
    assert client.choose(state="s", instructions="i", criteria={"a": "A", "b": "B"}).choice == "a"
    assert slept == [2.0, 3.0]


def test_gives_up_after_max_attempts() -> None:
    post = FakePost([(500, {}, {"error": "x"})] * 2)
    with pytest.raises(DecisionsError, match="500"):
        DecisionsClient(api_key="k", http_post=post, sleep=lambda s: None, max_attempts=2).choose(
            state="s", instructions="i", criteria={"a": "A"})


def test_client_error_no_retry() -> None:
    post = FakePost([(400, {}, {"error": {"message": "bad"}})])
    with pytest.raises(DecisionsError, match="400"):
        DecisionsClient(api_key="k", http_post=post).choose(state="s", instructions="i", criteria={"a": "A"})
    assert len(post.calls) == 1


def test_choice_not_in_criteria() -> None:
    bad = {**OK, "answers": {"move": {**OK["answers"]["move"], "choice": "zzz"}}}
    with pytest.raises(DecisionsError, match="zzz"):
        DecisionsClient(api_key="k", http_post=FakePost([(200, {}, bad)])).choose(state="s", instructions="i", criteria={"a": "A"})


def test_malformed_body() -> None:
    post = FakePost([(200, {}, {"answers": {}})])
    with pytest.raises(DecisionsError, match="move"):
        DecisionsClient(api_key="k", http_post=post).choose(state="s", instructions="i", criteria={"a": "A"})


def test_default_http_post_is_urllib(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_robots_jev import _decisions

    class Resp:
        status = 200
        headers = {"Content-Type": "application/json"}
        def read(self) -> bytes: return json.dumps(OK).encode()
        def __enter__(self) -> "Resp": return self
        def __exit__(self, *a: object) -> None: return None

    monkeypatch.setattr(_decisions.urllib.request, "urlopen", lambda req, timeout: Resp())
    status, hdrs, body = _decisions.urllib_post("https://x", {"A": "b"}, b"{}", 1.0)
    assert status == 200 and json.loads(body) == OK and hdrs["content-type"] == "application/json"


def test_default_http_post_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error
    from inspect_robots_jev import _decisions

    def boom(req: Any, timeout: float) -> Any:
        raise urllib.error.HTTPError("https://x", 429, "slow", {"Retry-After": "1"}, io.BytesIO(b'{"e":1}'))

    monkeypatch.setattr(_decisions.urllib.request, "urlopen", boom)
    status, hdrs, body = _decisions.urllib_post("https://x", {}, b"{}", 1.0)
    assert status == 429 and hdrs["retry-after"] == "1" and body == b'{"e":1}'
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Implement**

```python
"""Minimal client for Jev's Decisions API (OpenRouter alpha or TypeSafe direct).

One request = one ``state`` plus one Choice question. The HTTP edge is an
injectable ``http_post`` so tests never touch the network. Retries follow
the providers' guidance: 429 and 5xx retry with ``retry-after`` when given.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

HttpPost = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]

_BACKENDS: dict[str, tuple[str, str, str]] = {
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY", "typesafe/jev-1.13"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-latest"),
}


class DecisionsError(RuntimeError):
    """The Decisions API refused, failed, or returned an unusable answer."""


@dataclass(frozen=True)
class ChoiceAnswer:
    """One Choice answer: the pick, every option's probability, and confidence."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    model: str
    usage: Mapping[str, Any]
    raw: Mapping[str, Any]


def urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, str], bytes]:
    """Default ``HttpPost`` over stdlib urllib; HTTP errors become status tuples, not exceptions."""
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), {k.lower(): v for k, v in dict(resp.headers).items()}, resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), {k.lower(): v for k, v in dict(err.headers).items()}, err.read()


class DecisionsClient:
    """Ask Jev one Choice question about one state."""

    def __init__(
        self,
        *,
        api: str = "openrouter",
        model: str | None = None,
        api_key: str | None = None,
        http_post: HttpPost | None = None,
        timeout_s: float = 30.0,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if api not in _BACKENDS:
            raise ValueError(f"api must be one of {sorted(_BACKENDS)}, got {api!r}")
        url, key_env, default_model = _BACKENDS[api]
        key = api_key if api_key is not None else os.environ.get(key_env)
        if not key:
            raise DecisionsError(f"no API key: pass api_key= or set {key_env}")
        self._url = url
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model = model or default_model
        self._post = http_post if http_post is not None else urllib_post
        self._timeout = timeout_s
        self._max_attempts = max_attempts
        self._sleep = sleep

    def choose(
        self, *, state: Any, instructions: str, criteria: Mapping[str, str], question_id: str = "move"
    ) -> ChoiceAnswer:
        """POST one Choice question; return the validated answer."""
        body = json.dumps(
            {"model": self.model, "state": state,
             "questions": {question_id: {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}}}
        ).encode()
        status, headers, payload = 0, {}, b""
        for attempt in range(1, self._max_attempts + 1):
            status, headers, payload = self._post(self._url, self._headers, body, self._timeout)
            if status == 429 or status >= 500:
                if attempt == self._max_attempts:
                    break
                retry_after = headers.get("retry-after")
                self._sleep(float(retry_after) if retry_after else 1.5 * attempt)
                continue
            break
        if status != 200:
            raise DecisionsError(f"decisions API returned HTTP {status}: {payload[:300]!r}")
        raw = json.loads(payload)
        answer = raw.get("answers", {}).get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise DecisionsError(f"no choice answer for question {question_id!r}: {raw!r}"[:500])
        choice = str(answer.get("choice"))
        if choice not in criteria:
            raise DecisionsError(f"choice {choice!r} is not one of the offered options")
        return ChoiceAnswer(
            choice=choice,
            probabilities={str(k): float(v) for k, v in answer.get("probabilities", {}).items()},
            confidence=float(answer.get("confidence", 0.0)),
            model=str(raw.get("model", self.model)),
            usage=dict(raw.get("usage", {})),
            raw=raw,
        )
```

- [ ] **Step 4: Run tests, ruff, mypy; expect PASS/clean.**
- [ ] **Step 5: Commit** — `feat(jev): Decisions API client with retry and validation`.

---

### Task 4: Menu builder and option parser (`menu.py`)

**Files:**
- Create: `src/inspect_robots_jev/menu.py`, `tests/test_menu.py`

**Interfaces:**
- Consumes: `Arm`, `AXES`, `DIRECTION` from `world.py` only (no `phase.py` import, so `phase.py` can import `MenuKind` without a cycle).
- Produces:

```python
MenuKind = Literal["xyz", "z", "grip_close", "grip_open", "done"]
HOLD = "hold"

@dataclass(frozen=True)
class Move:
    arm: Arm
    axis: str | None          # "x"|"y"|"z" or None
    delta_m: float            # signed metres, 0 for gripper/hold
    gripper: float | None     # target opening 0..1, or None
    @property
    def is_hold(self) -> bool

def build_menu(arm: Arm, kind: MenuKind, target_name: str, steps_cm: Sequence[float]) -> dict[str, str]
def parse_option(option_id: str) -> Move
```

Option id grammar (parseable without the menu): `{arm}_{axis}_{plus|minus}_{cm:g}cm`, `{arm}_close`, `{arm}_open`, `hold`. Hint text per step size (steps sorted ascending, hints derived from neighbours): smallest → `use when the {target} is less than {next} cm away in that direction`; middle → `use when the {target} is {this} to {next} cm away in that direction`; largest → `use only when the {target} is more than {this} cm away in that direction`. `z` kind lists only UP/DOWN. `grip_close` → `{arm}_close`: `close the {arm} gripper on the {target}` + `hold`; `grip_open` → `{arm}_open`. `done` → `{HOLD}` only.

- [ ] **Step 1: Failing tests**

```python
# tests/test_menu.py
import pytest

from inspect_robots_jev.menu import HOLD, Move, build_menu, parse_option


def test_xyz_menu_shape_and_hints() -> None:
    menu = build_menu("left", "xyz", "red cube", [0.5, 2, 5])
    assert len(menu) == 19 and HOLD in menu
    assert menu["left_x_plus_2cm"] == "move the left gripper FORWARD by 2 cm; use when the red cube is 2 to 5 cm away in that direction"
    assert menu["left_z_minus_0.5cm"].startswith("move the left gripper DOWN by 0.5 cm; use when the red cube is less than 2 cm away")
    assert menu["left_y_minus_5cm"].endswith("use only when the red cube is more than 5 cm away in that direction")


def test_z_menu() -> None:
    menu = build_menu("right", "z", "bowl", [0.5, 2])
    assert set(menu) == {"right_z_plus_0.5cm", "right_z_minus_0.5cm", "right_z_plus_2cm", "right_z_minus_2cm", HOLD}


def test_gripper_menus() -> None:
    assert set(build_menu("left", "grip_close", "red cube", [1])) == {"left_close", HOLD}
    assert set(build_menu("left", "grip_open", "bowl", [1])) == {"left_open", HOLD}
    assert set(build_menu("left", "done", "bowl", [1])) == {HOLD}


@pytest.mark.parametrize(("option", "move"), [
    ("left_x_plus_2cm", Move("left", "x", 0.02, None)),
    ("right_z_minus_0.5cm", Move("right", "z", -0.005, None)),
    ("left_close", Move("left", None, 0.0, 0.0)),
    ("right_open", Move("right", None, 0.0, 1.0)),
])
def test_parse_option(option: str, move: Move) -> None:
    parsed = parse_option(option)
    assert (parsed.arm, parsed.axis, parsed.gripper) == (move.arm, move.axis, move.gripper)
    assert parsed.delta_m == pytest.approx(move.delta_m)
    assert not parsed.is_hold


def test_parse_hold_and_garbage() -> None:
    assert parse_option(HOLD).is_hold
    with pytest.raises(ValueError):
        parse_option("left_sideways_3cm")
```

- [ ] **Step 2: Run; expect ImportError.**
- [ ] **Step 3: Implement**

```python
"""Build the Choice options Jev picks from, and parse its pick back into a move.

Every option's description carries a when-to-use hint; the spike showed that
without hints Jev picks the largest step regardless of distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from inspect_robots_jev.world import AXES, DIRECTION, Arm

MenuKind = Literal["xyz", "z", "grip_close", "grip_open", "done"]
HOLD = "hold"
_SIGN = {"plus": 1, "minus": -1}


@dataclass(frozen=True)
class Move:
    """A parsed option: one axis displacement, a gripper target, or hold."""

    arm: Arm
    axis: str | None
    delta_m: float
    gripper: float | None

    @property
    def is_hold(self) -> bool:
        """True when the option asks for no motion at all."""
        return self.axis is None and self.gripper is None


def _hints(target: str, steps_cm: Sequence[float]) -> dict[float, str]:
    steps = sorted(steps_cm)
    hints: dict[float, str] = {}
    for i, s in enumerate(steps):
        if len(steps) == 1:
            hints[s] = f"the only step size; use it whenever the {target} is not yet reached"
        elif i == 0:
            hints[s] = f"use when the {target} is less than {steps[1]:g} cm away in that direction"
        elif i == len(steps) - 1:
            hints[s] = f"use only when the {target} is more than {s:g} cm away in that direction"
        else:
            hints[s] = f"use when the {target} is {s:g} to {steps[i + 1]:g} cm away in that direction"
    return hints


def build_menu(arm: Arm, kind: MenuKind, target_name: str, steps_cm: Sequence[float]) -> dict[str, str]:
    """Return ``{option_id: description}`` for one phase of the task."""
    menu: dict[str, str] = {}
    if kind in ("xyz", "z"):
        hints = _hints(target_name, steps_cm)
        axes = AXES if kind == "xyz" else ("z",)
        for axis in axes:
            for s in sorted(steps_cm):
                for word, sign in _SIGN.items():
                    menu[f"{arm}_{axis}_{word}_{s:g}cm"] = (
                        f"move the {arm} gripper {DIRECTION[(axis, sign)]} by {s:g} cm; {hints[s]}"
                    )
        menu[HOLD] = f"do not move this turn (only if the {arm} gripper is already at the {target_name})"
    elif kind == "grip_close":
        menu[f"{arm}_close"] = f"close the {arm} gripper on the {target_name}"
        menu[HOLD] = "do not close yet"
    elif kind == "grip_open":
        menu[f"{arm}_open"] = f"open the {arm} gripper to release into the {target_name}"
        menu[HOLD] = "do not open yet"
    else:
        menu[HOLD] = "the task is complete; do nothing"
    return menu


def parse_option(option_id: str) -> Move:
    """Invert ``build_menu``'s id grammar; raise ``ValueError`` on anything else."""
    if option_id == HOLD:
        return Move("left", None, 0.0, None)
    parts = option_id.split("_")
    if len(parts) == 2 and parts[0] in ("left", "right") and parts[1] in ("close", "open"):
        return Move(cast(Arm, parts[0]), None, 0.0, 0.0 if parts[1] == "close" else 1.0)
    if len(parts) == 4 and parts[0] in ("left", "right") and parts[1] in AXES and parts[2] in _SIGN and parts[3].endswith("cm"):
        cm = float(parts[3][:-2])
        return Move(cast(Arm, parts[0]), parts[1], _SIGN[parts[2]] * cm / 100.0, None)
    raise ValueError(f"unrecognised option id {option_id!r}")
```

- [ ] **Step 4: Run tests, ruff, mypy.** (Also add a test hitting the single-step hint branch: `build_menu("left","xyz","c",[2])["left_x_plus_2cm"].endswith("not yet reached")`.)
- [ ] **Step 5: Commit** — `feat(jev): phase menus with usage hints and option parser`.

---

### Task 5: Phase machine (`phase.py`)

**Files:**
- Create: `src/inspect_robots_jev/phase.py`, `tests/test_phase.py`

**Interfaces:**
- Consumes: `WorldState`, `Arm`, `ARMS`, `distance` from `world.py`; `MenuKind` from `menu.py`.
- Produces:

```python
PhaseName = Literal["approach", "descend", "grasp", "reopen", "lift", "carry", "lower", "release", "retreat", "done"]

@dataclass(frozen=True)
class TaskConfig:
    cube: str = "cube"
    bowl: str = "bowl"
    hover_m: float = 0.05           # hover height above cube centre / bowl centre
    xy_tol_m: float = 0.01
    z_tol_m: float = 0.01
    lift_m: float = 0.08            # lift height above cube's table height
    bowl_depth_m: float = 0.04      # how far below the bowl tag height counts as "in"
    closed_on_object: float = 0.35  # gripper opening ≤ this while closed on the cube
    closed_on_air: float = 0.02     # ≤ this means the jaws met: nothing was grasped
    open_threshold: float = 0.8
    verify_decisions: int = 8       # wait this long in `verify` for the cube's tag to reappear
    contact_decisions: int = 3      # flat measured height over this many decisions = contact
    cube_half_m: float = 0.0127

@dataclass(frozen=True)
class Phase:
    name: PhaseName
    arm: Arm
    target: str                     # object name the menu talks about
    menu_kind: MenuKind
    goal: Vec3 | None               # where the gripper should be (arm frame) for this phase, if positional

class PhaseMachine:
    def __init__(self, config: TaskConfig, *, bounds: tuple[Vec3, Vec3] | None = None) -> None
        # bounds = (low_xyz, high_xyz) of the arm workspace; every positional goal is
        # clamped into them before the tolerance test (YAM default z floor is 0.03 m,
        # below a cube centre on the table). None = no clamping (tests).
    @property
    def phase(self) -> Phase
    def reset(self, world: WorldState) -> Phase      # picks the arm whose BASE is nearer the cube:
                                                     #   min(ARMS, key=lambda a: norm(cube.in_frame[a]))
    def advance(self, world: WorldState) -> Phase    # apply at most one transition, return current
```

The machine stores `_grasp_point` (the gripper position at the `grasp → lift`
transition) because the `lift` goal is relative to where the grasp happened;
every other goal is recomputed from the current world each call, then
clamped into `bounds`. It also keeps a short trace of measured gripper
heights for the contact rule and a `verify` decision counter; both reset on
every phase change. The cube's pose is substituted from the gripper during
`lift`/`carry`/`lower` only (the policy's `_HELD_PHASES`).

Transition rules (evaluated in order, one transition per call):
- `approach`: goal = clamp(cube + (0,0,hover)). → `descend` when xy within `xy_tol` and z within `z_tol` of goal.
- `descend`: goal = clamp(cube centre). menu `z`. → `grasp` when z within `z_tol`. (On the rig the clamp raises this to the `z` floor when the cube centre is below it; set `eef_low` z per rig so the fingertips can reach the cube.)
- `grasp`: menu `grip_close`. → `lift` when gripper opening ≤ `closed_on_object` and > `closed_on_air` (closed on something). If opening ≤ `closed_on_air` (closed on air) → `reopen`.
- `reopen`: menu `grip_open`, target cube. → `approach` when opening ≥ `open_threshold`. (Without this phase a closed-on-air gripper would bounce grasp → approach → descend → grasp forever, since approach/descend menus cannot open it.)
- `lift`: goal = grasp position + (0,0,lift). menu `z`. → `carry` when z ≥ goal z − z_tol.
- `carry`: goal = bowl + (0,0,hover + cube_half). menu `xyz`. → `lower` when xy within `xy_tol`.
- `lower`: goal = clamp(bowl − (0,0,bowl_depth) + cube_half). menu `z`. → `release` when z within z_tol.
- `release`: menu `grip_open`. → `retreat` when opening ≥ `open_threshold`.
- `retreat`: goal = bowl + (0,0,lift). menu `z`. → `verify` when z ≥ goal z − z_tol.
- `verify`: goal = bowl + (0,0,lift + hover), menu `z`, target name `cube` (the state says "raise until the camera can see the cube again"). → `done` when the cube's `ObjectView` is `not from_gripper` and `steps_since_seen(cube) == 0`, or after `verify_decisions` decisions (then the scorer fails the trial honestly). Without this phase the trial could end while the gripper still hid the cube, and a physically successful run would score 0.
- `done`: stays.
- Contact: in `descend`/`lower`, `blocked` is True when the last `contact_decisions + 1` measured z values span ≤ `z_tol_m`; the phase then completes as if the goal were reached.

The cube's position during `grasp`/`lift`/`carry`/`lower` is taken from the world (the perceiver substitutes gripper + offset when hidden, Task 8), so the machine never guesses.

- [ ] **Step 1: Failing tests** (table-driven; one helper builds a world)

```python
# tests/test_phase.py
from inspect_robots_jev.phase import PhaseMachine, TaskConfig
from inspect_robots_jev.world import GripperView, ObjectView, WorldState

CUBE = (0.30, 0.10, 0.013)
BOWL = (0.35, -0.10, 0.02)


def world(left=(0.20, 0.0, 0.20), opening=1.0, cube=CUBE, step=0):
    return WorldState(step=step,
        objects={"cube": ObjectView("cube", {"left": cube, "right": (cube[0], cube[1] + 0.5, cube[2])}, step),
                 "bowl": ObjectView("bowl", {"left": BOWL, "right": (BOWL[0], BOWL[1] + 0.5, BOWL[2])}, step)},
        grippers={"left": GripperView(left, opening), "right": GripperView((0.2, 0.5, 0.2), 1.0)})


def test_reset_picks_closer_arm_and_hover_goal() -> None:
    m = PhaseMachine(TaskConfig())
    p = m.reset(world())
    assert (p.name, p.arm, p.target, p.menu_kind) == ("approach", "left", "cube", "xyz")
    assert p.goal == (0.30, 0.10, 0.063)


def test_full_happy_path() -> None:
    m = PhaseMachine(TaskConfig()); m.reset(world())
    assert m.advance(world(left=(0.30, 0.10, 0.063))).name == "descend"
    assert m.advance(world(left=(0.30, 0.10, 0.013))).name == "grasp"
    assert m.advance(world(left=(0.30, 0.10, 0.013), opening=0.3)).name == "lift"
    assert m.advance(world(left=(0.30, 0.10, 0.093), opening=0.3, cube=(0.30, 0.10, 0.093))).name == "carry"
    assert m.advance(world(left=(0.35, -0.10, 0.093), opening=0.3, cube=(0.35, -0.10, 0.093))).name == "lower"
    p = m.advance(world(left=(0.35, -0.10, -0.0073), opening=0.3, cube=(0.35, -0.10, -0.0073)))
    assert p.name == "release" and p.menu_kind == "grip_open"
    assert m.advance(world(left=(0.35, -0.10, -0.0073), opening=0.9)).name == "retreat"
    assert m.advance(world(left=(0.35, -0.10, 0.10), opening=0.9)).name == "done"
    assert m.advance(world()).name == "done"


def test_grasp_on_air_returns_to_approach() -> None:
    m = PhaseMachine(TaskConfig()); m.reset(world())
    m.advance(world(left=(0.30, 0.10, 0.063))); m.advance(world(left=(0.30, 0.10, 0.013)))
    assert m.phase.name == "grasp"
    assert m.advance(world(left=(0.30, 0.10, 0.013), opening=0.01)).name == "approach"


def test_no_transition_when_far() -> None:
    m = PhaseMachine(TaskConfig()); m.reset(world())
    assert m.advance(world(left=(0.10, 0.10, 0.30))).name == "approach"
```

- [ ] **Step 2: Run; expect ImportError.**
- [ ] **Step 3: Implement** (dataclasses as in Interfaces; `advance` is a chain of `if self._phase.name == ...` checks each computing the goal from the current world and testing tolerances with `abs(dx) <= tol` per axis; `_goal()` helper recomputes `goal` for the current phase so `Phase.goal` always reflects the latest world). Keep the file under 150 lines; no I/O.
- [ ] **Step 4: Run tests, ruff, mypy; expect 100 % of `phase.py` covered. Add: a `right`-arm reset test with the cube nearer the right arm's base; a bounds test where the `descend` goal is below `low` z and the transition fires at the floor; `pytest.approx` on every goal assertion.**
- [ ] **Step 5: Commit** — `feat(jev): cube-into-bowl phase machine`.

---

### Task 6: State serializer (`serializer.py`)

**Files:**
- Create: `src/inspect_robots_jev/serializer.py`, `tests/test_serializer.py`

**Interfaces:**
- Consumes: `WorldState`, `describe_offset`, `distance` (world.py); `Phase` (phase.py); `HOLD` (menu.py).
- Produces:

```python
def build_state(world: WorldState, phase: Phase, history: Sequence[str], *, history_len: int = 5) -> dict[str, Any]
def instructions_for(phase: Phase) -> str
```

`build_state(world, phase, history, *, history_len=5, open_threshold=0.8, closed_threshold=0.35)` output keys (reopen → `The {arm} gripper closed on nothing. Open it again before retrying the {target}.`), in this order: `task` (one sentence per phase, see table; it names the target point), `target_point_relative_to_<arm>_gripper` (list from `describe_offset(goal, gripper)` — the *goal* point, e.g. 5 cm above the cube, is what Jev steers to, so the key says "target point", not the object's name; for gripper phases this key **and** `straight_line_distance_cm` are omitted), `straight_line_distance_cm` (rounded to 0.1), `<arm>_gripper` (`"open"`/`"closed"`/`"partly closed"` from opening ≥ `open_threshold` / ≤ `closed_threshold` / else; the policy passes `TaskConfig.open_threshold` and `closed_on_object` so the two never drift), `other_objects_on_table_ignore_them` (sorted names except the target), `recent_moves_oldest_first` (last `history_len` entries of `history`, which the policy fills with menu *descriptions* truncated before `;`), and `note` only when `world.steps_since_seen(target) > 0`: `"the <target> was last seen <n> moves ago; assume it has not moved"`.

Task sentences: approach → `Move the {arm} gripper to the target point 5 cm above the {target}. Move toward that point, never away.`; descend → `Lower the {arm} gripper straight down onto the {target}.`; grasp → `Close the {arm} gripper on the {target}.`; lift → `Raise the {arm} gripper straight up, carrying the {target}.`; carry → `Move the {arm} gripper to the target point above the {target}, carrying the cube. Move toward that point, never away.`; lower → `Lower the {arm} gripper into the {target}.`; release → `Open the {arm} gripper to drop the cube into the {target}.`; retreat → `Raise the {arm} gripper straight up, away from the {target}.`; done → `The task is complete.`

`instructions_for` (the Choice `instructions` field): positional phases → `Which move takes the {arm} gripper toward the target point? Move in the direction the target is in, along the axis with the largest remaining gap, with a step no larger than that gap.`; grip phases → `Should the {arm} gripper act now?`; done → `Pick hold.`

- [ ] **Step 1: Failing tests** — build a world and an `approach` phase; assert the exact dict equality for one case (pin wording), assert the `note` appears with `last_seen_step` older than `step`, assert gripper-phase output has no offset key, assert history is truncated to `history_len` and oldest-first.
- [ ] **Step 2: Run; expect ImportError.** **Step 3: Implement** (≤ 90 lines). **Step 4: Run tests, ruff, mypy.**
- [ ] **Step 5: Commit** — `feat(jev): curated state serializer in directional words`.

---

### Task 7: Motion mapper (`motion.py`)

**Files:**
- Create: `src/inspect_robots_jev/motion.py`, `tests/test_motion.py`

**Interfaces:**
- Consumes: `Move` (menu.py); core `Box`, `ActionSemantics`, `Action`, `ActionChunk`.
- Produces:

```python
class MotionMapper:
    def __init__(self, action_space: Box, *, control_hz: float | None, max_playout_s: float = 10.0) -> None
        # requires semantics.dim_labels with "{arm}_{x|y|z|gripper}" entries; raises ValueError otherwise
    def index(self, label: str) -> int
    def gripper_position(self, eef_state: npt.NDArray[np.floating[Any]], arm: Arm) -> Vec3
    def gripper_opening(self, eef_state: ..., arm: Arm) -> float
    def chunk(self, move: Move, current: npt.NDArray[np.floating[Any]]) -> ActionChunk
        # `current` is the pose to interpolate FROM. The policy passes its last commanded
        # target when one exists in this trial (else the measured eef_state), so IK tracking
        # lag never appears as a spurious delta clamp on the first tick and a hold repeats the
        # last command instead of walking the arm back toward its measured pose.
        # hold -> one Action equal to `current` clipped to bounds
        # axis move -> target = current; target[idx] += delta; clip; split into
        #   n = max(1, ceil(|delta| / step_limit)) actions linearly interpolated (last == target)
        # gripper -> target[gripper idx] = value; split by gripper step limit the same way
```

Step limit per index: `semantics.max_step[i]` when declared, else 5 % of `(high - low)[i]` — the same default `DeltaLimitApprover` derives, so the mapper never emits a step the approver would clamp. On the real YAM `eef_pos` box `max_step` is `None` unless `gripper_max_step` is set, so the 5 % fallback is the branch that runs on the rig; the tests must cover a Box **without** `max_step` (x range 0.33 m → 1.65 cm/tick → a 5 cm move splits into 4 actions; gripper 0–1 → 0.05/tick → 20 actions for a full close) as well as one with it. If `n` exceeds `ceil(max_playout_s * (control_hz or 10))`, raise `ValueError` (the menu's 5 cm cap makes this unreachable in practice; it is a guard).

- [ ] **Step 1: Failing tests** — build a 14-dim absolute Box with labels `left_x, left_y, left_z, left_yaw, left_pitch, left_roll, left_gripper, right_*`, low/high from the YAM defaults (`x 0.15–0.48, y −0.25–0.25, z 0.03–0.40, yaw ±π, pitch/roll 0, gripper 0–1`), `max_step` `(0.01, 0.01, 0.01, None, None, None, 0.2, ...)`. Assert: a `Move("left","x",0.02,None)` from `left_x=0.30` yields 2 actions ending exactly at 0.32 with other dims unchanged; a 0.5 cm move yields 1 action; a move that would exceed `high` is clipped to `high`; `Move("left",None,0.0,0.0)` from gripper 1.0 yields 5 actions ending at 0.0; hold yields one action equal to current; `gripper_position` returns `(x,y,z)` for each arm; a Box without labels raises `ValueError`; `control_hz` attached to the chunk.
- [ ] **Step 2: Run; expect ImportError.** **Step 3: Implement** (≤ 100 lines, numpy only). **Step 4: Run tests, ruff, mypy.**
- [ ] **Step 5: Commit** — `feat(jev): map menu picks to bounded Cartesian action chunks`.

---

### Task 8: Calibration (`calibration.py`)

**Files:**
- Create: `src/inspect_robots_jev/calibration.py`, `tests/test_calibration.py`

**Interfaces:**
- Produces:

```python
Mat4 = npt.NDArray[np.float64]  # 4x4 homogeneous transform

def transform(T: Mat4, p: Vec3) -> Vec3
def invert(T: Mat4) -> Mat4
def pose_to_matrix(R: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> Mat4

@dataclass(frozen=True)
class Calibration:
    camera_to_arm: Mapping[Arm, Mat4]
    def to_arm(self, arm: Arm, p_camera: Vec3) -> Vec3
    def save(self, path: Path) -> None          # JSON {"left": [[...]*4], "right": [[...]*4]}
    @classmethod
    def load(cls, path: Path) -> Calibration

def table_tag_pose_from_corners(corners_arm: Sequence[Vec3]) -> Mat4
    # 4 touched corner positions (arm frame, order: top-left, top-right, bottom-right, bottom-left
    # as seen in the tag image) -> tag pose in arm frame: origin = centroid, x along TL->TR, y along TL->BL, z = x×y

def solve_calibration(*, tag_in_camera: Mat4, tag_in_arm: Mapping[Arm, Mat4]) -> Calibration
    # camera_to_arm[arm] = tag_in_arm[arm] @ inv(tag_in_camera)
```

- [ ] **Step 1: Failing tests** — round-trip: build a random rotation+translation `T`, check `transform(invert(T), transform(T, p)) ≈ p`; `table_tag_pose_from_corners` on a square in the xy-plane at z=0 centred at (0.3, 0.1) with 0.08 m side returns translation (0.3, 0.1, 0) and rotation identity (within 1e-9); `solve_calibration` with a known camera pose reproduces it; save/load round-trip preserves matrices exactly.
- [ ] **Step 2–4: Run; implement (numpy only, ≤ 90 lines); run tests, ruff, mypy.**
- [ ] **Step 5: Commit** — `feat(jev): camera-to-arm calibration from a touched table tag`.

---

### Task 9: Perceiver (`perceiver.py`)

**Files:**
- Create: `src/inspect_robots_jev/perceiver.py`, `tests/test_perceiver.py`, `tests/fixtures/tags_cube_bowl.json`

**Interfaces:**
- Consumes: `Calibration`, `pose_to_matrix`, `transform` (calibration.py); `WorldState`, `ObjectView`, `GripperView` (world.py); `MotionMapper.gripper_position/opening` (motion.py, passed in as callables to keep the perceiver free of the action space).
- Produces:

```python
@dataclass(frozen=True)
class TagSpec:
    tag_id: int
    object: str
    size_m: float                 # black square edge
    offset_m: Vec3                # object centre in the TAG's frame: x right, y down, z INTO the tag (away from
                                  # the camera), so a face tag on a 25.4 mm cube has offset (0, 0, +0.0127)

@dataclass(frozen=True)
class TagLayout:
    tags: Mapping[int, TagSpec]
    table_tag_id: int
    @classmethod
    def load(cls, path: Path) -> TagLayout   # JSON: {"table_tag_id": 20, "tags": [{"id":0,"object":"cube","size_m":0.020,"offset_m":[0,0,0.0127]}, ...]}

class Detection(Protocol):        # structural match for pupil_apriltags.Detection
    tag_id: int; center: Any; corners: Any; pose_R: Any; pose_t: Any; decision_margin: float

Detector = Callable[[npt.NDArray[np.uint8], tuple[float, float, float, float], float], Sequence[Detection]]
#  (gray_image, (fx, fy, cx, cy), tag_size_m) -> detections with pose

def apriltag_detector() -> Detector      # lazy-imports pupil_apriltags; builds ONE Detector(families="tag36h11") and
                                         # reuses it; passes a C-contiguous 2-D uint8 array (detect() asserts uint8)

class Perceiver:
    def __init__(self, *, layout: TagLayout, camera: str, calibration: Calibration,
                 detector: Detector | None = None, depth_window_px: int = 5,
                 grasp_offset_m: Vec3 = (0.0, 0.0, 0.0)) -> None
    def update(self, observation: Observation, step: int, *, grippers: Mapping[Arm, GripperView],
               held: tuple[Arm, str] | None = None) -> WorldState
```

`update` algorithm: gray = `np.ascontiguousarray(observation.images[camera].mean(axis=2).astype(np.uint8))`; K from `observation.extra[f"{camera}_intrinsics"]` (3×3) → `(fx, fy, cx, cy)`; because detections need one tag size per call, group `layout.tags` by `size_m` and call the detector once per size, keeping only detections whose id has that size. For each detection: `T_tag_cam = pose_to_matrix(pose_R, pose_t)`; if `f"{camera}_depth"` present (array or zero-arg callable returning array, per YAM docs), take the median of finite positive depth in a `depth_window_px` square at `center` and rescale `pose_t` so its z equals that depth; object centre in camera = `transform(T_tag_cam, spec.offset_m)`; per arm: `calibration.to_arm(arm, centre_cam)`. Average all detections per object. Objects with no detection this step keep their previous `ObjectView` (with the old `last_seen_step`); if `held == (arm, name)`, that object's position is overwritten with `grippers[arm].position + grasp_offset_m` and marked seen. Objects never seen are omitted from `WorldState.objects` (the policy treats a missing target as "hold and log", Task 10).

- [ ] **Step 1: Failing tests** — a `FakeDetector` returning scripted detections with identity `pose_R` and chosen `pose_t`; identity calibration for both arms (right arm translated by (0, −0.5, 0)); assert: object centres land where expected including the tag→centre offset (pin the sign: identity `pose_R`, `pose_t=(0,0,0.7)`, offset `(0,0,0.0127)` → centre z `0.7127`); the real-detector wrapper hands a C-contiguous 2-D `uint8` array to a stubbed `Detector` and constructs it once across calls; two tags on one object average; a depth image overrides the tag range; an object missing on step 2 keeps its step-1 pose and `steps_since_seen == 1`; `held` substitution; `TagLayout.load` on the fixture file; `apriltag_detector()` raises a clear `ImportError` message when `pupil_apriltags` is absent (monkeypatch `sys.modules`).
- [ ] **Step 2–4: Run; implement (≤ 140 lines); run tests, ruff, mypy.**
- [ ] **Step 5: Commit** — `feat(jev): AprilTag perceiver with bundle fusion, depth refinement, and occlusion memory`.

---

### Task 10: The policy (`policy.py`)

**Files:**
- Create: `src/inspect_robots_jev/policy.py`, `tests/test_policy.py`
- Modify: `src/inspect_robots_jev/__init__.py` (export `JevPolicy`, `JevPolicyConfig`, `jev_policy`)

**Interfaces:**
- Consumes: everything above.
- Produces:

```python
@dataclass(frozen=True)
class JevPolicyConfig:
    api: str = "openrouter"
    model: str | None = None
    steps_cm: tuple[float, ...] = (0.5, 2.0, 5.0)
    camera: str = "top_cam"
    tags: str = "tags.json"
    calibration: str = "calibration.json"
    history: int = 5
    stale_after: int = 10
    task: TaskConfig = TaskConfig()

class JevPolicy(PolicyBase):
    settings: JevPolicyConfig      # NOT `config`: PolicyBase.config is the core PolicyConfig recorded in the log
    def __init__(self, config: JevPolicyConfig | None = None, *, client: DecisionsClient | None = None,
                 perceiver: Perceiver | None = None) -> None
    def bind(self, embodiment_info: EmbodimentInfo) -> None      # builds MotionMapper from embodiment action space; adopts info.action_space;
                                                                 # builds PhaseMachine(config.task, bounds=(low_xyz, high_xyz) of the active labels)
    def reset(self, scene: Scene) -> None                          # clears world/history/transcript/phase
    def act(self, observation: Observation) -> ActionChunk
    def transcript(self) -> list[dict[str, Any]]
    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None
        # record.metadata["jev"] = {"final_phase": ..., "final_world": {obj: {arm: [x,y,z]}}, "decisions": n, "stalls": n}

def jev_policy(**kwargs: Any) -> JevPolicy   # registry entry. Every -P value arrives as str: steps_cm "0.5,2,5" -> tuple[float],
                                             # history/stale_after -> int, cube=/bowl= -> TaskConfig names. Other TaskConfig
                                             # tolerances are NOT CLI-tunable in v1 (construct JevPolicyConfig in Python).
                                             # Unknown keys -> TypeError.
```

`act` algorithm: `eef = observation.state["eef_state"]`; grippers from the mapper; `start = last commanded target if any else eef`; `world = perceiver.update(..., held=self._held)` where `self._held` was set at the end of the *previous* step as `(phase.arm, cfg.task.cube) if phase.name in ("lift", "carry", "lower") else None` (the world for this step is built before `advance`). Note: with a top-down camera the cube's top tag is hidden by the gripper during `descend`/`grasp`, so the stale counter runs there. Stale budgets (`stale_after`, the scorer's `max_unseen`, `ObjectView.last_seen_step`) count **policy decisions**, one per `act`, never `env_step` control ticks (one pick plays 1–20 ticks). Only `approach`/`carry` stall on a stale target, and a stall there **rises by `hover_m`** rather than holding, because the likeliest reason the tag is unseen is the gripper hovering over it; `descend`/`lower` steer to the remembered pose and held phases pin the cube to the gripper. The per-action `meta["jev"]["held"]` describes the world the step was built with (the value passed to the perceiver), not the next step's flag, and `world[obj]["from_gripper"]` marks a pose inferred from the gripper so the scorer never credits it; on the first act after reset, `phase = machine.reset(world)` else `machine.advance(world)`. If `phase.name == "done"` → hold chunk whose single `Action.meta` carries `{"request_stop": True, "stop_reason": "task_done", "jev": {...}}` (the rollout reads `request_stop` from `Action.meta`, not chunk meta). If the target is missing from the world or `steps_since_seen > stale_after` and the phase is in `_STALE_GATED` (`approach`, `carry`) → a chunk that rises by `hover_m` (bounded by the box), `stalls += 1`, transcript entry with `"stall": True`; other phases never stall. Else `state = build_state(...)`, `menu = build_menu(...)`, `answer = client.choose(...)`, `move = parse_option(answer.choice)`, `chunk = mapper.chunk(move, start)`; append `menu[answer.choice].split(";")[0]` to history; transcript entry `{"step", "phase", "state", "instructions", "menu", "answer": {"choice", "probabilities", "confidence", "model", "usage"}, "actions": len(chunk.actions)}`; every `Action` in every returned chunk (including hold and stall chunks) carries `meta["jev"] = {"phase", "arm", "choice", "confidence", "held": bool, "world": {obj: {arm: [x, y, z], "seen_ago": int}}}` so the scorer can read the final world from `record.steps[-1].action.meta` (scoring runs before `on_trial_end`; approvers preserve `meta`). `on_trial_end` still writes the summary to `record.metadata["jev"]` for the log. `DecisionsError` propagates (core wraps it as `PolicyError`).

`info`: `PolicyInfo(name="jev", action_space=<placeholder Box until bind>)`; `bind` replaces `self.info` with the embodiment's action space, mirroring the agent plugin's embodiment-adaptive pattern.

- [ ] **Step 1: Failing tests** — with a `FakeClient` (scripted `ChoiceAnswer`s) and a `FakePerceiver` (scripted `WorldState`s), drive `bind` → `reset` → several `act`s: assert the first call resets the machine and asks with an `xyz` menu; the chosen move produces the expected chunk; history accumulates descriptions; stall path returns hold and increments stalls; `done` sets `request_stop`; `transcript()` is JSON-serialisable (`json.dumps`); `on_trial_end` writes metadata; `jev_policy(steps_cm="0.5,2")` parses; `bind` on a Box without labels raises.
- [ ] **Step 2–4: Run; implement (≤ 200 lines); run tests, ruff, mypy.**
- [ ] **Step 5: Commit** — `feat(jev): JevPolicy wiring perceiver, phases, menu, Decisions client, and motion`.

---

### Task 11: Scorer (`scorer.py`)

**Files:**
- Create: `src/inspect_robots_jev/scorer.py`, `tests/test_scorer.py`

**Interfaces:**
```python
def cube_in_bowl(*, cube: str = "cube", bowl: str = "bowl", radius_m: float = 0.04, max_above_m: float = 0.03, max_unseen: int = 3) -> Scorer
```
Reads `record.steps[-1].action.meta["jev"]` (scoring runs before `on_trial_end`, so `record.metadata` is not available; approvers preserve `Action.meta`). Success requires **all** of: the cube's tag was actually re-detected after release, i.e. `meta["held"] is False` and `world[cube]["seen_ago"] <= max_unseen` (default 3) — otherwise the last pose is the in-gripper estimate and would score a cube stuck in the jaws as "in the bowl"; and, in the arm frame used by the policy (`"left"` if present else `"right"`), horizontal distance ≤ `radius_m` and `cube_z − bowl_z ≤ max_above_m`. No steps or missing meta → `Score(value=False, explanation="no jev world state recorded")`. `name == "jev_cube_in_bowl"` (same as the entry point).

- [ ] **Steps 1–5:** tests for success, horizontal miss, cube left on the rim (too high), cube still held / not re-seen after release, missing metadata; implement (≤ 40 lines); commit `feat(jev): pose-based cube_in_bowl scorer`.

---

### Task 12: End-to-end through `eval()`

**Files:**
- Create: `tests/test_end_to_end.py`, `tests/_fake_rig.py`

`_fake_rig.py`: a minimal `Embodiment` with the 14-dim YAM-like absolute Cartesian action box (labels as in Task 7), `eef_state` observation, a `top_cam` image (zeros, 64×64×3), `top_cam_intrinsics`, a kinematic toy world where a "cube" and "bowl" have fixed positions, the gripper closes to 0.3 when within 1.5 cm of the cube, the cube follows the gripper while closed, `step()` always returns `terminated=False` so the trial can only end through the policy's `request_stop`, and a `FakeDetector` that reports tags at the true object positions (so the perceiver sees a consistent world). A scripted `FakeClient` that always picks the greedy-best option from the menu (computed from the state text's directional words, so the test also exercises the wording round trip).

- [ ] **Step 1:** write the test: `task = Task(name="jev-cube-in-bowl", scenes=[Scene(id="s0", instruction="put the cube in the bowl")], scorer=[cube_in_bowl()], max_steps=600)` (a plain list: `Task.scenes` is `Sequence[Scene]` and `ListSceneDataset` is not a `Sequence` under mypy strict; 600 ticks because a nominal run is ~150 ticks); `logs = eval(task, JevPolicy(config, client=fake, perceiver=Perceiver(...)), FakeRig(), log_dir=str(tmp_path))`; assert the log status is `"success"`, the trial ended by `request_stop` (`truncated` with `termination_reason` naming the policy stop), the `jev_cube_in_bowl` score is True, the policy transcript is present and non-empty, and every recorded action stayed within the box bounds. (`eval()` has no scorer argument; scorers live on the `Task`.)
- [ ] **Step 2–4:** run (expect failure until the fakes are right), fix, run the plugin's full suite with `--cov-fail-under=100`.
- [ ] **Step 5: Commit** — `test(jev): end-to-end eval on a fake Cartesian rig`.

---

### Task 13: Docs, live probe, changelog

**Files:**
- Create: `examples/jev_textsim.py` (repo root; the cleaned spike v3b: same menu/wording via the plugin's `build_menu`/`describe_offset`, 30 seeded episodes, prints success rate, exits 1 below 27/30). Mark clearly "requires OPENROUTER_API_KEY; costs < $0.01; not run in CI".
- Modify: `plugins/inspect-robots-jev/README.md` (install; `.env` keys; tag printing instructions from spec §5 with the table; calibration procedure: run the YAM pose CLI, touch the four table-tag corners with each gripper, paste into `corners.json`, run `python -m inspect_robots_jev.calibrate --corners corners.json --frame <png> --tags tags.json --out calibration.json`), `CLAUDE.md`, root `CHANGELOG.md` (`### Added` entry), root `CLAUDE.md` plugin list (one line).
- Create: `src/inspect_robots_jev/calibrate.py` (+ `tests/test_calibrate.py`): `main(argv)` that loads corners JSON `{"left": [[x,y,z]*4], "right": [[...]*4]}`, a frame saved as `.npy` (H×W×3 uint8; the README shows a two-line snippet that grabs one observation from the embodiment and `np.save`s `images["top_cam"]` and `extra["top_cam_intrinsics"]`), and an intrinsics `.npy`, runs the detector on the table tag, calls `solve_calibration`, writes the calibration JSON. No PNG decoding (stdlib has none; core `_pngenc` only encodes). Tests use the fake detector and `tmp_path` `.npy` files; missing table tag → exit code 2 with a message naming the tag id.

- [ ] **Steps:** write tests for `calibrate.main` (happy path, missing table tag → exit 2); implement; write docs; run the full plugin gates; commit `docs(jev): README, calibration CLI, live wording probe, changelog`.

---

### Task 14: Rig configuration note and run recipe (no code)

Add to the plugin README a "Running on the YAM rig" section:

```ini
# ~/.config/inspect-robots/config.ini additions for Jev runs (rig 6)
[embodiment.args]
control_interface = eef_pos
top_depth_serial = <D435 serial>      # hands the top slot to librealsense: intrinsics + depth
capture_width = 1920                  # NEW in inspect-robots-yam (companion PR): native capture size
capture_height = 1080
cam_width = 1920                      # no downscale: tags need every pixel
cam_height = 1080
gripper_max_step = 0.25               # a full close/open in 4 ticks instead of 20
```

**Companion change (separate PR on `robocurve/inspect-robots-yam`):** add
`capture_width`/`capture_height` to `YamConfig` (defaults 640/480, so no
behaviour change for existing rigs), thread them into
`_capture_proc.REALSENSE_CAPTURE_WIDTH/HEIGHT` and the OpenCV reader's
`CAP_PROP_FRAME_WIDTH/HEIGHT`, and scale intrinsics from the capture size.
Merge and release that before the first rig run.

and the command:

```bash
inspect-robots "put the cube in the bowl" --policy jev \
  -P tags=$HOME/jev/tags.json -P calibration=$HOME/jev/calibration.json --embodiment yam \
  --scorer jev_cube_in_bowl --max-steps 600
```

`YamConfig` accepts non-square `cam_width`/`cam_height` today (no square check in `config.py`), so nothing to verify there.

---

## Self-review

**Spec coverage:** §1 client → T3; §2 rules R1–R5 → T4 (hints), T6 (curated wording, history), T5/T10 (code owns phases and completion), T10 (confidence logged in transcript, not gated); §3 layout → T1–T11 file table; §3.1 data flow → T10; §3.1a phases → T5; §3.2/3.3 examples → T6/T4 tests pin them; §4 D1–D9 → T1 (plugin), T9/T10 (perceiver in policy), T5 (phases), T4/T10 (one Choice, arm in id), T7 (own splitter rather than import, since the agent plugin's `Toolset` is not a public API — D5 resolved as "copy"), T3 (two backends), T10 (transcript), T10/T11 (done + scorer), T9/T10 (occlusion, stale hold); §5 → T9 (tags, fusion, depth), T8/T13 (calibration by touch), T14 (camera config), spec `doctor` check deferred to a follow-up issue (noted here so it is not silently dropped); §6 → T7 (bounds), T5 (grasp gate), task `max_steps` (T14 recipe); §7 settings → T10 config; §8 testing → each task + T12; live probe → T13.

**Placeholder scan:** none of TBD/TODO/"handle edge cases". Tasks 5–9 give algorithms and exact interfaces with prose steps rather than full listings; the interfaces are complete and the tests are specified concretely.

**Critique round 1 (fresh-context subagent, 2026-09-19):** 6 major, 6 minor, 2 nits, all folded into this revision: scorer reads the last step's `Action.meta` because `eval()` scores before `on_trial_end`; `Task(scorer=...)` replaces the non-existent `eval(scorer=)`; tag-frame z points into the tag so face offsets are `+half`; YAM captures at a fixed 640 × 480 so a companion `capture_width/height` change is required (Task 14); positional goals are clamped into workspace bounds so the 0.03 m `z` floor cannot deadlock `descend`/`lower`; `ci-ok` needs, `uv.lock`, and the `pypi-jev` environment are called out; uint8 grayscale and single detector construction; `request_stop` on `Action.meta`; arm choice by base distance; motion test without `max_step`; serializer key renamed to `target_point_…` and thresholds sourced from `TaskConfig`; calibrate CLI reads `.npy`; `build_menu` signature and scorer name aligned.

**Critique round 2 (fresh-context subagent, 2026-09-19):** all round-1 fixes verified present in plan and code; 2 major, 6 minor, 3 nits, all folded in: `max_steps` counts control ticks, not decisions (§0, §6, Task 12 → 600, Task 14 → 600 + `gripper_max_step`); a `reopen` phase after closing on air (Task 5, §3.1a, serializer); `Task(scenes=[...])` not `ListSceneDataset`; scorer requires the cube to be re-seen after release (`held`/`seen_ago` in `meta["jev"]`); chunks interpolate from the last commanded target; `closed_on_air` in `TaskConfig`; `held` window and top-down occlusion note spelled out; `jev_policy` coercion rules; `pose_t.reshape(3)`; `$HOME` in the recipe; `FakeRig` never terminates.

**Code review round 2 (fresh eyes, 2026-09-19):** 3 major, 2 minor, all folded in: `retreat → done` could fire while the gripper still hid the cube (scorer false negative) → new `verify` phase; `lower`/`descend` deadlocked on contact when the computed goal was unreachable → contact rule + command/measured resync; the e2e fake rig saw through the gripper → occlusion and shallow-bowl models, e2e parametrised over four rigs; stalls in `approach`/`carry` now rise instead of hold; held window wording aligned.

**Type consistency:** `Move(arm, axis, delta_m, gripper)` used identically in T4/T7/T10; `WorldState.offset/steps_since_seen` in T5/T6/T10; `Phase(name, arm, target, menu_kind, goal)` in T5/T6/T10; `ChoiceAnswer` fields in T3/T10; `Calibration.to_arm` in T8/T9; `GripperView(position, opening)` in T2/T5/T9/T10.
