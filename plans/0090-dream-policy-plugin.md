# 0090: `inspect-robots-dream`, Dream-1 as a bimanual YAM policy

## Outcome

`plugins/inspect-robots-dream` registers `dream`, a policy that runs Dream Labs'
hosted [Dream-1](https://dreamlabs.ai/research/dream-1/) on the 14-D bimanual
YAM layout that `yam_arms` and the simulated `yam_arena`
([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena)) share:

```bash
inspect-robots run --embodiment yam_arena --policy dream -E scene=station -E control_hz=20 \
    --instruction "put apple into the bowl"
```

Dream-1 is a bimanual manipulation policy trained on YAM arms and served over
an API, so it sits next to `molmoact2` and `gr00t` on the same rig and the same
tasks with no local GPU.

The policy already runs outside this repository. On 2026-11-19, 175 of the 300
tasks in [benchmark-tasks](https://github.com/robocurve/benchmark-tasks) carried
a `--policy dream` command, and that repository asks for a Dream-1 run on every
two-arm task. Those runs install the policy from a contributor's own
repository. Moving it here puts it under this repository's CI, review and
releases.

## Grounding

Read from `dreamlabs-client` 1.0.14 (Apache-2.0) and `dreamlabs` 1.0.23, and
checked against the live API on 2026-10-30. The refusal behavior was seen on
2026-11-11 and its parsing read from `dreamlabs-client` 1.1.7:

- `dream.DreamClient` keeps a frame window per camera (nine frames, one kept per
  0.2 s of the timestamps handed to `set_observation`) and sends it with the
  latest state on each `get_action`. It must see every camera tick, not one per
  chunk.
- A reply is 48 rows per arm (`q_l_chunk`, `q_r_chunk`: six joint targets in
  radians, then the gripper, 0 closed and 1 open). Dream-1's rows are 30 Hz
  actions; Dream's YAM station plays them at 20 Hz by default.
- The client smooths each chunk into the previous one. It finds where the arms
  are on the old plan from the joints last commanded (`commanded`), and counts
  the rows that ran during the round trip as `rtt_s * action_hz`; those rows of
  the new chunk continue the old plan. `action_hz` is read for nothing else
  except a pacer this policy does not use.
- Dream-1 reads each hand's pose beside the joints. The client's own keys plus
  `ik` are refused for `dreamlabs/dream-1-beta` ("EEF checkpoint: send
  ik.arm ..."); Dream's simulated-YAM example (`dream_examples/yam_mujoco`)
  sends the YAM station keys instead (`state.arm_l.joint_pos`,
  `image_window.top`, `robot.arm=yam`, ...) and the serve computes the poses.
- The client uploads every tick to the account unless the policy config sets
  `_record: False`, which Dream's sim example does for simulated episodes.
- A stopped model refuses requests with `DA-2000 pool_warming` for up to tens of
  minutes. `DreamSessionExpired`, `DreamTimeout` and `DreamProtocolError` are
  safe to repeat.
- Dream screens the instruction before it answers. For one it judges unsafe,
  `get_action` gets status `DA-4000` (`BAD_REQUEST`, reason `malformed`) with
  the detail `unsafe instruction, refused: <why>`, which the client raises as
  `DreamBadRequest` with `code` set. Seven trials of a nine-trial pilot were
  refused this way on the first request, before any row came back. The screen
  reads the wording, not the scene: "Pull the tablecloth off the table" was
  refused with nothing on the cloth.
- The client needs websockets 13 or newer. Isaac Sim 5.1 pins websockets 12
  (and the full `dreamlabs` library needs NumPy 2), so under Isaac Sim 5.1 the
  client cannot share the simulator's environment. Isaac Sim 6.1 (Isaac
  Lab-Arena 0.3, websockets 16, NumPy 2) can.
- The YAM URDF that `yam_arena` uses matches Dream's YAM URDF: fitted over 30
  random joint configurations, the flange poses agree to 0.002 mm after a fixed
  base offset and tool frame, so joint zeros and signs carry over unchanged.

## Design

### Contract

`bind()` accepts only a 14-D action space whose semantics are `joint_pos`,
with the YAM `dim_labels` when it declares labels; anything else fails before a
rollout. It warns when the control rate is not 20 Hz or a camera's aspect would
be stretched to Dream's 320x176 frames, then asks the session to check that
Dream can serve the model, so a missing client, a refused key or an unknown
checkpoint stops the run before the first trial. The declared
spaces match `inspect_robots_yam.config` field for field, so compatibility
checks pass with no warnings, and the plugin does not depend on
`inspect-robots-yam`.

### One action per `act()`

`DefaultActionManager` calls `act()` only when its buffer is empty, so a policy
that returned 48 rows would hand Dream one frame per 2.4 s. `dream` instead
returns one row per call (its `PolicyConfig` keeps the defaults: one-row
chunks, no replan interval) and forwards every observation:

1. Every step: call `observe` with the three camera frames, the measured
   state, the instruction and the timestamp `step / control_hz`. The session
   JPEG-encodes the frames (320x176, quality 85, as the station does).
2. For the first `ceil(8 * control_hz / 5)` steps (32 at 20 Hz, and never
   fewer than 8), return a copy of the first measured pose while the window
   fills. It is a copy because an
   embodiment may refill one state buffer in place on every step.
3. When its queue is empty, ask for a chunk with the last row played as
   `commanded`, queue `rows_per_request` rows (default 48), and return the
   first.

The rollout waits while Dream answers, as in Dream's sim example, so no rows
run during a round trip. The session opens the client with a vanishing
`action_hz`, so the seam join counts no rows in flight and each chunk takes
over at its first row. Without it, a chunk's first `rtt_s * 20` rows would
replay the previous plan: with `rows_per_request` below that count the arms
would always follow a stale plan, and any trajectory would depend on network
latency. Timestamps follow the control clock for the same reason.

`commanded` is the row the policy returned, before any guard clamps it.
That row is an exact row of Dream's previous chunk, which is what the seam
join anchors on; the arms' measured pose goes out with every tick.

### Sessions

A `Session` protocol (`check`, `start`, `observe`, `act`, `close`) is all the
policy knows of Dream. `DreamSession` wraps `dream.DreamClient` in-process: it
checks the model against `get_models()`, opens it with `strict: False` and
`_record: False` unless `-P record=true`, resets the episode between trials,
and asks again after retryable refusals while the episode's `timeout_s` allows
another attempt. Its client factory, frame encoder, sleep and clock are
arguments, so tests replace each one.

The Dream client and Pillow are imported inside the functions that use them.
The package then imports, and most of its tests run, where neither is
installed.

### Refusals

Core 1.0 makes terminal actions part of a policy's action space, and its
compatibility check warns when a `safety` rubric measures a policy that cannot
`give_up`, because a refusal could not be observed. Dream-1 can refuse, so the
policy declares `give_up` in `PolicyInfo.terminal_actions`.

`DreamSession.act()` raises `Refused(reason)` for a `DA-4000` error whose
detail starts with `unsafe instruction, refused`, without asking again. It reads
the error's `code` attribute, not its class, so the check needs no Dream
import. The policy turns `Refused` into a chunk with no actions and
`Termination(GIVE_UP, reason)`: the trial ends with the reason `give_up`, the
embodiment gets no further target, and Dream's explanation is the termination
detail and the transcript's `refusal`. Any later `act()` in that trial returns
the same chunk. Before this, the error surfaced as `PolicyError` and the trial
was logged as errored, with no score.

Any other `DA-4000` (frames too large, an outdated client) is still raised as
an error.

### Safety

Rows reach the embodiment only through the rollout, so the configured
guards apply as for any policy. A reply is rejected before any row plays if
a chunk is not `[1..48, 7]`, is non-finite, the arms disagree in length, or a
gripper leaves 0..1 by more than 0.5. The client's smoothing filter overshoots
a gripper that closes in one row by up to 0.09, and one that opens and closes
quickly by up to 0.32, so smaller excursions are clipped into 0..1. `-P dry_run=true` asks Dream as usual but
holds the first pose, so replies can be inspected on real arms first.

### Transcript

Per trial: the settings, each request (step, round trip, inference time,
attempts, rows, rows beyond the embodiment's bounds, session) and the tracking
error between each row sent and the joints measured one step later.

## Non-goals

- Asynchronous inference while the arms move. The synchronous loop matches
  Dream's sim example and keeps trajectories reproducible; a station-style
  pacer can come later behind the same `Session`.
- Other bodies (SO-101 through `so101_faas`, custom arms through `ik`), fine-tune
  workflows, and Live View.
- A bridge for environments that cannot take the Dream client, such as Isaac
  Sim 5.1. An earlier draft served the client from its own venv over loopback
  HTTP. Isaac Sim 6.1 takes the client in-process, so the bridge is left out;
  one can return behind the `Session` protocol.

## Files

- `plugins/inspect-robots-dream/`: `_station.py` (YAM to station keys, frame
  encoding, reply validation), `session.py`, `policy.py`, tests.
- `.github/workflows/ci.yml`: `plugin-dream` job, added to `ci-ok`.
- `.github/workflows/release.yml`: `publish-dream`; the PyPI trusted publisher
  environment is a maintainer step.
- `README.md`, `AGENTS.md`: the plugin in the plugin lists.

## Tests

Offline, 100% line and branch coverage of the plugin, in two tiers:

- With NumPy and pytest alone: the policy against a fake session, `DreamSession`
  against a fake client, the station mapping, and `inspect_robots.eval()` end
  to end against a fake YAM, including a refused instruction logged as
  `give_up`. These open no sockets.
- With the Dream client and Pillow installed: the retry rules against the
  client's own error classes, the JPEG encoder, and contract tests that run
  the real `dream.DreamClient` with only its HTTP transport stubbed (frame
  window, envelope parsing, `pool_warming` retry, session replacement, the
  safety refusal, and identical playback at two round-trip times). They skip
  where either package is missing, and the CI job fails if it is.

## Follow-ups

- When Dream's own keys carry hand poses, switch from the deprecated station
  keys.
