# 0087: `inspect-robots-dream`, Dream-1 as a bimanual YAM policy

## Outcome

`plugins/inspect-robots-dream` registers `dream`, a policy that runs Dream Labs'
hosted [Dream-1](https://dreamlabs.ai/research/dream-1/) on the 14-D bimanual
YAM layout that `yam_arms` and `yam_sim` share:

```bash
inspect-robots run --embodiment yam_sim --policy dream -E control_hz=20 \
    --instruction "put the red cube on the plate"
```

Dream-1 is a bimanual manipulation policy trained on YAM arms and served over
an API, so it sits next to `molmoact2` and `gr00t` on the same rig and the same
tasks with no local GPU.

## Grounding

Read from `dreamlabs-client` 1.0.14 (Apache-2.0) and `dreamlabs` 1.0.23, and
checked against the live API on 2026-10-30:

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
- The client needs websockets 13 or newer. Isaac Sim 5.1 pins websockets 12
  (and the full `dreamlabs` library needs NumPy 2), so under Isaac Sim 5.1 the
  client cannot share the simulator's environment. Isaac Sim 6.1 (Isaac
  Lab-Arena 0.3, websockets 16, NumPy 2) can.
- `yam_sim`'s URDF matches Dream's YAM URDF: fitted over 30 random joint
  configurations, the flange poses agree to 0.002 mm after a fixed base offset
  and tool frame, so joint zeros and signs carry over unchanged.

## Design

### Contract

`bind()` accepts only a 14-D action space whose semantics are `joint_pos`,
with the YAM `dim_labels` when it declares labels; anything else fails before a
rollout. It warns when the control rate is not 20 Hz or a camera's aspect would
be stretched to Dream's 320x176 frames, then asks the session to check that
Dream can serve the model, so a missing client, a down bridge, a refused key or
an unknown checkpoint stops the run before the embodiment starts. The declared
spaces match `inspect_robots_yam.config` field for field, so compatibility
checks pass with no warnings, and the plugin does not depend on
`inspect-robots-yam`.

### One action per `act()`

`DefaultController` calls `act()` only when its buffer is empty, so a policy
that returned 48 rows would hand Dream one frame per 2.4 s. `dream` instead
returns one row per call (its `PolicyConfig` keeps the defaults: one-row
chunks, no replan interval) and forwards every observation:

1. Every step: JPEG-encode the three cameras (320x176, quality 85, as the
   station does) and call `observe` with the measured state, the instruction
   and the timestamp `step / control_hz`.
2. For the first `ceil(8 * control_hz / 5)` steps (32 at 20 Hz), return the
   first measured pose while the window fills.
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

`commanded` is the row the policy returned, before any approver clamps it.
That row is an exact row of Dream's previous chunk, which is what the seam
join anchors on; the arms' measured pose goes out with every tick.

### Sessions

A `Session` protocol (`check`, `start`, `observe`, `act`, `close`) hides where
the client runs. `DreamSession` wraps `dream.DreamClient` in-process: it checks
the model against `get_models()`, opens it with `strict: False` and
`_record: False` unless `-P record=true`, resets the episode between trials,
and asks again after retryable refusals while the episode's `timeout_s` allows
another attempt.

`BridgeSession` speaks JSON over loopback HTTP to `inspect-robots-dream-bridge`,
a stdlib server that runs a `DreamSession` in its own venv: `POST /check`,
`/start`, `/observe`, `/act` and `GET /health`. Requests reach the Dream client
one at a time under a lock, on a threaded server so an idle connection cannot
stall it. `/start` returns an episode token that `/observe` and `/act` must
carry, so a second run on the same bridge stops the first with an error
instead of mixing frames. A bridge that does not answer raises
`ConnectionError`, so the rollout's connection hint names `server_url` and the
policy's `remedy`; a refusal raises `BridgeError` with the bridge's error.
`/act` waits for the episode's timeout plus one full Dream request, the longest
the bridge may keep retrying. `--warm` asks for one chunk on a still scene
before serving.

### Safety

Rows reach the embodiment only through the rollout, so the configured
approvers apply as for any policy. A reply is rejected before any row plays if
a chunk is not `[1..48, 7]`, is non-finite, the arms disagree in length, or a
gripper leaves 0..1 by more than 0.05. `-P dry_run=true` asks Dream as usual but
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
- Launching the bridge from the policy.

## Files

- `plugins/inspect-robots-dream/`: `_station.py` (YAM to station keys, frame
  encoding, reply validation), `session.py`, `bridge.py`, `policy.py`, tests.
- `.github/workflows/ci.yml`: `plugin-dream` job, added to `ci-ok`.
- `.github/workflows/release.yml`: `publish-dream`; the PyPI trusted publisher
  environment is a maintainer step.
- `README.md`, `AGENTS.md`: the plugin in the plugin lists.

## Tests

Offline, 100% line and branch coverage of the plugin: fakes for the client,
a real loopback bridge (episode ownership, an idle connection, a silent or
foreign server), `inspect_robots.eval()` end to end against a fake YAM
in-process and through the bridge, and contract tests that run the real
`dream.DreamClient` with only its HTTP transport stubbed: frame window,
envelope parsing, `pool_warming` retry, session replacement, and identical
playback at two different round-trip times.

## Follow-ups

- When Dream's own keys carry hand poses, switch from the deprecated station
  keys.
- `yam_sim`'s overhead camera sits lower and looks less steeply than the station
  camera Dream was trained with; a calibrated variant would make sim results
  more comparable.
