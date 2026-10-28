# 0076: Actuate demo thermal gate

Demo-branch only (`demo/actuate-conference`, PR #397; never merges to main).
All paths below are relative to `examples/actuate/` unless noted.

## Problem

The demo loop (`run.py`) runs evals back to back with fixed pauses (10 s, 30 s
after a failure) and no awareness of motor temperature. The only thermal
protection is the DM motor firmware's own over-temperature fault (error codes
0xB MOSFET / 0xC rotor), which the pinned i2rt (`ac096928`,
`enable_auto_recovery=False`) surfaces by failing the episode. At the booth
that failure mode is ugly: the eval dies mid-audience, the loop waits 30 s,
then relaunches against the still-hot motor, up to 10 times before the
log-less-streak halt fires. There is no "wait until cooled" state anywhere.

## Goal

Before each eval, read every motor's temperature. If any motor is at or above
a wait threshold, suspend the loop (arms stay parked and torque-off), show a
cooling state on the console and the conference monitor, and start the next
eval only after the hottest motor drops below a lower resume threshold
(hysteresis). Degrade gracefully on machines without hardware.

## How temperatures are read (the probe)

Verified against the installed i2rt (`ac096928`) and yam plugin:

- The yam `BimanualDriver` protocol does not expose temperatures, and a full
  `get_yam_robot()` bringup is unsuitable for a probe: it costs a multi-second
  gripper auto-calibration (physical motion) plus a PD hold.
- `i2rt.motor_drivers.dm_driver.DMChainCanInterface(motor_list, offsets,
  directions, channel, start_thread=False, motor_chain_name=...)` is the
  light path. Its constructor enables each motor (`motor_on`), and each enable
  reply is a feedback frame carrying `temperature_mos` / `temperature_rotor`.
  With `start_thread=False` no control loop starts and no torque is ever
  commanded. `read_states()` then returns `MotorInfo` records with
  `temp_mos` / `temp_rotor` from those frames.
- End state for healthy motors is identical to normal eval teardown:
  `MotorChainRobot.close()` also leaves motors enabled at zero torque and
  closes the CAN socket (no `motor_off` anywhere in the stack). Two caveats
  the module docstring must state honestly: (a) `motor_on` clears firmware
  fault latches (`clean_error`) on any motor holding one, so a probe is not
  state-neutral for a faulted motor; (b) if the chain constructor raises
  mid-bringup the CAN socket is not deterministically closed on that path,
  and a parent-side timeout SIGKILL likewise skips the child's `finally`
  close (both harmless: the probe child process exits and the OS closes the
  socket, and SocketCAN is non-exclusive).
- **The probe must run in a child process with a deadline.** i2rt's
  `motor_on` contains an unbounded fault-clean loop: a motor latched in an
  over-temperature fault that is still hot re-asserts the fault after every
  `clean_error`, and the loop spins forever with logging suppressed. That is
  the single most likely state for the gate's first probe after a thermal
  incident, and it hangs rather than raises, so no exception handler can
  catch it. Therefore the in-process probe code runs only in a child:
  `run.py` invokes `sys.executable _thermal.py --probe <channels...>` via
  `subprocess.run(..., timeout=PROBE_TIMEOUT_S)`; a timeout kills the child
  and counts as an ordinary probe error.
- Motor list: `_load_arm_config(ArmType.YAM).motor_list` (6 arm motors, CAN
  IDs and types exactly as the real bringup uses) plus the gripper motor
  `[0x07, GripperType.LINEAR_4310.get_motor_type(ArmType.YAM)]`. Reusing
  `_load_arm_config` (private but stable at the pinned commit) beats
  hardcoding IDs that would drift from i2rt.
- The probe runs only between evals, when no other process owns the CAN
  socket. Offsets/directions do not matter for temperature (pass zeros/ones).

## Changes

### New file: `_thermal.py`

Booth-editable constants at the top, `_roster.py` style:

```python
GRIPPER_TYPE = "LINEAR_4310"      # GripperType enum NAME; look up via GripperType[GRIPPER_TYPE]
TEMP_WAIT_C = 60.0                # hottest reading at/above this suspends evals
TEMP_RESUME_C = 50.0              # once suspended, resume only below this
COOL_POLL_S = 30                  # re-probe interval while cooling
PROBE_TIMEOUT_S = 20              # child-process deadline per probe
COOL_PROBE_ERRORS_GIVE_UP = 5     # consecutive probe errors while cooling
FALLBACK_CHANNELS = ("can_left", "can_right")  # used only if RIG_CONFIG has no channels
```

No `CAN_CHANNELS` constant: channel names come from the rig config (rig-1's
`config.ini` has `left_channel = can_left` / `right_channel = can_right`
under `[embodiment.args]`; a hardcoded `can0`/`can1` default would make the
gate silently inert on the very rig it targets). `FALLBACK_CHANNELS` matches
the real rig, not the i2rt defaults.

API:

- `class ThermalProbeError(RuntimeError)`: any failed probe (child nonzero
  exit, timeout, unparseable output). Carries a short reason string.
- `class ThermalProbeUnavailable(ThermalProbeError)`: the child could not
  import i2rt (dev machine without the driver; distinct child exit code 3).
  The caller disables the gate for the session.
- `config_channels(config_path) -> tuple[str, str]`: read
  `left_channel`/`right_channel` from `[embodiment.args]` with
  `configparser` (`inline_comment_prefixes=("#",)` — the ini carries inline
  comments — and `interpolation=None`), falling back to `FALLBACK_CHANNELS`
  on any missing file, section, or key, and on `configparser.Error` (a
  malformed config must degrade to the fallback, not crash `run.py` at
  startup).
- `read_motor_temps(channels) -> list[MotorTemp]` where `MotorTemp` is a
  small frozen dataclass: `channel: str`, `motor_id: int`, `temp_mos: float`,
  `temp_rotor: float`, plus a `max_c` property and a short `label` like
  `"can_left 0x03"`. **Parent-side**: runs
  `subprocess.run([sys.executable, str(__file__), "--probe", *channels],
  capture_output=True, timeout=PROBE_TIMEOUT_S)` and parses one JSON list
  from the child's stdout. Timeout, nonzero exit, or bad JSON raise
  `ThermalProbeError` (exit code 3 raises `ThermalProbeUnavailable`). Only
  `except Exception`-level errors are ever translated; `KeyboardInterrupt`
  propagates untouched.
- `hottest(temps) -> MotorTemp`: max by `max_c`.
- **Child side** (`if __name__ == "__main__":` with `--probe <channels...>`):
  imports i2rt (exit 3 on ImportError), builds the motor list as above, and
  per channel constructs `DMChainCanInterface(..., start_thread=False,
  motor_chain_name=f"thermal-probe-{channel}")`, collects `read_states()`,
  closes the chain in a `finally`, prints the JSON list, exit 0. Any other
  exception: traceback to stderr, exit 1. A wedged fault-clean loop is
  handled by the parent's timeout kill, not in the child.

Module docstring states the safety contract: probe only with the loop idle
(never during an eval), the enable-at-zero-torque end state for healthy
motors, and the two caveats above (fault-latch clearing; constructor-failure
socket lifetime).

### `run.py`: the gate

New function `_thermal_gate(eval_index, gate_state)` called at the **top of
each loop iteration**, before roles are drawn and before the eval's
`_write_status` (this also covers the first eval after a relaunch on a hot
rig). Behavior:

Channels are resolved once at startup with `config_channels(RIG_CONFIG)`
(after the existing `RIG_CONFIG.is_file()` check) and passed to every probe.

1. If the gate is disabled (prior `ThermalProbeUnavailable`), return.
2. Probe. On `ThermalProbeUnavailable`: print one warning, disable the gate
   for the rest of the session, return. On `ThermalProbeError` (which a dead
   single channel also produces — one dead channel forfeits the whole
   round's gate, an intentional trade-off since a dead channel is fatal to
   the eval anyway): print a warning and return (skip this round; a
   genuinely broken rig is already handled by the eval itself and the
   existing `FAILURE_STREAK_STOP`). Probe errors never count toward the
   failure streak, and every handler is `except <specific>` — never bare or
   `BaseException` — so `KeyboardInterrupt` always reaches main's existing
   handler.
3. If `hottest < TEMP_WAIT_C`: print a one-line summary
   (`motor temps ok: max 41.2C (can_left 0x03)`) and return.
4. Otherwise enter the cooling loop: each round, write `status.json` with a
   `cooling` payload (below), print the same line to the console, sleep
   `COOL_POLL_S`, re-probe. Exit when `hottest < TEMP_RESUME_C`. While
   cooling, a `ThermalProbeError` prints a warning and counts a
   consecutive-error streak; after `COOL_PROBE_ERRORS_GIVE_UP` consecutive
   errors, print loudly and proceed anyway (a dead CAN would otherwise
   suspend the show forever, and the eval's own fail-fast plus the
   failure-streak halt cover a truly broken rig). A successful probe resets
   that streak. No overall time cap: staying suspended is the correct
   behavior while temperatures actually read hot; the operator can Ctrl-C
   (the existing KeyboardInterrupt handler already exits cleanly from any
   sleep, and the probe's subprocess timeout means no probe can wedge the
   loop).

Cooling `status.json` payload (written wholesale, like the existing status
writes; no `roles`/`models` since none are drawn yet):

```json
{"eval_index": N, "cooling": {"max_c": 63.4, "hottest": "can_left 0x03",
 "wait_c": 60.0, "resume_c": 50.0, "since": "<iso8601>"}}
```

The next eval's normal status write replaces this file, clearing the state.
Known cosmetic edge, accepted: if the loop is killed mid-cooling with an
empty `results.jsonl`, `_next_eval_index`'s status-file fallback resumes at
N+1, skipping the never-started N in the numbering.

### `serve.py`

`_status_payload()` gains a `"cooling": None` default and copies
`status.get("cooling")` through, same pattern as the existing keys.

### `monitor.html`

When `status.cooling` is present, the current-eval section shows a cooling
banner in place of the role draw: text like
`Cooling motors: 63.4C at can_left 0x03. Evals resume below 50C.` styled with the
page's existing muted/accent tokens (no new layout regions; hide it when the
field is absent). Keep the change minimal; visual taste is reviewed by the
main session, not delegated.

### `README.md` (demo README)

Booth-checklist bullet: the loop checks motor temperatures before every eval;
at/above 60 C it suspends and shows the cooling banner, resuming below 50 C;
thresholds live in `_thermal.py`, channel names come from the rig config;
the gate only exists when `run.py` is launched with a python that has i2rt
installed (the rig venv) — on machines without i2rt or CAN it disables
itself with a console warning and the demo runs as before. Also note the
probe clears any latched motor fault codes as a side effect of reading.

## Invariants retained

- The eval command line, role draw, scoring, results append, clip rendering,
  failure-streak halt (`FAILURE_STREAK_STOP = 10`), and the 10 s / 30 s pause
  semantics are unchanged; the gate only inserts a wait before an eval
  starts, and thermal waiting never increments the failure streak.
- `status.json` remains write-wholesale via the existing atomic
  `_write_status` (tmp + `os.replace`).
- The probe never runs concurrently with an eval subprocess (it strictly
  precedes the launch, and its child process has exited — by success,
  failure, or timeout kill — before the eval starts), and for healthy motors
  it leaves the same enabled-zero-torque state normal teardown leaves.

## Out of scope

- Mid-eval temperature monitoring (would need CAN sharing with the live
  control loop or a yam-plugin channel; the firmware fault plus fail-fast
  already cover mid-eval overheating).
- Exposing temperatures through the yam plugin/health CLI (upstream work, not
  demo-branch work).
- Tests: `examples/` has no test suite on this branch; correctness is gated
  by ruff plus on-rig validation.

## Validation

- `ruff check examples/actuate/` and `ruff format --check` pass.
- Dev machine (no i2rt/CAN): `run.py` starts, prints the gate-disabled or
  probe-error warning once, and proceeds exactly as today.
- On the rig: run one probe standalone
  (`python examples/actuate/_thermal.py --probe can_left can_right`) and
  confirm plausible temperatures on all 14 motors; then lower
  `TEMP_WAIT_C` below ambient to force the cooling path and confirm the
  console line, the monitor banner, and that raising it back resumes the
  draw. Confirm the next eval's bringup works normally after a probe.
