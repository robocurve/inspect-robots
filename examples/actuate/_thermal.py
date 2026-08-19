"""Read Actuate motor temperatures under a strict idle-loop safety contract.

Probe only while the demo loop is idle, never during an eval. Healthy motors
finish enabled at zero torque, the same state as normal eval teardown. Two
caveats apply: ``motor_on`` clears firmware fault latches with ``clean_error``
on any faulted motor, so probing is not state-neutral for a faulted motor. A
constructor failure during bringup or a parent timeout's SIGKILL also skips a
deterministic socket close. This is harmless because child exit closes the
socket and SocketCAN is non-exclusive.
"""

from __future__ import annotations

import configparser
import json
import subprocess
import sys
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

GRIPPER_TYPE = "LINEAR_4310"  # GripperType enum NAME; look up via GripperType[GRIPPER_TYPE]
TEMP_WAIT_C = 60.0  # hottest reading at/above this suspends evals
TEMP_RESUME_C = 50.0  # once suspended, resume only below this
COOL_POLL_S = 30  # re-probe interval while cooling
PROBE_TIMEOUT_S = 20  # child-process deadline per probe
COOL_PROBE_ERRORS_GIVE_UP = 5  # consecutive probe errors while cooling
FALLBACK_CHANNELS = ("can_left", "can_right")  # used only if RIG_CONFIG has no channels


class ThermalProbeError(RuntimeError):
    """Report a failed motor-temperature probe with a short reason."""


class ThermalProbeUnavailable(ThermalProbeError):
    """Report that the probe child could not import i2rt."""


@dataclass(frozen=True)
class MotorTemp:
    """One motor's MOSFET and rotor temperatures."""

    channel: str
    motor_id: int
    temp_mos: float
    temp_rotor: float

    @property
    def max_c(self) -> float:
        """Return the hotter of the motor's two temperature readings."""
        return max(self.temp_mos, self.temp_rotor)

    @property
    def label(self) -> str:
        """Return the short channel and hexadecimal motor identifier."""
        return f"{self.channel} 0x{self.motor_id:02x}"


def config_channels(config_path: Path) -> tuple[str, str]:
    """Read both CAN channels from a rig config, with safe fallbacks."""
    config = configparser.ConfigParser(
        inline_comment_prefixes=("#",),
        interpolation=None,
    )
    try:
        config.read(config_path)
        args = config["embodiment.args"]
        return args["left_channel"], args["right_channel"]
    except (configparser.Error, KeyError):
        return FALLBACK_CHANNELS


def read_motor_temps(channels: Iterable[str]) -> list[MotorTemp]:
    """Run the deadline-bound probe child and parse its temperature records."""
    try:
        completed = subprocess.run(
            [sys.executable, str(__file__), "--probe", *channels],
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as error:
        raise ThermalProbeError(f"probe timed out after {PROBE_TIMEOUT_S}s") from error
    except OSError as error:
        raise ThermalProbeError(f"could not start probe: {error}") from error

    if completed.returncode == 3:
        raise ThermalProbeUnavailable("probe child could not import i2rt")
    if completed.returncode != 0:
        raise ThermalProbeError(f"probe child exited {completed.returncode}")

    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list):
            raise TypeError("probe output is not a list")
        return [
            MotorTemp(
                channel=str(item["channel"]),
                motor_id=int(item["motor_id"]),
                temp_mos=float(item["temp_mos"]),
                temp_rotor=float(item["temp_rotor"]),
            )
            for item in payload
        ]
    except Exception as error:
        raise ThermalProbeError(f"invalid probe output ({type(error).__name__})") from error


def hottest(temps: Iterable[MotorTemp]) -> MotorTemp:
    """Return the motor with the hottest MOSFET or rotor reading."""
    return max(temps, key=lambda temp: temp.max_c)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--probe":
        print(f"usage: {Path(__file__).name} --probe <channel> [<channel> ...]", file=sys.stderr)
        raise SystemExit(2)

    try:
        from i2rt.motor_drivers.dm_driver import DMChainCanInterface
        from i2rt.robots.utils import ArmType, GripperType, _load_arm_config
    except ImportError:
        raise SystemExit(3) from None

    try:
        arm_type = ArmType.YAM
        motor_list = [list(motor) for motor in _load_arm_config(arm_type).motor_list]
        motor_list.append([0x07, GripperType[GRIPPER_TYPE].get_motor_type(arm_type)])
        offsets = [0.0] * len(motor_list)
        directions = [1.0] * len(motor_list)
        temperatures = []
        for channel in sys.argv[2:]:
            chain = None
            try:
                chain = DMChainCanInterface(
                    motor_list,
                    offsets,
                    directions,
                    channel,
                    start_thread=False,
                    motor_chain_name=f"thermal-probe-{channel}",
                )
                temperatures.extend(
                    {
                        "channel": channel,
                        "motor_id": state.id,
                        "temp_mos": state.temp_mos,
                        "temp_rotor": state.temp_rotor,
                    }
                    for state in chain.read_states()
                )
            finally:
                if chain is not None:
                    chain.close()
        print(json.dumps(temperatures))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
