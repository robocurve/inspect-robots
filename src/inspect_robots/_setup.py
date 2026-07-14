"""Pure, IO-injected support for the ``inspect-robots setup`` wizard."""

from __future__ import annotations

import configparser
import os
import re
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import IO

from inspect_robots._defaults import _config_path, parse_value
from inspect_robots.conformance import DeviceSlot, device_slots

# Same minimal-ANSI convention as cli.py (#37): plain when piped or NO_COLOR.
_BOLD = "1"
_DIM = "2"
_CYAN = "36"
_GREEN = "32"
_YELLOW = "33"


def _paint(text: str, code: str, out: IO[str]) -> str:
    """ANSI-wrap ``text`` when ``out`` is an interactive terminal.

    Mirrors cli.py's ``_styled`` but tests the injected stream, so scripted
    test runs (StringIO) and piped output stay escape-free.
    """
    if os.environ.get("NO_COLOR") or not out.isatty():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


SUGGESTED: dict[str, str] = {
    "policy": "molmoact2",
    "embodiment": "yam_arms",
    "scorer": "success_at_end",
    "max_steps": "1200",
    "rerun": "true",
    "store_frames": "true",
}
CAM_ROLES: tuple[str, ...] = ("top", "left", "right")  # -> {role}_cam_device in [embodiment.args]
CAMERA_KEYS: tuple[str, ...] = tuple(f"{role}_cam_device" for role in CAM_ROLES)
V4L_BY_ID: Path = Path("/dev/v4l/by-id")
V4L_BY_PATH: Path = Path("/dev/v4l/by-path")
SYSFS_NET: Path = Path("/sys/class/net")
SERIAL_BY_ID: Path = Path("/dev/serial/by-id")

_DEFAULT_COMMENTS: dict[str, str] = {
    "policy": "from the inspect-robots-yam plugin",
    "embodiment": "same plugin; cameras configured below",
    "max_steps": "120 s at 10 Hz",
    "rerun": "live viewer of cameras/state/actions each run",
    "store_frames": "save each run's camera frames under logs/frames/",
}
_PROMPT_LABELS: dict[str, str] = {
    "policy": "policy",
    "embodiment": "embodiment",
    "scorer": "scorer",
    "max_steps": "max steps",
    "rerun": "live rerun viewer",
    "store_frames": "store camera frames",
}

_Validator = Callable[[str], bool]


def _valid_text(_value: str) -> bool:
    return True


def _valid_max_steps(value: str) -> bool:
    parsed = parse_value(value)
    return isinstance(parsed, int) and not isinstance(parsed, bool) and parsed >= 1


def _valid_bool(value: str) -> bool:
    return isinstance(parse_value(value), bool)


def _ask(
    prompt: str,
    default: str,
    validate: _Validator,
    constraint: str,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> str:
    """Ask for one value, accepting Enter as the displayed default."""
    while True:
        entered = input_fn(f"{prompt} [{_paint(default, _CYAN, out)}]: ").strip()
        value = entered if entered else default
        if validate(value):
            return value
        print(_paint(constraint, _YELLOW, out), file=out)


def _ask_yes_no(
    prompt: str,
    default: bool,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> bool:
    """Ask a conventional yes/no question, re-prompting unclear answers."""
    suffix = _paint("[Y/n]" if default else "[y/N]", _CYAN, out)
    while True:
        entered = input_fn(f"{prompt} {suffix} ").strip().lower()
        if not entered:
            return default
        if entered in ("y", "yes"):
            return True
        if entered in ("n", "no"):
            return False
        print(_paint("please answer yes or no", _YELLOW, out), file=out)


def _is_registered(kind: str, name: str) -> bool:
    """Resolve ``name`` against the registry without importing it eagerly."""
    from inspect_robots.registry import registered

    return name in registered(kind)


def _warn_unregistered(kind: str, name: str, out: IO[str]) -> None:
    """Warn about unavailable plugins the moment a name is accepted."""
    if not _is_registered(kind, name):
        print(
            _paint(
                f"'{name}' is not registered here — install its plugin, e.g. "
                "`uv pip install inspect-robots-yam`",
                _YELLOW,
                out,
            ),
            file=out,
        )


def _runtime_requirement_lines(defaults: Mapping[str, str]) -> list[str]:
    """Build setup checklist lines for registered configured components."""
    from inspect_robots.conformance import missing_runtime_requirements
    from inspect_robots.registry import registered

    lines: list[str] = []
    for kind in ("policy", "embodiment"):
        name = defaults[kind]
        factories = registered(kind)
        if name not in factories:
            continue
        for module, remedy in missing_runtime_requirements(factories[name]).items():
            lines.append(f"  ✗ {module} ({name}) → {remedy}")
    return lines


def _prompt_defaults(
    carried: dict[str, dict[str, str]],
    *,
    headless: bool,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> dict[str, str]:
    """Prompt for managed defaults, using valid raw config values first."""
    validators: dict[str, tuple[_Validator, str]] = {
        "policy": (_valid_text, ""),
        "embodiment": (_valid_text, ""),
        "scorer": (_valid_text, ""),
        "max_steps": (_valid_max_steps, "max_steps must be an integer >= 1"),
        "rerun": (_valid_bool, "rerun must be true or false"),
        "store_frames": (_valid_bool, "store_frames must be true or false"),
    }
    raw_defaults = carried.get("defaults", {})
    defaults: dict[str, str] = {}
    for key, suggestion in SUGGESTED.items():
        validate, constraint = validators[key]
        if key == "rerun" and headless:
            suggestion = "false"
            print(
                _paint(
                    "no display detected (SSH?): the rerun viewer cannot open here; "
                    "frames still record with store_frames",
                    _YELLOW,
                    out,
                ),
                file=out,
            )
        default = suggestion
        if key in raw_defaults:
            configured = raw_defaults[key]
            if validate(configured):
                default = configured
            else:
                print(
                    _paint(f"ignoring invalid {key} {configured!r} from config.ini", _YELLOW, out),
                    file=out,
                )
        value = _ask(
            _PROMPT_LABELS[key],
            default,
            validate,
            constraint,
            input_fn=input_fn,
            out=out,
        )
        defaults[key] = value
        if key in ("policy", "embodiment"):
            _warn_unregistered(key, value, out)
    return defaults


def _print_camera_listing(devices: list[str], directory: Path, out: IO[str]) -> None:
    print(f"Found {len(devices)} camera device(s) under {directory}:", file=out)
    for number, device in enumerate(devices, start=1):
        print(f"  {number}. {Path(device).name}", file=out)


def _identify_by_replug(
    role: str,
    devices: list[str],
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
    rescan: Callable[[], list[str]],
    noun: str = "camera device",
    plural_noun: str = "camera devices",
    retry_noun: str = "camera",
    unplug_label: str | None = None,
) -> str | None:
    """Identify one device by diffing the listing while it is unplugged."""
    label = unplug_label if unplug_label is not None else f"{role} camera"
    input_fn(f"Unplug the {label} now, then press Enter...")
    unplugged_devices = rescan()
    disappeared = [device for device in devices if device not in unplugged_devices]
    if not disappeared:
        print(
            _paint(f"no {noun} disappeared; unplug one {retry_noun} and try again", _YELLOW, out),
            file=out,
        )
        return None
    if len(disappeared) > 1:
        print(
            _paint(
                f"{len(disappeared)} {plural_noun} disappeared; unplug only one and try again",
                _YELLOW,
                out,
            ),
            file=out,
        )
        return None

    identified = disappeared[0]
    name = Path(identified).name
    print(f"That was: {_paint(name, _GREEN, out)}", file=out)
    input_fn("Plug it back in, then press Enter...")
    if identified not in rescan():
        input_fn(f"{name} was not detected; press Enter to rescan...")
        if identified not in rescan():
            print(
                _paint(
                    f"warning: {name} was still not detected; keeping the assignment", _YELLOW, out
                ),
                file=out,
            )
    return identified


def _device_slot_prompt(
    label: str,
    kind: str,
    devices: list[str],
    current: str | None,
    advertise_path_toggle: bool,
    *,
    out: IO[str],
    camera_fallback: bool = False,
) -> str:
    """Build one kind-aware device prompt with an Enter-accept current value."""
    choices = f"{label} — number, 'u' to identify by unplugging"
    if camera_fallback and advertise_path_toggle:
        choices += ", 'p' to switch listing"
    choices += ", 's' to skip"
    if not camera_fallback and kind == "v4l2" and advertise_path_toggle:
        choices += ", 'p' to switch listing"
    if current is None:
        return choices + ": "
    status = "current" if current in devices else "current, not detected"
    return f"{choices} [{_paint(f'{current} ({status})', _CYAN, out)}]: "


def _prompt_device_slot(
    label: str,
    kind: str,
    by_id_devices: list[str],
    by_path_devices: list[str],
    active_is_by_id: bool,
    by_id_dir: Path,
    by_path_dir: Path,
    current: str | None,
    assigned: dict[str, tuple[str, str, str]],
    advertise_path_toggle: bool,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
    rescan_by_id: Callable[[], list[str]],
    rescan_by_path: Callable[[], list[str]],
    camera_role: str | None = None,
) -> tuple[str | None, bool]:
    """Prompt for one slot and return its device plus active listing state."""
    while True:
        devices = by_id_devices if active_is_by_id else by_path_devices
        device_dir = by_id_dir if active_is_by_id else by_path_dir
        prompt = _device_slot_prompt(
            label,
            kind,
            devices,
            current,
            advertise_path_toggle,
            out=out,
            camera_fallback=camera_role is not None,
        )
        entered = input_fn(prompt).strip()
        selected: str | None = None
        if entered.lower() == "s":
            return None, active_is_by_id
        if entered.lower() == "p" and kind == "v4l2":
            active_is_by_id = not active_is_by_id
            devices = by_id_devices if active_is_by_id else by_path_devices
            device_dir = by_id_dir if active_is_by_id else by_path_dir
            _print_camera_listing(devices, device_dir, out)
            continue
        if entered.lower() == "u":
            nouns = {
                "v4l2": ("camera device", "camera devices", "camera"),
                "can": ("CAN interface", "CAN interfaces", "CAN interface"),
                "serial": ("serial device", "serial devices", "serial device"),
            }
            selected = _identify_by_replug(
                camera_role or label,
                devices,
                input_fn=input_fn,
                out=out,
                rescan=rescan_by_id if active_is_by_id else rescan_by_path,
                noun=nouns[kind][0],
                plural_noun=nouns[kind][1],
                retry_noun=nouns[kind][2],
                unplug_label=None if camera_role is not None else label,
            )
            if selected is None:
                continue
        elif not entered and current is not None:
            selected = current
        elif kind != "can" and (entered.startswith("/") or Path(entered).is_absolute()):
            # startswith("/") keeps POSIX rig paths accepted on Windows
            # workstations, where "/dev/v4l/..." is not drive-absolute.
            if not Path(entered).exists():
                print(
                    _paint(
                        f"warning: {entered} does not exist here "
                        "(ok if this config is for another machine)",
                        _YELLOW,
                        out,
                    ),
                    file=out,
                )
            selected = entered
        elif entered.isdigit():
            number = int(entered)
            if 1 <= number <= len(devices):
                selected = devices[number - 1]
        elif kind == "can" and entered and "/" not in entered:
            if entered not in devices:
                print(
                    _paint(
                        f"warning: {entered} is not in the scanned CAN interfaces "
                        "(ok if this config is for another machine)",
                        _YELLOW,
                        out,
                    ),
                    file=out,
                )
            selected = entered
        if selected is None:
            instruction = (
                "enter an interface number, bare interface name, 'u' to identify, or 's' to skip"
                if kind == "can"
                else "enter a device number, absolute path, 'u' to identify, or 's' to skip"
            )
            print(
                _paint(instruction, _YELLOW, out),
                file=out,
            )
            continue

        other = next(
            (
                (assigned_label, device)
                for assigned_kind, assigned_label, device in assigned.values()
                if assigned_kind == kind and device == selected
            ),
            None,
        )
        if other is not None:
            other_label, _device = other
            if camera_role is not None:
                warning_label = f"the {other_label} camera"
                question = f"Use {selected} for both {other_label} and {camera_role} cameras?"
            else:
                warning_label = other_label
                question = f"Use {selected} for both {other_label} and {label}?"
            print(
                _paint(
                    f"warning: {selected} is already assigned to {warning_label}",
                    _YELLOW,
                    out,
                ),
                file=out,
            )
            if not _ask_yes_no(
                question,
                False,
                input_fn=input_fn,
                out=out,
            ):
                continue
        return selected, active_is_by_id


def _camera_section(
    carried: dict[str, dict[str, str]],
    by_id_dir: Path,
    by_path_dir: Path,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> dict[str, str]:
    """Offer and collect camera assignments from both stable V4L listings."""
    by_id_devices = _scan_cameras(by_id_dir)
    by_path_devices = _scan_cameras(by_path_dir)
    active_is_by_id = bool(by_id_devices) or not by_path_devices
    devices = by_id_devices if active_is_by_id else by_path_devices
    device_dir = by_id_dir if active_is_by_id else by_path_dir
    advertise_path_toggle = len(by_path_devices) > len(by_id_devices)

    existing_args = carried.get("embodiment.args", {})
    default_enabled = bool(devices) or any(key in existing_args for key in CAMERA_KEYS)
    if not _ask_yes_no("Configure cameras?", default_enabled, input_fn=input_fn, out=out):
        return _preserve_managed_args(carried)

    if devices:
        _print_camera_listing(devices, device_dir, out)
    else:
        print(
            _paint("no /dev/v4l devices found (not Linux, or no cameras attached)", _YELLOW, out),
            file=out,
        )
    if advertise_path_toggle:
        print(
            _paint(
                f"only {len(by_id_devices)} by-id entries for "
                f"{len(by_path_devices)} detected cameras — identical cameras without serials "
                "collide there; by-path names are stable per physical USB port",
                _YELLOW,
                out,
            ),
            file=out,
        )

    while True:
        assignments: dict[str, str] = {}
        assigned_devices: dict[str, tuple[str, str, str]] = {}
        for role in CAM_ROLES:
            key = f"{role}_cam_device"
            selected, active_is_by_id = _prompt_device_slot(
                f"{role} camera",
                "v4l2",
                by_id_devices,
                by_path_devices,
                active_is_by_id,
                by_id_dir,
                by_path_dir,
                existing_args.get(key),
                assigned_devices,
                advertise_path_toggle,
                input_fn=input_fn,
                out=out,
                rescan_by_id=partial(_scan_cameras, by_id_dir),
                rescan_by_path=partial(_scan_cameras, by_path_dir),
                camera_role=role,
            )
            if selected is not None:
                assignments[key] = selected
                assigned_devices[key] = ("v4l2", role, selected)
        if len(assignments) in (0, len(CAM_ROLES)):
            return assignments
        print(
            _paint(
                "yam_arms needs all three cameras or none; writing none unless you go back",
                _YELLOW,
                out,
            ),
            file=out,
        )
        if not _ask_yes_no("Go back and choose cameras again?", True, input_fn=input_fn, out=out):
            return {}


def _preserve_managed_args(
    carried: dict[str, dict[str, str]],
    managed_args: tuple[str, ...] = CAMERA_KEYS,
) -> dict[str, str]:
    """Return existing assignments for the keys managed by a device section."""
    existing_args = carried.get("embodiment.args", {})
    return {key: existing_args[key] for key in managed_args if key in existing_args}


def _print_slot_listing(
    kind: str,
    devices: list[str],
    directory: Path,
    out: IO[str],
) -> None:
    """Print the first-use listing for one declared device kind."""
    if kind == "v4l2":
        _print_camera_listing(devices, directory, out)
        return
    noun = "CAN interface" if kind == "can" else "serial device"
    # as_posix: these are canonical Linux locations; Windows str(Path) would
    # render them with backslashes.
    display_dir = (SYSFS_NET if kind == "can" else SERIAL_BY_ID).as_posix()
    print(f"Found {len(devices)} {noun}(s) under {display_dir}:", file=out)
    for number, device in enumerate(devices, start=1):
        name = device if kind == "can" else Path(device).name
        print(f"  {number}. {name}", file=out)


def _device_section(
    embodiment: str,
    slots: tuple[DeviceSlot, ...],
    carried: dict[str, dict[str, str]],
    by_id_dir: Path,
    by_path_dir: Path,
    sysfs_net: Path,
    serial_by_id_dir: Path,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> dict[str, str]:
    """Offer and collect assignments for plugin-declared device slots."""
    kinds = {slot.kind for slot in slots}
    by_id_devices = _scan_cameras(by_id_dir) if "v4l2" in kinds else []
    by_path_devices = _scan_cameras(by_path_dir) if "v4l2" in kinds else []
    can_devices = _scan_can(sysfs_net) if "can" in kinds else []
    serial_devices = _scan_serial(serial_by_id_dir) if "serial" in kinds else []
    active_is_by_id = bool(by_id_devices) or not by_path_devices
    advertise_path_toggle = len(by_path_devices) > len(by_id_devices)
    devices_by_kind = {
        "v4l2": by_id_devices if active_is_by_id else by_path_devices,
        "can": can_devices,
        "serial": serial_devices,
    }
    managed_args = tuple(slot.arg for slot in slots)
    existing_args = carried.get("embodiment.args", {})
    default_enabled = any(devices_by_kind[kind] for kind in kinds) or any(
        arg in existing_args for arg in managed_args
    )
    if not _ask_yes_no("Configure devices?", default_enabled, input_fn=input_fn, out=out):
        return _preserve_managed_args(carried, managed_args)

    listed_kinds: set[str] = set()
    assignments: dict[str, str] = {}
    assigned_devices: dict[str, tuple[str, str, str]] = {}

    def prompt_slot(slot: DeviceSlot) -> None:
        nonlocal active_is_by_id
        if slot.kind == "v4l2":
            primary_devices = by_id_devices
            secondary_devices = by_path_devices
            primary_dir = by_id_dir
            secondary_dir = by_path_dir
            current_devices = primary_devices if active_is_by_id else secondary_devices
            current_dir = primary_dir if active_is_by_id else secondary_dir
            rescan_primary = partial(_scan_cameras, primary_dir)
            rescan_secondary = partial(_scan_cameras, secondary_dir)
        elif slot.kind == "can":
            primary_devices = secondary_devices = can_devices
            primary_dir = secondary_dir = SYSFS_NET
            current_devices = can_devices
            current_dir = SYSFS_NET
            rescan_primary = rescan_secondary = partial(_scan_can, sysfs_net)
        else:
            primary_devices = secondary_devices = serial_devices
            primary_dir = secondary_dir = SERIAL_BY_ID
            current_devices = serial_devices
            current_dir = SERIAL_BY_ID
            rescan_primary = rescan_secondary = partial(_scan_serial, serial_by_id_dir)

        if slot.kind not in listed_kinds:
            _print_slot_listing(slot.kind, current_devices, current_dir, out)
            listed_kinds.add(slot.kind)
            if slot.kind == "v4l2" and advertise_path_toggle:
                print(
                    _paint(
                        f"only {len(by_id_devices)} by-id entries for "
                        f"{len(by_path_devices)} detected cameras — identical cameras without "
                        "serials collide there; by-path names are stable per physical USB port",
                        _YELLOW,
                        out,
                    ),
                    file=out,
                )

        selected, active_is_by_id = _prompt_device_slot(
            slot.label,
            slot.kind,
            primary_devices,
            secondary_devices,
            active_is_by_id,
            primary_dir,
            secondary_dir,
            existing_args.get(slot.arg),
            assigned_devices,
            advertise_path_toggle,
            input_fn=input_fn,
            out=out,
            rescan_by_id=rescan_primary,
            rescan_by_path=rescan_secondary,
        )
        assignments.pop(slot.arg, None)
        assigned_devices.pop(slot.arg, None)
        if selected is not None:
            assignments[slot.arg] = selected
            assigned_devices[slot.arg] = (slot.kind, slot.label, selected)

    for slot in slots:
        prompt_slot(slot)

    groups = tuple(dict.fromkeys(slot.group for slot in slots if slot.group is not None))
    for group in groups:
        group_slots = tuple(slot for slot in slots if slot.group == group)
        while 0 < sum(slot.arg in assignments for slot in group_slots) < len(group_slots):
            print(
                _paint(
                    f"{embodiment} needs all {group} slots or none; "
                    "writing none unless you go back",
                    _YELLOW,
                    out,
                ),
                file=out,
            )
            if not _ask_yes_no(
                "Go back and choose devices again?", True, input_fn=input_fn, out=out
            ):
                for slot in group_slots:
                    assignments.pop(slot.arg, None)
                    assigned_devices.pop(slot.arg, None)
                break
            for slot in group_slots:
                assignments.pop(slot.arg, None)
                assigned_devices.pop(slot.arg, None)
                prompt_slot(slot)
    return assignments


def _scan_cameras(v4l_dir: Path) -> list[str]:
    """Return sorted device paths, preferring V4L2 color-stream entries."""
    try:
        entries = sorted(v4l_dir.iterdir())
    except FileNotFoundError:
        return []
    color_entries = [entry for entry in entries if entry.name.endswith("-video-index0")]
    return [str(entry) for entry in color_entries or entries]


def _scan_can(sysfs_net: Path) -> list[str]:
    """Return sorted SocketCAN interface names from a sysfs net directory."""
    try:
        entries = sorted(sysfs_net.iterdir())
    except OSError:
        return []
    interfaces: list[str] = []
    for entry in entries:
        try:
            interface_type = (entry / "type").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if interface_type == "280":
            interfaces.append(entry.name)
    return interfaces


def _scan_serial(serial_by_id_dir: Path) -> list[str]:
    """Return sorted absolute paths from a serial by-id directory."""
    try:
        entries = sorted(serial_by_id_dir.iterdir())
    except OSError:
        return []
    return [str(entry.absolute()) for entry in entries]


def _can_serial(sysfs_net: Path, ifname: str) -> str | None:
    """Read a CAN adapter serial through its sysfs device link, if available."""
    try:
        serial_path = (sysfs_net / ifname / "device").resolve().parent / "serial"
        return serial_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _suggest_can_pinning(
    sysfs_net: Path,
    slots: tuple[DeviceSlot, ...],
    assignments: Mapping[str, str],
    *,
    out: IO[str],
) -> None:
    """Suggest serial-pinned udev names for assigned order-dependent CAN devices."""
    scanned = _scan_can(sysfs_net)
    order_dependent = [ifname for ifname in scanned if re.fullmatch(r"can\d+", ifname)]
    if len(order_dependent) < 2 or not any(
        assignments.get(slot.arg) in order_dependent for slot in slots if slot.kind == "can"
    ):
        return

    def _is_usb(ifname: str) -> bool:
        # Real sysfs buses are numbered (usb1, usb2, ...); a bare "usb"
        # segment never occurs on hardware.
        parts = (sysfs_net / ifname / "device").resolve().parts
        return any(re.fullmatch(r"usb\d*", part) for part in parts)

    if not any(_is_usb(ifname) for ifname in order_dependent):
        return

    warning = "these CAN interfaces have order-dependent names; a replug can swap them."
    serials = [_can_serial(sysfs_net, ifname) for ifname in order_dependent]
    if not all(serials) or len(set(serials)) != len(serials):
        print(_paint(warning, _YELLOW, out), file=out)
        return

    derived_names: list[str] = []
    for ifname in order_dependent:
        assigned_arg = next(
            (
                slot.arg
                for slot in slots
                if slot.kind == "can" and assignments.get(slot.arg) == ifname
            ),
            None,
        )
        if assigned_arg is None:
            derived_names.append(f"can_{ifname}")
            continue
        stem = re.sub(r"_(?:channel|bus)$", "", assigned_arg)
        derived_names.append(stem if stem.startswith("can") else f"can_{stem}")

    collision = len(set(derived_names)) != len(derived_names) or any(
        name in scanned for name in derived_names
    )
    if collision or any(len(name) > 15 for name in derived_names):
        derived_names = [f"can_{chr(ord('a') + index)}" for index in range(len(derived_names))]

    serial_values = [serial for serial in serials if serial is not None]
    rules = [
        f'  SUBSYSTEM=="net", ACTION=="add", ATTRS{{serial}}=="{serial}", NAME="{name}"'
        for serial, name in zip(serial_values, derived_names, strict=True)
    ]
    block = "\n".join(
        [
            warning,
            "pin them by adapter serial (paste into /etc/udev/rules.d/70-can-names.rules,",
            "then replug or reboot), and re-run setup to record the pinned names:",
            *rules,
        ]
    )
    print(_paint(block, _YELLOW, out), file=out)


def _read_raw_config(path: Path) -> dict[str, dict[str, str]] | str:
    """Read raw section values, returning parse-error text instead of raising."""
    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(";", "#"),
    )
    try:
        with path.open(encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except (configparser.Error, UnicodeDecodeError) as exc:
        return str(exc)
    return {section: dict(parser.items(section, raw=True)) for section in parser.sections()}


def _render_config(
    defaults: dict[str, str],
    embodiment_args: dict[str, str],
    carried: dict[str, dict[str, str]],
    managed_args: tuple[str, ...] = CAMERA_KEYS,
) -> str:
    """Render a full commented config while carrying unmanaged raw values."""
    sections: list[str] = []

    default_lines: list[str] = []
    for key in SUGGESTED:
        if key not in defaults:
            continue
        # Prompted values can be multiline too: a continuation-line value in
        # an existing config passes free-text validation and survives Enter.
        value = defaults[key].replace("\n", "\n\t")
        line = f"{key} = {value}"
        if comment := _DEFAULT_COMMENTS.get(key):
            line = f"{line:<26}# {comment}" if len(line) < 26 else f"{line}  # {comment}"
        default_lines.append(line)
    for key, value in carried.get("defaults", {}).items():
        if key not in SUGGESTED:
            value = value.replace("\n", "\n\t")
            default_lines.append(f"{key} = {value}")
    if default_lines:
        sections.append("[defaults]\n" + "\n".join(default_lines))

    embodiment_lines: list[str] = []
    for key in managed_args:
        if key in embodiment_args:
            # Enter-accepted "(current)" values come from the raw read and
            # can be multiline like any carried value.
            value = embodiment_args[key].replace("\n", "\n\t")
            embodiment_lines.append(f"{key} = {value}")
    for key, value in carried.get("embodiment.args", {}).items():
        if key not in managed_args:
            value = value.replace("\n", "\n\t")
            embodiment_lines.append(f"{key} = {value}")
    if embodiment_lines:
        sections.append("[embodiment.args]\n" + "\n".join(embodiment_lines))

    for section, values in carried.items():
        if section in ("defaults", "embodiment.args"):
            continue
        if values:
            lines = []
            for key, value in values.items():
                value = value.replace("\n", "\n\t")
                lines.append(f"{key} = {value}")
            sections.append(f"[{section}]\n" + "\n".join(lines))

    return "\n\n".join(sections) + "\n"


def run_setup(
    env: Mapping[str, str],
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
    interactive: bool,
    by_id_dir: Path = V4L_BY_ID,
    by_path_dir: Path = V4L_BY_PATH,
    sysfs_net: Path = SYSFS_NET,
    serial_by_id_dir: Path = SERIAL_BY_ID,
) -> int:
    """Drive the interactive setup wizard and return its process exit code."""
    if not interactive:
        raise SystemExit("setup is interactive; see the README for manual config")

    path = _config_path(env)
    if path is None:
        raise SystemExit("cannot locate a config home: set $XDG_CONFIG_HOME or $HOME")

    print(f"{_paint('inspect-robots setup', _BOLD, out)} — writes {path}", file=out)
    print(
        _paint(
            "Each prompt shows a [suggested] value: press Enter to accept it, "
            "or type a replacement.",
            _DIM,
            out,
        ),
        file=out,
    )

    try:
        carried: dict[str, dict[str, str]] = {}
        if path.is_file():
            raw_config = _read_raw_config(path)
            if isinstance(raw_config, str):
                print(_paint(raw_config, _YELLOW, out), file=out)
                repair = _ask_yes_no(
                    "Back up the broken file and start fresh?",
                    True,
                    input_fn=input_fn,
                    out=out,
                )
                if not repair:
                    print(_paint("setup aborted; nothing written", _YELLOW, out), file=out)
                    return 1
            else:
                carried = raw_config
                print(
                    _paint(
                        "Found an existing config; its values are the suggestions "
                        f"below (the old file will be saved as {path.name}.bak).",
                        _DIM,
                        out,
                    ),
                    file=out,
                )

        headless = "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env
        defaults = _prompt_defaults(
            carried,
            headless=headless,
            input_fn=input_fn,
            out=out,
        )
        from inspect_robots.registry import registered

        embodiment_factories = registered("embodiment")
        configured_embodiment = defaults["embodiment"]
        slots = (
            device_slots(embodiment_factories[configured_embodiment])
            if configured_embodiment in embodiment_factories
            else ()
        )
        if slots:
            managed_args = tuple(slot.arg for slot in slots)
            embodiment_args = _device_section(
                configured_embodiment,
                slots,
                carried,
                by_id_dir,
                by_path_dir,
                sysfs_net,
                serial_by_id_dir,
                input_fn=input_fn,
                out=out,
            )
            _suggest_can_pinning(sysfs_net, slots, embodiment_args, out=out)
        else:
            managed_args = CAMERA_KEYS
            embodiment_args = _camera_section(
                carried,
                by_id_dir,
                by_path_dir,
                input_fn=input_fn,
                out=out,
            )
    except (EOFError, KeyboardInterrupt):
        print(_paint("setup aborted; nothing written", _YELLOW, out), file=out)
        return 1

    text = _render_config(defaults, embodiment_args, carried, managed_args)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.is_file():
        path.replace(path.with_name(path.name + ".bak"))
    tmp.replace(path)
    print(_paint(f"Wrote {path}", _GREEN, out), file=out)
    # Repeat the plugin reminder where it cannot scroll away: the per-prompt
    # warning is easy to miss while Enter-accepting the suggestions.
    missing = [
        f"{kind} '{defaults[kind]}'"
        for kind in ("policy", "embodiment")
        if not _is_registered(kind, defaults[kind])
    ]
    if missing:
        print(
            _paint(
                f"reminder: {' and '.join(missing)} not registered here; install "
                "the plugin (e.g. `uv pip install inspect-robots-yam`) before "
                "your first run",
                _YELLOW,
                out,
            ),
            file=out,
        )
    runtime_lines = _runtime_requirement_lines(defaults)
    if runtime_lines:
        count = len(runtime_lines)
        dependency = "dependency is" if count == 1 else "dependencies are"
        block = "\n".join(
            [f"setup complete, but {count} runtime {dependency} missing:", *runtime_lines]
        )
        print(_paint(block, _YELLOW, out), file=out)
    next_cmd = 'uv run inspect-robots "place the fork on the plate"'
    print(f"Next: {_paint(next_cmd, _CYAN, out)}", file=out)
    return 0
