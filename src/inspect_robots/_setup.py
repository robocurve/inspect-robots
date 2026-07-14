"""Pure, IO-injected support for the ``inspect-robots setup`` wizard."""

from __future__ import annotations

import configparser
import os
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import IO

from inspect_robots._defaults import _config_path, parse_value

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
V4L_BY_ID: Path = Path("/dev/v4l/by-id")
V4L_BY_PATH: Path = Path("/dev/v4l/by-path")

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
) -> str | None:
    """Identify one device by diffing the listing while it is unplugged."""
    input_fn(f"Unplug the {role} camera now, then press Enter...")
    unplugged_devices = rescan()
    disappeared = [device for device in devices if device not in unplugged_devices]
    if not disappeared:
        print(
            _paint("no camera device disappeared; unplug one camera and try again", _YELLOW, out),
            file=out,
        )
        return None
    if len(disappeared) > 1:
        print(
            _paint(
                f"{len(disappeared)} camera devices disappeared; unplug only one and try again",
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


def _camera_role_prompt(
    role: str,
    devices: list[str],
    current: str | None,
    advertise_path_toggle: bool,
    *,
    out: IO[str],
) -> str:
    """Prompt text for one camera role, with the Enter-accept current value."""
    choices = f"{role} camera — number, 'u' to identify by unplugging"
    if advertise_path_toggle:
        choices += ", 'p' to switch listing"
    choices += ", 's' to skip"
    if current is None:
        return choices + ": "
    status = "current" if current in devices else "current, not detected"
    return f"{choices} [{_paint(f'{current} ({status})', _CYAN, out)}]: "


def _prompt_camera_role(
    role: str,
    by_id_devices: list[str],
    by_path_devices: list[str],
    active_is_by_id: bool,
    by_id_dir: Path,
    by_path_dir: Path,
    current: str | None,
    assigned: dict[str, str],
    advertise_path_toggle: bool,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> tuple[str | None, bool]:
    """Prompt for one role and return its device plus active listing state."""
    while True:
        devices = by_id_devices if active_is_by_id else by_path_devices
        device_dir = by_id_dir if active_is_by_id else by_path_dir
        prompt = _camera_role_prompt(role, devices, current, advertise_path_toggle, out=out)
        entered = input_fn(prompt).strip()
        selected: str | None = None
        if entered.lower() == "s":
            return None, active_is_by_id
        if entered.lower() == "p":
            active_is_by_id = not active_is_by_id
            devices = by_id_devices if active_is_by_id else by_path_devices
            device_dir = by_id_dir if active_is_by_id else by_path_dir
            _print_camera_listing(devices, device_dir, out)
            continue
        if entered.lower() == "u":
            selected = _identify_by_replug(
                role,
                devices,
                input_fn=input_fn,
                out=out,
                rescan=partial(_scan_cameras, device_dir),
            )
            if selected is None:
                continue
        elif not entered and current is not None:
            selected = current
        elif entered.startswith("/") or Path(entered).is_absolute():
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
        if selected is None:
            print(
                _paint(
                    "enter a device number, absolute path, 'u' to identify, or 's' to skip",
                    _YELLOW,
                    out,
                ),
                file=out,
            )
            continue

        other_role = next(
            (
                assigned_key.removesuffix("_cam_device")
                for assigned_key, device in assigned.items()
                if device == selected
            ),
            None,
        )
        if other_role is not None:
            print(
                _paint(
                    f"warning: {selected} is already assigned to the {other_role} camera",
                    _YELLOW,
                    out,
                ),
                file=out,
            )
            if not _ask_yes_no(
                f"Use {selected} for both {other_role} and {role} cameras?",
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

    camera_keys = tuple(f"{role}_cam_device" for role in CAM_ROLES)
    existing_args = carried.get("embodiment.args", {})
    default_enabled = bool(devices) or any(key in existing_args for key in camera_keys)
    if not _ask_yes_no("Configure cameras?", default_enabled, input_fn=input_fn, out=out):
        return {key: existing_args[key] for key in camera_keys if key in existing_args}

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
        for role in CAM_ROLES:
            key = f"{role}_cam_device"
            selected, active_is_by_id = _prompt_camera_role(
                role,
                by_id_devices,
                by_path_devices,
                active_is_by_id,
                by_id_dir,
                by_path_dir,
                existing_args.get(key),
                assignments,
                advertise_path_toggle,
                input_fn=input_fn,
                out=out,
            )
            if selected is not None:
                assignments[key] = selected
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


def _scan_cameras(v4l_dir: Path) -> list[str]:
    """Return sorted device paths, preferring V4L2 color-stream entries."""
    try:
        entries = sorted(v4l_dir.iterdir())
    except FileNotFoundError:
        return []
    color_entries = [entry for entry in entries if entry.name.endswith("-video-index0")]
    return [str(entry) for entry in color_entries or entries]


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

    camera_keys = tuple(f"{role}_cam_device" for role in CAM_ROLES)
    embodiment_lines: list[str] = []
    for key in camera_keys:
        if key in embodiment_args:
            # Enter-accepted "(current)" values come from the raw read and
            # can be multiline like any carried value.
            value = embodiment_args[key].replace("\n", "\n\t")
            embodiment_lines.append(f"{key} = {value}")
    for key, value in carried.get("embodiment.args", {}).items():
        if key not in camera_keys:
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

    text = _render_config(defaults, embodiment_args, carried)
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
    next_cmd = 'uv run inspect-robots "place the fork on the plate"'
    print(f"Next: {_paint(next_cmd, _CYAN, out)}", file=out)
    return 0
