"""Pure, IO-injected support for the ``inspect-robots setup`` wizard."""

from __future__ import annotations

import configparser
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO

from inspect_robots._defaults import _config_path, parse_value

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
        entered = input_fn(f"{prompt} [{default}]: ")
        value = entered if entered else default
        if validate(value):
            return value
        print(constraint, file=out)


def _ask_yes_no(
    prompt: str,
    default: bool,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> bool:
    """Ask a conventional yes/no question, re-prompting unclear answers."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        entered = input_fn(f"{prompt} {suffix} ").strip().lower()
        if not entered:
            return default
        if entered in ("y", "yes"):
            return True
        if entered in ("n", "no"):
            return False
        print("please answer yes or no", file=out)


def _warn_unregistered(kind: str, name: str, out: IO[str]) -> None:
    """Warn about unavailable plugins without importing the registry eagerly."""
    from inspect_robots.registry import registered

    if name not in registered(kind):
        print(
            f"'{name}' is not registered here — install its plugin, e.g. "
            "`uv pip install inspect-robots-yam`",
            file=out,
        )


def _prompt_defaults(
    carried: dict[str, dict[str, str]],
    *,
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
        default = suggestion
        if key in raw_defaults:
            configured = raw_defaults[key]
            if validate(configured):
                default = configured
            else:
                print(f"ignoring invalid {key} {configured!r} from config.ini", file=out)
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


def _prompt_camera_role(
    role: str,
    devices: list[str],
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> str | None:
    """Prompt for a numbered device, an absolute path, or a skipped role."""
    prompt = f"{role} camera — number, absolute path, or 's' to skip: "
    while True:
        entered = input_fn(prompt).strip()
        if entered.lower() == "s":
            return None
        if entered.startswith("/"):
            if not Path(entered).exists():
                print(
                    f"warning: {entered} does not exist here "
                    "(ok if this config is for another machine)",
                    file=out,
                )
            return entered
        if entered.isdigit():
            number = int(entered)
            if 1 <= number <= len(devices):
                return devices[number - 1]
        print("enter a device number, absolute path, or 's' to skip", file=out)


def _camera_section(
    carried: dict[str, dict[str, str]],
    by_id_dir: Path,
    by_path_dir: Path,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
) -> dict[str, str]:
    """Offer and collect the Task 2 camera assignments."""
    devices = _scan_cameras(by_id_dir)
    device_dir = by_id_dir
    if not devices:
        devices = _scan_cameras(by_path_dir)
        device_dir = by_path_dir

    camera_keys = tuple(f"{role}_cam_device" for role in CAM_ROLES)
    existing_args = carried.get("embodiment.args", {})
    default_enabled = bool(devices) or any(key in existing_args for key in camera_keys)
    if not _ask_yes_no("Configure cameras?", default_enabled, input_fn=input_fn, out=out):
        return {}

    if devices:
        _print_camera_listing(devices, device_dir, out)
    else:
        print(
            "no /dev/v4l devices found (not Linux, or no cameras attached)",
            file=out,
        )

    while True:
        assignments: dict[str, str] = {}
        for role in CAM_ROLES:
            selected = _prompt_camera_role(role, devices, input_fn=input_fn, out=out)
            if selected is not None:
                assignments[f"{role}_cam_device"] = selected
        if len(assignments) in (0, len(CAM_ROLES)):
            return assignments
        print(
            "yam_arms needs all three cameras or none; writing none unless you go back",
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
        line = f"{key} = {defaults[key]}"
        if comment := _DEFAULT_COMMENTS.get(key):
            line = f"{line:<26}# {comment}"
        default_lines.append(line)
    for key, value in carried.get("defaults", {}).items():
        if key not in SUGGESTED:
            default_lines.append(f"{key} = {value}")
    if default_lines:
        sections.append("[defaults]\n" + "\n".join(default_lines))

    camera_keys = tuple(f"{role}_cam_device" for role in CAM_ROLES)
    embodiment_lines: list[str] = []
    for key in camera_keys:
        if key in embodiment_args:
            embodiment_lines.append(f"{key} = {embodiment_args[key]}")
    for key, value in carried.get("embodiment.args", {}).items():
        if key not in camera_keys:
            embodiment_lines.append(f"{key} = {value}")
    if embodiment_lines:
        sections.append("[embodiment.args]\n" + "\n".join(embodiment_lines))

    for section, values in carried.items():
        if section in ("defaults", "embodiment.args"):
            continue
        if values:
            lines = [f"{key} = {value}" for key, value in values.items()]
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

    try:
        carried: dict[str, dict[str, str]] = {}
        if path.is_file():
            raw_config = _read_raw_config(path)
            if isinstance(raw_config, str):
                print(raw_config, file=out)
                repair = _ask_yes_no(
                    "Back up the broken file and start fresh?",
                    True,
                    input_fn=input_fn,
                    out=out,
                )
                if not repair:
                    return 1
            else:
                carried = raw_config

        defaults = _prompt_defaults(carried, input_fn=input_fn, out=out)
        embodiment_args = _camera_section(
            carried,
            by_id_dir,
            by_path_dir,
            input_fn=input_fn,
            out=out,
        )
    except (EOFError, KeyboardInterrupt):
        print("setup aborted; nothing written", file=out)
        return 1

    text = _render_config(defaults, embodiment_args, carried)
    if path.is_file():
        path.replace(path.with_name(path.name + ".bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    print(f"Wrote {path}", file=out)
    print('Next: uv run inspect-robots "place the fork on the plate"', file=out)
    return 0
