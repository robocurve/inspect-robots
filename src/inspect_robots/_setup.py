"""Pure support for the ``inspect-robots setup`` wizard from plan 0009.

Task 1 ships only camera scanning, raw config reading, and commented config
rendering. The IO-driven wizard that composes these helpers comes later.
"""

from __future__ import annotations

import configparser
from pathlib import Path

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
