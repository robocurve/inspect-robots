"""Public user-defaults API and config-path behavior."""

from __future__ import annotations

from pathlib import Path

import inspect_robots
from inspect_robots import defaults


def _write_config(config_home: Path, body: str) -> Path:
    path = config_home / "inspect-robots" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_public_surface_is_exact_and_resolves() -> None:
    assert defaults.__all__ == ["Defaults", "config_path", "load_defaults"]
    for name in defaults.__all__:
        assert hasattr(defaults, name)


def test_package_exports_defaults_module() -> None:
    assert "defaults" in inspect_robots.__all__
    assert inspect_robots.defaults is defaults


def test_config_path_xdg_wins_over_home(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    home = tmp_path / "home"
    assert (
        defaults.config_path({"XDG_CONFIG_HOME": str(xdg), "HOME": str(home)})
        == xdg / "inspect-robots" / "config.ini"
    )


def test_config_path_falls_back_to_home_dot_config(tmp_path: Path) -> None:
    assert defaults.config_path({"HOME": str(tmp_path)}) == (
        tmp_path / ".config" / "inspect-robots" / "config.ini"
    )


def test_config_path_without_config_home_is_none() -> None:
    assert defaults.config_path({}) is None


def test_config_path_returns_file_path_whether_or_not_it_exists(tmp_path: Path) -> None:
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    path = defaults.config_path(env)
    assert path is not None
    assert path.parts[-2:] == ("inspect-robots", "config.ini")
    assert not path.exists()
    _write_config(tmp_path, "[defaults]\nembodiment = arm\n")
    assert defaults.config_path(env) == path
    assert path.is_file()


def test_public_load_defaults_round_trips_embodiment_args(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "[defaults]\n"
        "embodiment = yam-bimanual\n\n"
        "[embodiment.args]\n"
        "top_cam_device = /dev/video-top\n"
        "can_channel = can0\n",
    )
    loaded = defaults.load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    assert loaded.embodiment_args == {
        "top_cam_device": "/dev/video-top",
        "can_channel": "can0",
    }
    assert loaded.embodiment_args_owner == "yam-bimanual"
