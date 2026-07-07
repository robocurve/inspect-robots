"""User defaults for the zero-config CLI: config-file parsing and precedence."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from inspect_robots._defaults import (
    ENV_EMBODIMENT,
    ENV_POLICY,
    Defaults,
    load_defaults,
    parse_value,
)


def _write_config(config_home: Path, body: str) -> Path:
    path = config_home / "inspect-robots" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


_FULL_CONFIG = """
[defaults]
policy = molmoact2-yam
embodiment = yam-bimanual
scorer = operator      ; ad-hoc runs only
max_steps = 450

[policy.args]
checkpoint = ~/ckpts/molmoact2-yam.pt
temperature = 0.5
verbose = true

[embodiment.args]
cameras = wrist,front
port = none
"""


def test_full_config_parses_with_inline_comments_and_expansion(tmp_path: Path) -> None:
    _write_config(tmp_path, _FULL_CONFIG)
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    assert d.policy == "molmoact2-yam"
    assert d.embodiment == "yam-bimanual"
    assert d.scorer == "operator"  # inline comment stripped
    assert d.max_steps == 450
    assert d.policy_source == str(tmp_path / "inspect-robots" / "config.ini")
    assert d.embodiment_source == d.policy_source
    # ~ expanded, and value parsing matches the CLI's -P/-E parsing.
    checkpoint = d.policy_args["checkpoint"]
    assert isinstance(checkpoint, str) and not checkpoint.startswith("~")
    assert checkpoint.endswith("ckpts/molmoact2-yam.pt")
    assert d.policy_args["temperature"] == 0.5
    assert d.policy_args["verbose"] is True
    assert d.embodiment_args == {"cameras": "wrist,front", "port": None}


def test_env_vars_override_config_names_but_not_args(tmp_path: Path) -> None:
    _write_config(tmp_path, _FULL_CONFIG)
    d = load_defaults(
        {
            "XDG_CONFIG_HOME": str(tmp_path),
            ENV_POLICY: "other-policy",
            ENV_EMBODIMENT: "other-arm",
        }
    )
    assert d.policy == "other-policy"
    assert d.policy_source == f"${ENV_POLICY}"
    assert d.embodiment == "other-arm"
    assert d.embodiment_source == f"${ENV_EMBODIMENT}"
    # Config-file args still apply to whatever component ends up selected.
    assert d.policy_args["temperature"] == 0.5


def test_env_vars_work_without_any_config_file(tmp_path: Path) -> None:
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path), ENV_POLICY: "p1"})
    assert d.policy == "p1"
    assert d.embodiment is None
    assert d.policy_args == {}


def test_home_fallback_when_xdg_unset(tmp_path: Path) -> None:
    _write_config(tmp_path / ".config", "[defaults]\npolicy = from-home\n")
    d = load_defaults({"HOME": str(tmp_path)})
    assert d.policy == "from-home"
    assert d.embodiment is None  # unset keys stay None


def test_no_home_and_no_xdg_means_no_config(tmp_path: Path) -> None:
    assert load_defaults({}) == Defaults()


def test_missing_config_file_means_empty_defaults(tmp_path: Path) -> None:
    assert load_defaults({"XDG_CONFIG_HOME": str(tmp_path)}) == Defaults()


def test_unknown_sections_and_keys_are_ignored(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "[defaults]\npolicy = p\nfuture_knob = 7\n\n[future.section]\nx = 1\n",
    )
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    # Full equality: the unknown key and section contributed nothing at all.
    assert d == Defaults(policy="p", policy_source=str(path))


def test_malformed_ini_raises_system_exit_naming_file(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "not an ini file [\n===\n")
    # re.escape: a Windows path's backslashes are not a regex.
    with pytest.raises(SystemExit, match=re.escape(str(path))):
        load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})


@pytest.mark.parametrize("bad", ["lots", "0", "-3", "true", "2.5"])
def test_invalid_max_steps_raises_system_exit(tmp_path: Path, bad: str) -> None:
    _write_config(tmp_path, f"[defaults]\nmax_steps = {bad}\n")
    with pytest.raises(SystemExit, match="max_steps"):
        load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})


def test_parse_value_scalars() -> None:
    assert parse_value("true") is True
    assert parse_value("False") is False
    assert parse_value("none") is None
    assert parse_value("42") == 42
    assert parse_value("2.5") == 2.5
    assert parse_value("hello") == "hello"
