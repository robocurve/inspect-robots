"""User defaults for the zero-config CLI: config-file parsing and precedence."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from inspect_robots.defaults import (
    _ENV_EMBODIMENT as ENV_EMBODIMENT,
)
from inspect_robots.defaults import (
    _ENV_POLICY as ENV_POLICY,
)
from inspect_robots.defaults import (
    _ENV_SIM_EMBODIMENT as ENV_SIM_EMBODIMENT,
)
from inspect_robots.defaults import (
    Defaults,
    load_defaults,
)
from inspect_robots.defaults import (
    _parse_value as parse_value,
)
from inspect_robots.defaults import (
    _set_default as set_default,
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


def test_grader_args_section_parses_with_the_grader_as_owner(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "[defaults]\ngrader = vlm\n[grader.args]\nmodel = judge\nmax_cameras = 2\n",
    )
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    assert d.grader == "vlm"
    assert d.grader_args == {"model": "judge", "max_cameras": 2}
    assert d.grader_args_owner == "vlm"


def test_taskgen_args_section_parses_without_an_owner(tmp_path: Path) -> None:
    """Parse [taskgen.args] like the other args sections, with no owner field."""
    _write_config(
        tmp_path,
        "[taskgen.args]\nmodel = gpt-5.2\nmax_cameras = 2\n"
        "instructions_file = ~/prompts/jungle.txt\n",
    )
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    assert d.taskgen_args["model"] == "gpt-5.2"
    assert d.taskgen_args["max_cameras"] == 2
    instructions_file = d.taskgen_args["instructions_file"]
    assert isinstance(instructions_file, str) and not instructions_file.startswith("~")
    assert instructions_file.endswith("prompts/jungle.txt")
    # Deliberately ownerless (plan 0071): taskgen is never registry-selected.
    assert not hasattr(d, "taskgen_args_owner")


def test_config_value_with_literal_percent_loads_unchanged(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "[defaults]\npolicy = 50%off\n")
    defaults = load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    assert defaults.policy == "50%off"
    assert defaults.policy_source == str(path)


def test_set_default_round_trips_literal_percent(tmp_path: Path) -> None:
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    path = set_default(env, "policy", "50%off")
    assert set_default(env, "policy", "50%off") == path
    assert load_defaults(env).policy == "50%off"
    assert "policy = 50%off" in path.read_text(encoding="utf-8")


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
    # Config-file args stay loaded, but their owner stays the *file's* name:
    # the CLI applies them only when the selected component matches it, so an
    # env-selected "other-policy" never inherits molmoact2-yam's args (#44).
    assert d.policy_args["temperature"] == 0.5
    assert d.policy_args_owner == "molmoact2-yam"
    assert d.embodiment_args_owner == "yam-bimanual"


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
    assert d == Defaults(policy="p", policy_source=str(path), policy_args_owner="p")


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


_SIM_CONFIG = """
[defaults]
embodiment = yam-bimanual
sim_embodiment = yam-bimanual-isaac

[embodiment.args]
port = /dev/ttyUSB0

[sim_embodiment.args]
headless = true
scene_file = ~/scenes/kitchen.usd
"""


def test_sim_embodiment_config_is_independent_of_real(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _SIM_CONFIG)
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path)})
    scene_file = d.sim_embodiment_args["scene_file"]
    assert isinstance(scene_file, str) and not scene_file.startswith("~")  # ~ expanded
    # Full equality: sim and real defaults live side by side without bleed.
    assert d == Defaults(
        embodiment="yam-bimanual",
        embodiment_source=str(path),
        sim_embodiment="yam-bimanual-isaac",
        sim_embodiment_source=str(path),
        embodiment_args={"port": "/dev/ttyUSB0"},
        sim_embodiment_args={"headless": True, "scene_file": scene_file},
        embodiment_args_owner="yam-bimanual",
        sim_embodiment_args_owner="yam-bimanual-isaac",
    )


def test_env_sim_embodiment_overrides_config_but_not_real(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _SIM_CONFIG)
    d = load_defaults({"XDG_CONFIG_HOME": str(tmp_path), ENV_SIM_EMBODIMENT: "other-sim"})
    assert d.sim_embodiment == "other-sim"
    assert d.sim_embodiment_source == f"${ENV_SIM_EMBODIMENT}"
    assert d.embodiment == "yam-bimanual"  # the real default is untouched
    assert d.embodiment_source == str(path)
    # The env var overrides only the *name*; config-file sim args still apply.
    assert d.sim_embodiment_args["headless"] is True
    scene_file = d.sim_embodiment_args["scene_file"]
    assert isinstance(scene_file, str) and scene_file.endswith("scenes/kitchen.usd")


def test_config_store_frames_parses_bool(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\nstore_frames = true\n")
    defaults = load_defaults({"XDG_CONFIG_HOME": str(config_home)})
    assert defaults.store_frames is True


def test_config_store_frames_defaults_false(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\npolicy = x\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).store_frames is False


def test_config_store_frames_rejects_non_bool(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\nstore_frames = 12\n")
    with pytest.raises(SystemExit, match="store_frames"):
        load_defaults({"XDG_CONFIG_HOME": str(config_home)})


def test_config_rerun_parses_bool(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\nrerun = true\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).rerun is True


def test_config_rerun_defaults_false(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\npolicy = x\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).rerun is False


def test_config_rerun_rejects_non_bool(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\nrerun = viewer\n")
    with pytest.raises(SystemExit, match="rerun"):
        load_defaults({"XDG_CONFIG_HOME": str(config_home)})


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
def test_config_rerun_save_parses_bool(tmp_path: Path, raw: str, expected: bool) -> None:
    """The per-rig recording default accepts only explicit booleans."""
    config_home = tmp_path / "cfg"
    _write_config(config_home, f"[defaults]\nrerun_save = {raw}\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).rerun_save is expected


def test_config_rerun_save_defaults_true(tmp_path: Path) -> None:
    """Live runs save by default when the key is absent."""
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\npolicy = x\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).rerun_save is True


def test_config_rerun_save_rejects_non_bool(tmp_path: Path) -> None:
    """A non-boolean recording default is rejected with the documented message."""
    config_home = tmp_path / "cfg"
    path = _write_config(config_home, "[defaults]\nrerun_save = sometimes\n")
    with pytest.raises(
        SystemExit,
        match=(
            rf"{re.escape(str(path))}.*\[defaults\] rerun_save must be true or false, "
            "got 'sometimes'"
        ),
    ):
        load_defaults({"XDG_CONFIG_HOME": str(config_home)})


def test_set_default_validates_and_round_trips_rerun_save(tmp_path: Path) -> None:
    """Config editing applies the same boolean contract as config loading."""
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    with pytest.raises(SystemExit, match="rerun_save must be true or false"):
        set_default(env, "rerun_save", "sometimes")
    set_default(env, "rerun_save", "false")
    assert load_defaults(env).rerun_save is False


def test_config_rerun_port_parses_int(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\nrerun_port = 9877\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).rerun_port == 9877


def test_config_rerun_port_defaults_none(tmp_path: Path) -> None:
    config_home = tmp_path / "cfg"
    _write_config(config_home, "[defaults]\npolicy = x\n")
    assert load_defaults({"XDG_CONFIG_HOME": str(config_home)}).rerun_port is None


@pytest.mark.parametrize("bad", ["true", "0", "65536", "9876.5"])
def test_config_rerun_port_rejects_invalid(tmp_path: Path, bad: str) -> None:
    config_home = tmp_path / "cfg"
    path = _write_config(config_home, f"[defaults]\nrerun_port = {bad}\n")
    with pytest.raises(SystemExit, match=rf"{re.escape(str(path))}.*1-65535"):
        load_defaults({"XDG_CONFIG_HOME": str(config_home)})


def test_set_default_requires_config_home() -> None:
    with pytest.raises(SystemExit, match="config home"):
        set_default({}, "policy", "scripted")


def test_set_default_rejects_malformed_existing_config(tmp_path: Path) -> None:
    path = tmp_path / "inspect-robots" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_text("policy = dangling, no section header\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="malformed config"):
        set_default({"XDG_CONFIG_HOME": str(tmp_path)}, "policy", "scripted")


@pytest.mark.parametrize("key", ["policy", "embodiment", "sim_embodiment", "grader"])
def test_set_default_warns_when_component_change_leaves_owned_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    path = tmp_path / "inspect-robots" / "config.ini"
    _write_config(
        tmp_path,
        f"[defaults]\n{key} = old-component\n[{key}.args]\nport = can0\n",
    )
    set_default({"XDG_CONFIG_HOME": str(tmp_path)}, key, "new-component")
    assert capsys.readouterr().err == (
        f"warning: [{key}.args] was configured for 'old-component'; "
        f"it will be ignored for 'new-component': update or remove it in {path}\n"
    )

    set_default({"XDG_CONFIG_HOME": str(tmp_path)}, key, "new-component")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("args_section", ["", "[embodiment.args]\n"])
def test_set_default_does_not_warn_without_nonempty_args_section(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    args_section: str,
) -> None:
    _write_config(tmp_path, f"[defaults]\nembodiment = old-arm\n{args_section}")
    set_default({"XDG_CONFIG_HOME": str(tmp_path)}, "embodiment", "new-arm")
    assert capsys.readouterr().err == ""


def test_set_default_does_not_warn_when_args_have_no_prior_owner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(tmp_path, "[embodiment.args]\nport = can0\n")
    set_default({"XDG_CONFIG_HOME": str(tmp_path)}, "embodiment", "new-arm")
    assert capsys.readouterr().err == ""


def test_public_re_export_init_dotenv() -> None:
    from inspect_robots.defaults import init_dotenv

    assert init_dotenv is not None
