"""User-level default components for the zero-config CLI (plan 0005).

``inspect-robots "place the spoon on the plate"`` needs a policy and an
embodiment without flags. They come from, in order: explicit CLI flags
(handled in ``cli.py``), the ``INSPECT_ROBOTS_POLICY`` /
``INSPECT_ROBOTS_EMBODIMENT`` environment variables, then the user config
file ``<config-home>/inspect-robots/config.ini`` (INI via stdlib
``configparser`` — the core supports py3.10, which has no ``tomllib``).

There is deliberately **no project-local config file**: a checked-in
``./inspect-robots.ini`` choosing which policy runs on the user's hardware
would be a trust footgun for a tool that moves physical robots.
"""

from __future__ import annotations

import configparser
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_POLICY = "INSPECT_ROBOTS_POLICY"
ENV_EMBODIMENT = "INSPECT_ROBOTS_EMBODIMENT"
ENV_SIM_EMBODIMENT = "INSPECT_ROBOTS_SIM_EMBODIMENT"

# Fallbacks for ad-hoc (instruction) runs when neither flag nor config decides.
ADHOC_SCORER_FALLBACK = "operator"
ADHOC_MAX_STEPS_FALLBACK = 300


def parse_value(text: str) -> Any:
    """Best-effort scalar parse for ``k=v`` args (bool/int/float/None/str)."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


@dataclass(frozen=True)
class Defaults:
    """Resolved user defaults, each with a human-readable source for the run header."""

    policy: str | None = None
    policy_source: str | None = None
    embodiment: str | None = None
    embodiment_source: str | None = None
    # The sim counterpart --sim swaps in; real hardware is just the default
    # `embodiment`. Args are kept separate: real-rig args (ports, camera
    # serials) are wrong for a sim and vice versa.
    sim_embodiment: str | None = None
    sim_embodiment_source: str | None = None
    scorer: str | None = None
    max_steps: int | None = None
    policy_args: dict[str, Any] = field(default_factory=dict)
    embodiment_args: dict[str, Any] = field(default_factory=dict)
    sim_embodiment_args: dict[str, Any] = field(default_factory=dict)


def _config_path(env: Mapping[str, str]) -> Path | None:
    """The user config file location, derived from ``env`` only (testable)."""
    if xdg := env.get("XDG_CONFIG_HOME"):
        home = Path(xdg)
    elif user_home := env.get("HOME"):
        home = Path(user_home) / ".config"
    else:
        return None
    return home / "inspect-robots" / "config.ini"


def _die(path: Path, problem: str) -> SystemExit:
    return SystemExit(f"error in {path}: {problem}")


def _parse_args_section(parser: configparser.ConfigParser, section: str) -> dict[str, Any]:
    """An ``[<kind>.args]`` section as parsed kwargs, with ``~`` paths expanded."""
    if not parser.has_section(section):
        return {}
    out: dict[str, Any] = {}
    for key, raw in parser.items(section):
        value = parse_value(raw)
        if isinstance(value, str) and value.startswith("~"):
            # Checkpoint paths are the flagship use; a literal "~/..." string
            # would fail silently deep inside a plugin.
            value = os.path.expanduser(value)
        out[key] = value
    return out


def _read_config(path: Path) -> Defaults:
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    try:
        with path.open(encoding="utf-8") as fh:
            parser.read_file(fh)
    except (configparser.Error, UnicodeDecodeError) as exc:
        raise _die(path, f"malformed config: {exc}") from exc

    source = str(path)
    max_steps: int | None = None
    if raw_steps := parser.get("defaults", "max_steps", fallback=None):
        parsed = parse_value(raw_steps)
        if not isinstance(parsed, int) or isinstance(parsed, bool) or parsed < 1:
            raise _die(path, f"[defaults] max_steps must be an integer >= 1, got {raw_steps!r}")
        max_steps = parsed

    policy = parser.get("defaults", "policy", fallback=None)
    embodiment = parser.get("defaults", "embodiment", fallback=None)
    sim_embodiment = parser.get("defaults", "sim_embodiment", fallback=None)
    return Defaults(
        policy=policy,
        policy_source=source if policy else None,
        embodiment=embodiment,
        embodiment_source=source if embodiment else None,
        sim_embodiment=sim_embodiment,
        sim_embodiment_source=source if sim_embodiment else None,
        scorer=parser.get("defaults", "scorer", fallback=None),
        max_steps=max_steps,
        policy_args=_parse_args_section(parser, "policy.args"),
        embodiment_args=_parse_args_section(parser, "embodiment.args"),
        sim_embodiment_args=_parse_args_section(parser, "sim_embodiment.args"),
    )


def load_defaults(env: Mapping[str, str]) -> Defaults:
    """Load user defaults: environment variables override the config file.

    ``env`` is injected (pass ``os.environ``) so tests never touch the real
    home directory. A missing config file yields empty defaults; a malformed
    or type-invalid one raises ``SystemExit`` naming the file — the
    zero-config path must never print a traceback at the user.
    """
    from dataclasses import replace

    path = _config_path(env)
    defaults = _read_config(path) if path is not None and path.is_file() else Defaults()

    if policy := env.get(ENV_POLICY):
        defaults = replace(defaults, policy=policy, policy_source=f"${ENV_POLICY}")
    if embodiment := env.get(ENV_EMBODIMENT):
        defaults = replace(defaults, embodiment=embodiment, embodiment_source=f"${ENV_EMBODIMENT}")
    if sim := env.get(ENV_SIM_EMBODIMENT):
        defaults = replace(
            defaults, sim_embodiment=sim, sim_embodiment_source=f"${ENV_SIM_EMBODIMENT}"
        )
    return defaults
