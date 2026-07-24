"""Registry and decorators for tasks, policies, embodiments, scorers, and sinks.

Mirrors Inspect AI's extension model: components register by name via decorators
and are resolved from strings (so ``eval(policy="scripted")`` and the CLI work).
Out-of-tree packages publish components through ``importlib.metadata`` entry-point
groups, so an installed ``inspect-robots-openvla`` appears in ``inspect-robots list`` without
being imported first.

Entry-point groups:
``inspect_robots.tasks``, ``inspect_robots.policies``, ``inspect_robots.embodiments``,
``inspect_robots.scorers``, ``inspect_robots.sinks``.

Set ``INSPECT_ROBOTS_DISABLE_PLUGIN_AUTOLOAD`` to any non-empty value to skip
entry-point discovery: only in-tree builtins and components registered by hand
are then resolvable. This is a defense-in-depth and reproducibility switch for
locked-down eval environments, mirroring pytest's
``PYTEST_DISABLE_PLUGIN_AUTOLOAD``. It is not a security boundary: an installed
package can still run code when imported. The switch only stops this framework
from importing plugins on your behalf during discovery.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

Kind = str  # "task" | "policy" | "embodiment" | "scorer" | "sink"
KINDS: tuple[Kind, ...] = ("task", "policy", "embodiment", "scorer", "sink")

# Any non-empty value opts out of entry-point plugin autoloading (see module
# docstring). Read at discovery time, not import time, so it stays togglable.
DISABLE_AUTOLOAD_ENV = "INSPECT_ROBOTS_DISABLE_PLUGIN_AUTOLOAD"

_GROUPS: dict[Kind, str] = {
    "task": "inspect_robots.tasks",
    "policy": "inspect_robots.policies",
    "embodiment": "inspect_robots.embodiments",
    "scorer": "inspect_robots.scorers",
    "sink": "inspect_robots.sinks",
}

_FACTORIES: dict[Kind, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}
_loaded_entrypoints = False
_loaded_builtins = False

F = TypeVar("F", bound=Callable[..., Any])


def register(kind: Kind, name: str | None = None) -> Callable[[F], F]:
    """Register a factory under ``kind``/``name`` (defaults to its ``__name__``)."""
    if kind not in _FACTORIES:
        raise ValueError(f"unknown registry kind {kind!r}; valid: {KINDS}")

    def decorator(factory: F) -> F:
        key = name or getattr(factory, "__name__", None)
        if key is None:
            raise ValueError("cannot determine a registry name for the factory")
        _FACTORIES[kind][key] = factory
        return factory

    return decorator


def task(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a task factory under ``name``."""
    return register("task", name)


def policy(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a policy factory under ``name``."""
    return register("policy", name)


def embodiment(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register an embodiment factory under ``name``."""
    return register("embodiment", name)


def scorer(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a scorer factory under ``name``."""
    return register("scorer", name)


def sink(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a log-sink factory under ``name``."""
    return register("sink", name)


def _autoload_disabled() -> bool:
    """Whether entry-point plugin autoloading is turned off via the environment."""
    return bool(os.environ.get(DISABLE_AUTOLOAD_ENV))


def _ensure_loaded() -> None:
    global _loaded_builtins, _loaded_entrypoints
    if not _loaded_builtins:
        _loaded_builtins = True
        import inspect_robots._builtins  # noqa: F401  (registers builtin components)
    # The opt-out is deliberately not latched: it skips discovery without
    # marking it done, so clearing the env var later still loads plugins.
    if _loaded_entrypoints or _autoload_disabled():
        return
    _loaded_entrypoints = True
    for kind, group in _GROUPS.items():
        for ep in entry_points(group=group):
            try:
                factory = ep.load()
            except Exception as exc:
                # A broken plugin must not crash discovery, but it must not
                # vanish silently either — that is undebuggable.
                warnings.warn(
                    f"failed to load inspect_robots plugin {ep.name!r} from "
                    f"entry-point group {group!r}: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            _FACTORIES[kind].setdefault(ep.name, factory)


def registered(kind: Kind) -> dict[str, Callable[..., Any]]:
    """Return all registered factories for ``kind`` (builtins + plugins)."""
    if kind not in _FACTORIES:
        raise ValueError(f"unknown registry kind {kind!r}; valid: {KINDS}")
    _ensure_loaded()
    return dict(_FACTORIES[kind])


def resolve(kind: Kind, name: str, /, **kwargs: Any) -> Any:
    """Construct a registered component by name with the given keyword args."""
    factories = registered(kind)
    if name not in factories:
        raise KeyError(f"no {kind} named {name!r}; available: {sorted(factories)}")
    return factories[name](**kwargs)
