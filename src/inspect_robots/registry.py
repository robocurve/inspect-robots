"""Registry and decorators for tasks, policies, embodiments, scorers, and sinks.

Mirrors Inspect AI's extension model: components register by name via decorators
and are resolved from strings (so ``eval(policy="scripted")`` and the CLI work).
Out-of-tree packages publish components through ``importlib.metadata`` entry-point
groups, so an installed ``inspect-robots-openvla`` appears in ``inspect-robots list`` without
being imported first.

Entry-point groups:
``inspect_robots.tasks``, ``inspect_robots.policies``, ``inspect_robots.embodiments``,
``inspect_robots.scorers``, ``inspect_robots.sinks``.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

Kind = str  # "task" | "policy" | "embodiment" | "scorer" | "sink"
KINDS: tuple[Kind, ...] = ("task", "policy", "embodiment", "scorer", "sink")

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


def _ensure_loaded() -> None:
    global _loaded_builtins, _loaded_entrypoints
    if not _loaded_builtins:
        _loaded_builtins = True
        import inspect_robots._builtins  # noqa: F401  (registers builtin components)
    if not _loaded_entrypoints:
        _loaded_entrypoints = True
        for kind, group in _GROUPS.items():
            for ep in entry_points(group=group):
                try:
                    factory = ep.load()
                except Exception:
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
