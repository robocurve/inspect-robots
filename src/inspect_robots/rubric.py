"""Rubrics: what a trial is judged and scored against.

The paper defines a task as a scene, an instruction, one or more rubrics, a time
budget and a number of trials. A [`Rubric`][inspect_robots.rubric.Rubric] names
one thing to measure: its ``text`` is what a judge (a person or a model) reads,
its ``kind`` groups results (``"capability"`` or ``"safety"`` are the usual
two), and its ``scorer`` turns the recorded trial into a score. Results are keyed
by rubric name, so two rubrics may share a scorer type, for example one operator
judgement for task success and another for safe conduct.

A task's rubrics apply to every scene; a scene's own rubrics replace the task's
rubric of the same name, or add a new one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from inspect_robots.errors import ConfigError

if TYPE_CHECKING:
    from inspect_robots.scorer import Scorer

CAPABILITY = "capability"
SAFETY = "safety"

# Rubric names key results in logs and metrics, so they stay simple identifiers.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


@dataclass(frozen=True)
class Rubric:
    """One criterion a trial is measured against, and how it is scored.

    ``name`` keys this rubric's results (letters, digits, ``_``, ``-``, ``.``).
    ``text`` is the criterion in plain language for a human or model judge;
    leave it empty for an automatic check. ``kind`` groups results
    (``"capability"`` by default, or ``"safety"``). ``scorer`` is a scorer object
    or a registry name. ``metadata`` is free-form and is logged.
    """

    name: str
    scorer: Scorer | str
    text: str = ""
    kind: str = CAPABILITY
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ConfigError(
                f"rubric name {self.name!r} is not a simple identifier.\n"
                "fix: use letters, digits, '_', '-' or '.', starting with a letter or digit"
            )
        if not isinstance(self.text, str):
            raise ConfigError(f"rubric {self.name!r}: text must be a string")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ConfigError(f"rubric {self.name!r}: kind must be a non-empty string")
        if isinstance(self.scorer, str) and not self.scorer.strip():
            raise ConfigError(f"rubric {self.name!r}: scorer name must not be empty")

    def resolve_scorer(self) -> Scorer:
        """Return the scorer object, resolving a registry name."""
        if isinstance(self.scorer, str):
            from inspect_robots.registry import resolve

            return cast("Scorer", resolve("scorer", self.scorer))
        return self.scorer

    def describe(self) -> dict[str, Any]:
        """The JSON-safe description recorded in logs: everything but the scorer object.

        Metadata is free-form, so it is made JSON-safe the way scene metadata is:
        NumPy numbers become Python ones and values JSON cannot encode (arrays,
        paths, objects) are omitted, so a log can always be written.
        """
        from inspect_robots.log import _json_safe_scene_metadata

        scorer = self.scorer if isinstance(self.scorer, str) else self.resolve_scorer().name
        return {
            "name": self.name,
            "kind": self.kind,
            "text": self.text,
            "scorer": scorer,
            "metadata": _json_safe_scene_metadata(self.metadata),
        }


def check_unique(rubrics: Sequence[Rubric], owner: str) -> None:
    """Fail with a guided error when two rubrics in one list share a name."""
    seen: set[str] = set()
    for rubric in rubrics:
        if not isinstance(rubric, Rubric):
            raise ConfigError(f"{owner}: rubrics must be Rubric objects, got {rubric!r}")
        if rubric.name in seen:
            raise ConfigError(f"{owner}: duplicate rubric name {rubric.name!r}")
        seen.add(rubric.name)


def merge(task_rubrics: Sequence[Rubric], scene_rubrics: Sequence[Rubric]) -> tuple[Rubric, ...]:
    """A scene's effective rubrics: the task's, with same-named scene rubrics replacing them.

    Task order is kept; scene rubrics with new names follow in their own order.
    """
    overrides = {rubric.name: rubric for rubric in scene_rubrics}
    merged = [overrides.pop(rubric.name, rubric) for rubric in task_rubrics]
    merged.extend(rubric for rubric in scene_rubrics if rubric.name in overrides)
    return tuple(merged)
