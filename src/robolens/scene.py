"""Scenes — the robotics analog of Inspect AI's ``Sample``.

A :class:`Scene` is one initial condition of a benchmark: a language instruction,
an optional success ``Target``, an optional seed, and metadata. A benchmark
``Task`` iterates over a dataset of scenes (e.g. 50 object layouts), repeated
``epochs`` times.

Field mapping to Inspect: ``Sample(input, target, id, metadata, setup)`` ↔
``Scene(instruction, target, id, metadata, setup, init_seed)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Target:
    """A success specification the scorer reads. Embodiment-namespaced.

    ``kind`` names what the embodiment must realize/evaluate (e.g.
    ``"reach_object"``); ``spec`` carries the parameters. Kept intentionally open
    for the tracer; richer typed targets land with the scorer milestone.
    """

    kind: str
    spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene:
    """One initial condition of a benchmark."""

    id: str
    instruction: str
    target: Target | None = None
    init_seed: int | None = None
    setup: str | None = None  # registered setup-hook name (serializable)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ListSceneDataset:
    """A trivial in-memory scene dataset backed by a sequence."""

    def __init__(self, scenes: Sequence[Scene]):
        self._scenes = list(scenes)

    def __iter__(self) -> Iterator[Scene]:
        return iter(self._scenes)

    def __len__(self) -> int:
        return len(self._scenes)
