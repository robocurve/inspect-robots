from __future__ import annotations

import pytest
from _dream_fakes import Clock

from inspect_robots import Scene


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def scene() -> Scene:
    return Scene(id="cube", instruction="put the red cube on the plate")
