from __future__ import annotations

import pytest
from _stub_model import StubOpenVLAModel, StubProcessor


@pytest.fixture()
def stub_processor() -> StubProcessor:
    return StubProcessor()


@pytest.fixture()
def stub_model() -> StubOpenVLAModel:
    return StubOpenVLAModel()
