from __future__ import annotations

from collections.abc import Iterator

import pytest
from _stub_server import StubPolicyServer


@pytest.fixture()
def stub_server() -> Iterator[StubPolicyServer]:
    server = StubPolicyServer()
    yield server
    server.stop()
