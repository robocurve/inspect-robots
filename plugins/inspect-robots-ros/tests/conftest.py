from __future__ import annotations

from collections.abc import Iterator

import pytest
from _stub_server import StubRosbridgeServer


@pytest.fixture()
def stub_server() -> Iterator[StubRosbridgeServer]:
    server = StubRosbridgeServer()
    yield server
    server.stop()
