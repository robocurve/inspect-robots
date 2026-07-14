"""Suite-wide fixtures keeping tests hermetic against the developer's machine."""

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep a repo-root ``.env`` out of ``os.environ`` during CLI tests.

    ``cli.main()`` loads ``./.env`` on every call; without this fixture a
    developer's real keys would leak into the test process and make
    default-resolution assertions depend on their machine. The dotenv wiring
    test opts back in by restoring the real ``init_dotenv``.
    """
    import inspect_robots.cli

    def _noop(environ: Any, path: Any = None) -> None:
        return None

    monkeypatch.setattr(inspect_robots.cli, "init_dotenv", _noop)
    yield
