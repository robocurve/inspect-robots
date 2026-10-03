"""Suite-wide fixtures keeping tests hermetic against the developer's machine."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_LOGS_DIR = _REPO_ROOT / "logs"
_REPO_LOGS_EXISTED_AT_SESSION_START = False


def pytest_sessionstart(session: pytest.Session) -> None:
    """Snapshot normal developer logs before tests can create checkout litter."""
    del session
    global _REPO_LOGS_EXISTED_AT_SESSION_START
    _REPO_LOGS_EXISTED_AT_SESSION_START = _REPO_LOGS_DIR.exists()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail and clean up only when the suite created the repo-root logs directory."""
    del exitstatus
    if not _REPO_LOGS_EXISTED_AT_SESSION_START and _REPO_LOGS_DIR.exists():
        print(f"test suite created repo-root log litter: {_REPO_LOGS_DIR}")
        session.exitstatus = 1
        shutil.rmtree(_REPO_LOGS_DIR)


@pytest.fixture(autouse=True)
def _enable_plugin_autoload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear a developer's exported plugin-autoload opt-out for every test.

    Discovery tests assume entry points load; an inherited
    ``INSPECT_ROBOTS_DISABLE_PLUGIN_AUTOLOAD`` would make them depend on the
    machine. Tests of the opt-out set it explicitly.
    """
    monkeypatch.delenv("INSPECT_ROBOTS_DISABLE_PLUGIN_AUTOLOAD", raising=False)


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep a repo-root ``.env`` out of ``os.environ`` during CLI tests.

    ``cli.main()`` loads ``./.env`` on every call; without this fixture a
    developer's real keys would leak into the test process and make
    default-resolution assertions depend on their machine. The dotenv wiring
    test opts back in by restoring the real ``init_dotenv``. Patching only
    the ``inspect_robots.cli`` name assumes ``main()`` stays the sole entry
    point that loads ``.env`` — extend this fixture if another appears.
    """
    import inspect_robots.cli

    def _noop(environ: Any, path: Any = None) -> None:
        return None

    monkeypatch.setattr(inspect_robots.cli, "init_dotenv", _noop)
    yield
