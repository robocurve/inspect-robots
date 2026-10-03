"""Changelog fragments in ``changelog.d/`` must be ones towncrier will render.

towncrier silently drops a fragment whose type is misspelled, so a typo would
lose a release-note entry without any error. This checks every fragment name
and keeps the type list in step with ``[tool.towncrier]`` in pyproject.toml.
The test reads pyproject.toml as text: tomllib is 3.11+ and the repo supports
3.10.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS = ROOT / "changelog.d"
TYPES = ("added", "changed", "deprecated", "removed", "fixed", "security")
NON_FRAGMENTS = {"README.md", ".gitkeep"}
NAME = re.compile(r"^(\d+|\+[a-z0-9][a-z0-9-]*)\.(" + "|".join(TYPES) + r")(\.\d+)?\.md$")


def fragment_problems(directory: Path) -> list[str]:
    """Describe every file in ``directory`` that towncrier would mishandle."""
    problems = []
    for path in sorted(directory.iterdir()):
        if path.name in NON_FRAGMENTS or path.name.startswith("."):
            continue
        if not path.is_file() or not NAME.match(path.name):
            problems.append(f"{path.name}: not <issue>.<type>.md or +<slug>.<type>.md")
        elif not path.read_text(encoding="utf-8").strip():
            problems.append(f"{path.name}: empty")
    return problems


def test_type_list_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    configured = re.findall(r'^\[\[tool\.towncrier\.type\]\]\s*\ndirectory = "(\w+)"', text, re.M)
    assert configured, "no [[tool.towncrier.type]] tables found"
    assert tuple(configured) == TYPES


def test_fragments_are_well_formed() -> None:
    if not FRAGMENTS.is_dir():
        pytest.skip("changelog.d/ is not shipped in the sdist")
    assert fragment_problems(FRAGMENTS) == []


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("440.fixd.md", "typo in type"),
        ("440.fixed.txt", "wrong extension"),
        ("+My_Fix.added.md", "slug with capitals and underscore"),
        ("fix.added.md", "neither an issue number nor an orphan"),
        ("441.added.md", "   \n"),
    ],
)
def test_bad_fragments_are_reported(tmp_path: Path, name: str, content: str) -> None:
    (tmp_path / name).write_text(content, encoding="utf-8")
    assert len(fragment_problems(tmp_path)) == 1


@pytest.mark.parametrize(
    "name", ["440.fixed.md", "440.fixed.1.md", "+doctor-device-slots.added.md", "README.md"]
)
def test_good_fragments_pass(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("**Core:** an entry.\n", encoding="utf-8")
    assert fragment_problems(tmp_path) == []
