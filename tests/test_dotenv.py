"""Tests for dependency-free working-directory dotenv loading."""

import os
from pathlib import Path

import pytest

from inspect_robots._dotenv import init_dotenv, read_dotenv


def test_read_dotenv_parses_supported_syntax(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        """
          # indented comment
        PLAIN = value
        export EXPORTED=available
        WITH_EQUALS=first=second
        SINGLE='single quoted'
        DOUBLE="double quoted"
        EMPTY=
        SAME_ENDS=aba
        LITERAL_ESCAPE="left\\nright"
        UNMATCHED='left alone
        MISSING_EQUALS
        =empty key
        DUPLICATE=first
        DUPLICATE=second
        """,
        encoding="utf-8",
    )

    assert read_dotenv(path) == {
        "PLAIN": "value",
        "EXPORTED": "available",
        "WITH_EQUALS": "first=second",
        "SINGLE": "single quoted",
        "DOUBLE": "double quoted",
        "EMPTY": "",
        "SAME_ENDS": "aba",
        "LITERAL_ESCAPE": r"left\nright",
        "UNMATCHED": "'left alone",
        "DUPLICATE": "second",
    }


def test_read_dotenv_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_read_dotenv_returns_empty_for_os_error(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path) == {}


def test_init_dotenv_sets_absent_keys_without_overriding_present(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("NEW=from-file\nPRESENT=from-file\n", encoding="utf-8")
    environ = {"PRESENT": "from-environment"}

    init_dotenv(environ, path)

    assert environ == {"NEW": "from-file", "PRESENT": "from-environment"}


def test_init_dotenv_uses_working_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("FROM_CWD=yes\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    environ: dict[str, str] = {}

    init_dotenv(environ)

    assert environ == {"FROM_CWD": "yes"}


def test_cli_main_loads_working_directory_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "INSPECT_ROBOTS_TEST_DOTENV_SENTINEL"
    (tmp_path / ".env").write_text(f"{sentinel}=loaded\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(sentinel, raising=False)

    from inspect_robots.cli import main

    assert main(["list"]) == 0
    assert os.environ[sentinel] == "loaded"
