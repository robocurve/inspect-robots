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


def test_read_dotenv_strips_unquoted_inline_comments(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        'KEY=sk-123 # prod key\nQUOTED="value # kept"\nHASH_VALUE=#not-a-comment\nANCHOR=a#b\n',
        encoding="utf-8",
    )

    assert read_dotenv(path) == {
        "KEY": "sk-123",
        "QUOTED": "value # kept",
        "HASH_VALUE": "#not-a-comment",
        "ANCHOR": "a#b",
    }


def test_read_dotenv_strips_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"\xef\xbb\xbfANTHROPIC_API_KEY=abc\n")

    assert read_dotenv(path) == {"ANTHROPIC_API_KEY": "abc"}


def test_read_dotenv_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_read_dotenv_returns_empty_for_os_error(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path) == {}


def test_read_dotenv_returns_empty_for_undecodable_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"KEY=\xff\xfe\n")

    assert read_dotenv(path) == {}


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
    # setenv-then-delenv records both operations, so teardown restores absence
    # (delenv on a missing key records nothing and the value set by main()
    # would leak into the rest of the session).
    monkeypatch.setenv(sentinel, "preexisting")
    monkeypatch.delenv(sentinel)

    import inspect_robots.cli

    monkeypatch.setattr(inspect_robots.cli, "init_dotenv", init_dotenv)

    assert inspect_robots.cli.main(["list"]) == 0
    assert os.environ[sentinel] == "loaded"
