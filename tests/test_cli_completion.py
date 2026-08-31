"""Unit tests for `inspect-robots completion` subcommand helpers."""

from __future__ import annotations

import contextlib
import io

from inspect_robots.cli import (
    _BASH_COMPLETION_PARTS,
    _SUBCOMMANDS,
    _cmd_completion,
    _completion_zsh,
)


def test_bash_completion_lists_every_subcommand() -> None:
    body = "\n".join(_BASH_COMPLETION_PARTS(" ".join(_SUBCOMMANDS)))
    assert "_inspect_robots() {" in body
    assert "# inspect-robots bash completion" in body
    assert '# Install: eval "$(inspect-robots completion bash)"' in body
    for sub in _SUBCOMMANDS:
        assert sub in body, f"subcommand {sub!r} missing from bash script"


def test_zsh_completion_has_compdef_header() -> None:
    body = _completion_zsh()
    assert body.startswith("#compdef inspect-robots")
    assert '# Install: eval "$(inspect-robots completion zsh)"' in body
    for sub in _SUBCOMMANDS:
        assert sub in body, f"subcommand {sub!r} missing from zsh script"


def test_cmd_completion_bash_writes_to_stdout() -> None:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        rc = _cmd_completion(type("Args", (), {"shell": "bash"})())
    assert rc == 0
    assert sink.getvalue().startswith("# inspect-robots bash completion")


def test_cmd_completion_unsupported_returns_2() -> None:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _cmd_completion(type("Args", (), {"shell": "fish"})())
    assert rc == 2
    assert "unsupported shell" in err.getvalue()
