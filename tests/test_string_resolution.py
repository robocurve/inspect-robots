"""eval()/eval_set() registry-string resolution and the inspect CLI."""

from __future__ import annotations

from pathlib import Path

from roboinspect import eval, eval_set
from roboinspect.cli import main


def test_eval_resolves_all_strings(tmp_path: Path) -> None:
    logs = eval(
        "cubepick-reach",
        "scripted",
        "cubepick",
        log_dir=str(tmp_path),
    )
    assert logs[0].status == "success"
    assert logs[0].results.metrics["success_at_end"] == 1.0


def test_eval_set_resolves_strings(tmp_path: Path) -> None:
    success, logs = eval_set(
        ["cubepick-reach", "cubepick-reach"],
        "scripted",
        "cubepick",
        log_dir=str(tmp_path),
    )
    assert success is True
    assert len(logs) == 2


def test_cli_inspect(tmp_path: Path, capsys: object) -> None:
    eval("cubepick-reach", "scripted", "cubepick", log_dir=str(tmp_path))
    (log_path,) = tmp_path.glob("*.json")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["inspect", str(log_path)])
    assert rc == 0
    out = buf.getvalue()
    assert "status:      success" in out
    assert "success_at_end" in out
    assert "scenes:" in out
