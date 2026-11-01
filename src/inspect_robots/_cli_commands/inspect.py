"""The ``inspect-robots inspect`` command: print a saved eval log, with optional
policy transcripts and captured wire calls.."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from inspect_robots._html import _display_status
from inspect_robots._pointers import derive_blob_dir, read_jsonl_prefix, resolve_log_pointer

from ._shared import _DIM, _format_metric, _styled
from .summary import (
    _has_policy_transcripts,
    _outcome_line,
    _print_policy_transcripts,
    _print_step_limit_notice,
    _print_survivor_warning,
    _seconds_horizon_text,
)

if TYPE_CHECKING:
    from inspect_robots.log import EvalLog


class _WireTrial(NamedTuple):
    """A readable wire sidecar and its trial/run filesystem context."""

    trial_id: str
    rows: list[dict[str, Any]]
    blob_dir: Path


_WIRE_BLOB_RE = re.compile(r"\$blob:([0-9a-f]{64})")


def _wire_blob_shas(value: object) -> list[str]:
    """Return every valid symbolic blob reference in a captured JSON value."""
    if isinstance(value, str):
        return [match.group(1) for match in _WIRE_BLOB_RE.finditer(value)]
    if isinstance(value, list):
        return [sha for item in value for sha in _wire_blob_shas(item)]
    if isinstance(value, dict):
        return [sha for item in value.values() for sha in _wire_blob_shas(item)]
    return []


def _load_wire_trials(log: EvalLog, log_path: Path) -> list[_WireTrial]:
    """Load every readable guarded wire sidecar referenced by an eval log."""
    trials: list[_WireTrial] = []
    for scene in log.samples:
        for epoch in range(len(scene.epochs)):
            if epoch >= len(scene.trial_metadata):
                continue
            target = resolve_log_pointer(log_path, scene.trial_metadata[epoch].get("wire_capture"))
            if target is None:
                continue
            loaded = read_jsonl_prefix(target)
            if loaded is None:
                continue
            blob_dir = derive_blob_dir(log_path, target)
            if blob_dir is None:
                continue
            trials.append(
                _WireTrial(
                    f"{scene.scene_id}-e{epoch}",
                    loaded,
                    blob_dir,
                )
            )
    return trials


def _wire_request(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row's request mapping, degrading malformed foreign data."""
    request = row.get("request")
    return cast(dict[str, Any], request) if isinstance(request, dict) else {}


def _print_wire_table(trials: list[_WireTrial]) -> None:
    """Print the compact all-trials wire attempt table."""
    if not trials:
        print("no wire capture recorded")
        return
    print("wire calls:")
    print("  trial  call  attempt  endpoint  status  duration  images  new blob bytes")
    for trial in trials:
        seen: set[str] = set()
        for row in trial.rows:
            shas = _wire_blob_shas(_wire_request(row))
            first_shas = set(shas).difference(seen)
            seen.update(shas)
            new_bytes = 0
            for sha in first_shas:
                with suppress(OSError):
                    new_bytes += (trial.blob_dir / f"{sha}.png").stat().st_size
            duration = row.get("duration_s")
            shown_duration = (
                f"{duration:.3f}s"
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else "-"
            )
            status = "-" if row.get("status") is None else str(row["status"])
            print(
                f"  {trial.trial_id}  {row.get('call', '-')}  "
                f"{row.get('attempt', '-')}  {row.get('endpoint', '-')}  "
                f"{status}  {shown_duration}  {len(shas)}  {new_bytes}"
            )


def _available_wire_trials(trials: list[_WireTrial]) -> str:
    """Format captured trial ids for a guided CLI error."""
    return ", ".join(trial.trial_id for trial in trials)


def _print_wire_call(trials: list[_WireTrial], call: int, selected_trial: str | None) -> None:
    """Pretty-print every captured attempt for one logical call index."""

    from inspect_robots import cli

    if not trials:
        raise SystemExit("no wire capture recorded; omit CALL to print an empty table")
    if selected_trial is None:
        if len(trials) > 1:
            raise SystemExit(
                "--trial is required for --wire CALL; available trials: "
                f"{_available_wire_trials(trials)}"
            )
        trial = trials[0]
    else:
        matches = [trial for trial in trials if trial.trial_id == selected_trial]
        if not matches:
            raise SystemExit(
                f"wire trial {selected_trial!r} not found; available trials: "
                f"{_available_wire_trials(trials)}"
            )
        trial = matches[0]

    rows = [row for row in trial.rows if row.get("call") == call]
    if not rows:
        raise SystemExit(f"wire call {call} not found in trial {trial.trial_id}")
    print(f"wire trial {trial.trial_id}, call {call}:")
    for row in rows:
        print(f"attempt {row.get('attempt', '-')}:")
        print(f"  endpoint: {row.get('endpoint', '-')}")
        print(f"  status: {row.get('status')}")
        print(f"  duration_s: {row.get('duration_s', '-')}")
        if "error" in row:
            cli._print_degraded(f"  error: {row['error']}")
        for label in ("request", "response"):
            cli._print_degraded(f"  {label}:")
            dumped = json.dumps(row.get(label), indent=2, sort_keys=True, ensure_ascii=False)
            for line in dumped.splitlines():
                cli._print_degraded(f"    {line}")


def _print_wire_capture(
    log: EvalLog,
    log_path: Path,
    wire: int | bool,
    selected_trial: str | None,
) -> None:
    """Print either the capture table or every attempt for one call."""
    trials = _load_wire_trials(log, log_path)
    if wire is True:
        if selected_trial is not None:
            raise SystemExit("--trial requires an integer --wire CALL")
        _print_wire_table(trials)
        return
    _print_wire_call(trials, wire, selected_trial)


def _cmd_inspect(
    path: str,
    *,
    transcript: bool = False,
    wire: int | bool | None = None,
    trial: str | None = None,
) -> int:
    from inspect_robots import cli, read_eval_log

    log = read_eval_log(path)
    _print_step_limit_notice(log, log.eval.task in ("adhoc", "auto"))
    print(f"task:        {log.eval.task}")
    # One shared instruction (the adhoc case) reads as run-level identity;
    # differing instructions print per scene below instead. Instructions are
    # foreign text (dataset/operator-supplied), so printing degrades.
    instructions = {scene.instruction for scene in log.samples}
    shared = next(iter(instructions)) if len(instructions) == 1 else None
    if shared:
        cli._print_degraded(f"instruction: {shared}")
    print(f"policy:      {log.eval.policy}")
    print(f"embodiment:  {log.eval.embodiment}")
    print(f"run status:  {_display_status(log.status)}")
    outcome = _outcome_line(log)
    if outcome is not None:
        digest, has_unmapped = outcome
        line = f"outcome:     {digest}"
        if has_unmapped:
            cli._print_degraded(line)
        else:
            print(line)
    print(f"created:     {log.eval.created}")
    print(f"git:         {log.eval.git_commit}")
    horizon = _seconds_horizon_text(log)
    if horizon is not None:
        print(f"horizon:     {horizon}")
    trials = f"trials: {log.results.total_trials}"
    if log.results.errored_trials:
        trials += f" ({log.results.errored_trials} errored)"
    print(f"scenes:      {log.results.total_scenes}   {trials}")
    _print_survivor_warning(log)
    if log.stats.frames_dir is not None:
        from inspect_robots._video import count_frames, resolve_frames_dir

        root = resolve_frames_dir(log.stats.frames_dir, Path(path))
        if root is None:
            print(f"frames:      {log.stats.frames_dir} (not found from this directory)")
        else:
            # The resolved path, not the stored string: after a machine move
            # the stored string is exactly the path that does not work.
            n_frames = count_frames(root)
            plural = "frame" if n_frames == 1 else "frames"
            print(f"frames:      {root} ({n_frames} {plural})")
            if n_frames:
                print(_styled(f"hint: render videos with: inspect-robots video {path}", _DIM))
    print("metrics:")
    for name, value in sorted(log.results.metrics.items()):
        abstained = log.results.abstentions.get(name, 0)
        suffix = f" ({abstained} abstained)" if abstained else ""
        print(f"  {name}: {_format_metric(value)}{suffix}")
    print("scenes:")
    for scene in log.samples:
        reduced = "  ".join(f"{k}={_format_metric(v)}" for k, v in sorted(scene.reduced.items()))
        step_limit_count = sum(reason == "max_steps" for reason in scene.termination_reasons)
        details = [reduced] if reduced else []
        if step_limit_count:
            details.append(f"({step_limit_count}/{len(scene.epochs)} trials hit max_steps)")
        print(f"  [{scene.status}] {scene.scene_id}: {'  '.join(details)}")
        if not shared and scene.instruction:
            cli._print_degraded(f"      instruction: {scene.instruction}")
    if log.error:
        print(f"error: {log.error}")
    if transcript:
        _print_policy_transcripts(log)
    elif _has_policy_transcripts(log):
        print("policy transcripts: recorded (--transcript to print)")
        print(_styled(f"hint: HTML viewer: inspect-robots view {path}", _DIM))
    if wire is not None:
        _print_wire_capture(log, Path(path), wire, trial)
    elif trial is not None:
        raise SystemExit("--trial requires --wire CALL")
    return 0 if log.status == "success" else 1
