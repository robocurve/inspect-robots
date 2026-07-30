"""Rebuild an offline Rerun recording from an Inspect Robots eval log."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

_FRAME_NAME = re.compile(r"^(?P<scene>.+)-e(?P<epoch>\d+)_(?P<cam>.+)_(?P<t>\d{6})\.npy$")
_CAMERA_STEP = re.compile(r"camera\s+'[^']+'\s+\(step\s+(\d+)\):")
_STATE_LINE = re.compile(r"^\s*state\[([^\]]+)\]:\s*\[([^\]]*)\]\s*$", re.MULTILINE)
_IMAGE_PLACEHOLDER = "[image omitted: streamed camera frame]"


@dataclass
class Counts:
    """Successful Rerun writes reported in the final summary."""

    frames: int = 0
    state_points: int = 0
    action_points: int = 0
    transcript_rows: int = 0


def _note(message: str) -> None:
    print(f"Note: {message}")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _item_at(value: Any, index: int) -> Any:
    values = _sequence(value)
    return values[index] if index < len(values) else None


def _set_step(t: int) -> None:
    if hasattr(rr, "set_time"):
        rr.set_time("step", sequence=t)
    else:
        rr.set_time_sequence("step", t)


def _scalar(value: float) -> Any:
    scalars = getattr(rr, "Scalars", None)
    if scalars is not None:
        return scalars(value)
    return rr.Scalar(value)


def _image(image: np.ndarray[Any, Any], jpeg_quality: int, warnings: set[str]) -> Any:
    rerun_image = rr.Image(image)
    compress = getattr(rerun_image, "compress", None)
    if compress is None:
        if "compress" not in warnings:
            warnings.add("compress")
            _note("this Rerun SDK has no Image.compress; logging raw frames")
        return rerun_image
    try:
        return compress(jpeg_quality=jpeg_quality)
    except Exception as exc:
        if "compress" not in warnings:
            warnings.add("compress")
            _note(f"JPEG compression failed ({exc}); logging raw frames")
        return rerun_image


def _text_parts(message: Any) -> list[str]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]


def _message_step(message: Any) -> int | None:
    if not isinstance(message, dict) or str(message.get("role", "")) != "user":
        return None
    for text in _text_parts(message):
        match = _CAMERA_STEP.search(text)
        if match is not None:
            return int(match.group(1))
    return None


def _without_image_placeholders(message: Any) -> Any:
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return message
    filtered = dict(message)
    filtered["content"] = [
        part
        for part in message["content"]
        if not (
            isinstance(part, dict)
            and part.get("type") == "text"
            and str(part.get("text", "")) == _IMAGE_PLACEHOLDER
        )
    ]
    return filtered


def _render_message(message: Any) -> tuple[str, str]:
    """Render one transcript row with the live Rerun sink's formatting."""
    if not isinstance(message, dict):
        return "INFO", str(message)

    role = str(message.get("role", "unknown"))
    level = {
        "assistant": "INFO",
        "user": "INFO",
        "tool": "DEBUG",
    }.get(role, "TRACE")
    lines: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        lines.append(content)
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                part_type = part.get("type", "unknown") if isinstance(part, dict) else "unknown"
                parts.append(f"[{part_type} part]")
        if parts:
            lines.append("\n".join(parts))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if not isinstance(function, dict):
                function = {}
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            lines.append(f"tool_call {name}({arguments})")
    return level, f"{role}: " + "\n".join(lines)


def _state_values(message: Any, message_index: int) -> list[tuple[str, list[float]]]:
    parsed: list[tuple[str, list[float]]] = []
    for text in _text_parts(message):
        for match in _STATE_LINE.finditer(text):
            key, raw_values = match.groups()
            try:
                values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
            except ValueError:
                _note(f"message {message_index} has an invalid state[{key}] vector; skipped")
                continue
            parsed.append((key, values))
    return parsed


def _action_values(message: Any, message_index: int) -> list[tuple[str, float]]:
    if not isinstance(message, dict) or str(message.get("role", "")) != "assistant":
        return []
    actions: list[tuple[str, float]] = []
    for call in _sequence(message.get("tool_calls")):
        function = _mapping(call).get("function")
        arguments = _mapping(function).get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            decoded = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            _note(f"message {message_index} has invalid tool arguments JSON; actions skipped")
            continue
        targets = _mapping(decoded).get("targets")
        if not isinstance(targets, dict):
            continue
        for index, value in targets.items():
            try:
                actions.append((str(index), float(value)))
            except (TypeError, ValueError):
                _note(f"message {message_index} has a non-numeric action target {index!r}; skipped")
    return actions


def _log_transcript_document(
    prefix: str,
    entries: list[tuple[str, str]],
    warnings: set[str],
) -> None:
    text_document = getattr(rr, "TextDocument", None)
    if text_document is None:
        if "text_document" not in warnings:
            warnings.add("text_document")
            _note("this Rerun SDK has no TextDocument; llm/latest and run_info are skipped")
        return
    body = "\n\n---\n\n".join(
        f"**[{level}]** " + text.replace("\n", "  \n") for level, text in entries
    )
    rr.log(
        f"{prefix}/llm/latest",
        text_document(body, media_type="text/markdown"),
    )


def _replay_transcript(
    transcript: Any,
    prefix: str,
    counts: Counts,
    last_steps: dict[str, int],
    warnings: set[str],
) -> None:
    if not isinstance(transcript, list):
        return

    has_observation_steps = any(_message_step(message) is not None for message in transcript)
    current_step = 0
    entries_by_step: defaultdict[int, list[tuple[str, str]]] = defaultdict(list)

    for message_index, message in enumerate(transcript):
        parsed_step = _message_step(message)
        if parsed_step is not None:
            current_step = parsed_step
        step = current_step if has_observation_steps else message_index
        try:
            _set_step(step)
        except Exception as exc:
            _note(f"could not select step {step} for message {message_index} ({exc}); skipped")
            continue
        last_steps[prefix] = max(last_steps.get(prefix, 0), step)

        if isinstance(message, dict) and str(message.get("role", "")) == "user":
            for key, values in _state_values(message, message_index):
                for index, value in enumerate(values):
                    try:
                        rr.log(f"{prefix}/state/{key}/{index}", _scalar(value))
                    except Exception as exc:
                        _note(
                            f"could not log state[{key}][{index}] at step {step} ({exc}); skipped"
                        )
                    else:
                        counts.state_points += 1

        for index, value in _action_values(message, message_index):
            try:
                rr.log(f"{prefix}/action/{index}", _scalar(value))
            except Exception as exc:
                _note(f"could not log action[{index}] at step {step} ({exc}); skipped")
            else:
                counts.action_points += 1

        entry = _render_message(_without_image_placeholders(message))
        try:
            rr.log(f"{prefix}/llm", rr.TextLog(entry[1], level=entry[0]))
        except Exception as exc:
            _note(f"could not log transcript message {message_index} ({exc}); skipped")
            continue
        counts.transcript_rows += 1
        entries_by_step[step].append(entry)
        try:
            _log_transcript_document(prefix, entries_by_step[step], warnings)
        except Exception as exc:
            _note(f"could not update llm/latest at step {step} ({exc}); skipped")


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _run_info_markdown(
    log_path: Path,
    eval_info: dict[str, Any],
    stats: dict[str, Any],
    sample: dict[str, Any],
    epoch: int,
) -> str:
    config = _mapping(eval_info.get("policy_config"))
    policy_details = [
        f"model={_display(config.get('model'))}",
        f"effort={_display(config.get('effort'))}",
    ]
    lines = [
        "# Inspect Robots replay",
        "",
        f"- **Instruction:** {_display(sample.get('instruction'))}",
        f"- **Scene ID:** {_display(sample.get('scene_id'))}",
        f"- **Epoch:** {epoch}",
        f"- **Policy:** {_display(eval_info.get('policy'))} ({', '.join(policy_details)})",
        f"- **Embodiment:** {_display(eval_info.get('embodiment'))}",
        f"- **Started at:** {_display(stats.get('started_at'))}",
        f"- **Completed at:** {_display(stats.get('completed_at'))}",
        f"- **Duration (s):** {_display(stats.get('duration_s'))}",
        f"- **Total steps:** {_display(stats.get('total_steps'))}",
        f"- **Epoch scores:** {_display(_item_at(sample.get('epochs'), epoch))}",
        (
            "- **Operator judgement:** "
            f"{_display(_item_at(sample.get('operator_judgements'), epoch))}"
        ),
        f"- **Operator note:** {_display(_item_at(sample.get('operator_notes'), epoch))}",
        (
            "- **Termination reason:** "
            f"{_display(_item_at(sample.get('termination_reasons'), epoch))}"
        ),
    ]
    metadata = _mapping(_item_at(sample.get("trial_metadata"), epoch))
    llm_usage = metadata.get("llm_usage")
    if isinstance(llm_usage, dict):
        lines.append(f"- **LLM usage:** {_display(llm_usage)}")
    lines.extend(
        [
            (f"- **Inspect Robots version:** {_display(eval_info.get('inspect_robots_version'))}"),
            f"- **Log file:** {log_path.name}",
        ]
    )
    return "\n".join(lines)


def _epoch_count(sample: dict[str, Any]) -> int:
    epoch_fields = (
        "epochs",
        "operator_judgements",
        "operator_notes",
        "termination_reasons",
        "policy_transcripts",
        "trial_metadata",
    )
    return max([1, *[len(_sequence(sample.get(field))) for field in epoch_fields]])


def _log_run_info(
    log_path: Path,
    eval_info: dict[str, Any],
    stats: dict[str, Any],
    sample: dict[str, Any],
    epoch: int,
    prefix: str,
    warnings: set[str],
) -> None:
    text_document = getattr(rr, "TextDocument", None)
    if text_document is None:
        if "text_document" not in warnings:
            warnings.add("text_document")
            _note("this Rerun SDK has no TextDocument; llm/latest and run_info are skipped")
        return
    body = _run_info_markdown(log_path, eval_info, stats, sample, epoch)
    try:
        rr.log(
            f"{prefix}/run_info",
            text_document(body, media_type="text/markdown"),
            static=True,
        )
    except Exception as exc:
        _note(f"could not log run_info for {prefix} ({exc}); skipped")


def _safe_total_steps(stats: dict[str, Any]) -> int:
    value = stats.get("total_steps")
    try:
        return max(0, int(value)) if value is not None else 0
    except (TypeError, ValueError):
        _note(f"stats.total_steps is not an integer ({value!r}); ignoring it")
        return 0


def _resolve_frames_dir(
    log_path: Path,
    stats: dict[str, Any],
    frames_root: Path | None,
) -> Path | None:
    frames_value = stats.get("frames_dir")
    if not frames_value:
        print("Frames unavailable: stats.frames_dir is null or missing.")
        return None
    if not isinstance(frames_value, (str, Path)):
        _note(f"stats.frames_dir has an unsupported value {frames_value!r}")
        print("Frames unavailable: invalid stats.frames_dir.")
        return None
    frames_dir = Path(frames_value).expanduser()
    if not frames_dir.is_absolute():
        base = frames_root if frames_root is not None else log_path.parent.parent
        frames_dir = base / frames_dir
    if not frames_dir.is_dir():
        print(f"Frames unavailable: directory does not exist: {frames_dir}")
        return None
    return frames_dir


def _replay_frames(
    frames_dir: Path | None,
    jpeg_quality: int,
    counts: Counts,
    last_steps: dict[str, int],
    warnings: set[str],
) -> None:
    if frames_dir is None:
        return
    frames: list[tuple[str, int, str, int, Path]] = []
    for path in frames_dir.glob("*.npy"):
        match = _FRAME_NAME.match(path.name)
        if match is None:
            _note(f"skipping non-matching frame file: {path.name}")
            continue
        frames.append(
            (
                match.group("cam"),
                int(match.group("t")),
                match.group("scene"),
                int(match.group("epoch")),
                path,
            )
        )

    for camera, step, scene_id, epoch, path in sorted(frames, key=lambda item: (item[0], item[1])):
        prefix = f"trial/{scene_id}/e{epoch}"
        try:
            frame = np.load(path, allow_pickle=False)
            _set_step(step)
            rr.log(
                f"{prefix}/camera/{camera}",
                _image(frame, jpeg_quality, warnings),
            )
        except Exception as exc:
            _note(f"could not log frame {path.name} ({exc}); skipped")
            continue
        counts.frames += 1
        last_steps[prefix] = max(last_steps.get(prefix, 0), step)
    print(f"Frames logged: {counts.frames} from {frames_dir}")


def _termination_text(sample: dict[str, Any], epoch: int) -> str:
    reason = _item_at(sample.get("termination_reasons"), epoch)
    judgement = _item_at(sample.get("operator_judgements"), epoch)
    pieces: list[str] = []
    if reason is not None and str(reason):
        pieces.append(str(reason))
    if judgement is not None and str(judgement):
        pieces.append(f"operator judgement: {judgement}")
    return "; ".join(pieces) or "terminated"


def _finish_recording() -> None:
    get_recording = getattr(rr, "get_global_data_recording", None)
    if get_recording is not None:
        recording = get_recording()
        flush = getattr(recording, "flush", None)
        if flush is not None:
            flush()
    disconnect = getattr(rr, "disconnect", None)
    if disconnect is not None:
        disconnect()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=("replay_to_rerun.py LOG_JSON OUT_RRD [--frames-root DIR] [--jpeg-quality N]"),
        description="Rebuild a Rerun .rrd recording from an Inspect Robots eval log.",
    )
    parser.add_argument("log_json", type=Path, metavar="LOG_JSON")
    parser.add_argument("out_rrd", type=Path, metavar="OUT_RRD")
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="base directory for a relative stats.frames_dir",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        metavar="N",
        help="JPEG quality for camera frames (default: 75)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Replay one completed eval log into a standalone Rerun recording."""
    args = _parse_args(argv)
    log_path: Path = args.log_json
    out_path: Path = args.out_rrd

    try:
        raw_log = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read eval log {log_path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(raw_log, dict):
        print(f"Eval log root must be a JSON object: {log_path}", file=sys.stderr)
        return 1

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rr.init("inspect_robots_replay")
        rr.save(str(out_path))
    except Exception as exc:
        print(f"Could not start Rerun recording {out_path}: {exc}", file=sys.stderr)
        return 1

    counts = Counts()
    warnings: set[str] = set()
    last_steps: dict[str, int] = {}
    eval_info = _mapping(raw_log.get("eval"))
    stats = _mapping(raw_log.get("stats"))
    samples = _sequence(raw_log.get("samples"))
    trials: list[tuple[dict[str, Any], int, str]] = []

    if not samples:
        _note("the log has no samples; only any available sidecar frames will be replayed")
    for sample_index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, dict):
            _note(f"sample {sample_index} is not an object; skipped")
            continue
        sample = raw_sample
        scene_value = sample.get("scene_id")
        scene_id = str(scene_value) if scene_value is not None else f"sample-{sample_index}"
        transcripts = _sequence(sample.get("policy_transcripts"))
        if not transcripts:
            _note(f"{scene_id} has no policy transcripts; replaying metadata/events only")
        for epoch in range(_epoch_count(sample)):
            prefix = f"trial/{scene_id}/e{epoch}"
            trials.append((sample, epoch, prefix))
            _log_run_info(
                log_path,
                eval_info,
                stats,
                sample,
                epoch,
                prefix,
                warnings,
            )
            transcript = transcripts[epoch] if epoch < len(transcripts) else None
            if transcript is not None:
                try:
                    _replay_transcript(
                        transcript,
                        prefix,
                        counts,
                        last_steps,
                        warnings,
                    )
                except Exception as exc:
                    _note(f"could not replay transcript for {prefix} ({exc}); continuing")

    frames_dir = _resolve_frames_dir(log_path, stats, args.frames_root)
    _replay_frames(
        frames_dir,
        args.jpeg_quality,
        counts,
        last_steps,
        warnings,
    )

    total_steps = _safe_total_steps(stats)
    for sample, epoch, prefix in trials:
        terminal_step = max(last_steps.get(prefix, 0), total_steps)
        try:
            _set_step(terminal_step)
            rr.log(
                f"{prefix}/event/terminated",
                rr.TextLog(_termination_text(sample, epoch)),
            )
        except Exception as exc:
            _note(f"could not log terminal event for {prefix} ({exc}); skipped")

    try:
        _finish_recording()
    except Exception as exc:
        _note(f"could not explicitly finalize the Rerun recording ({exc})")

    try:
        size_bytes = out_path.stat().st_size
    except OSError as exc:
        print(f"Rerun output was not created at {out_path}: {exc}", file=sys.stderr)
        return 1

    print("Replay summary:")
    print(f"  Frames: {counts.frames}")
    print(f"  State scalar points: {counts.state_points}")
    print(f"  Action points: {counts.action_points}")
    print(f"  Transcript rows: {counts.transcript_rows}")
    print(f"  Output: {out_path}")
    print(f"  Size: {size_bytes} bytes ({size_bytes / 1024:.2f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
