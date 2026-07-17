"""Render evaluation logs as dependency-free, self-contained HTML documents."""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from inspect_robots._pngenc import png_data_url
from inspect_robots.frames import _safe
from inspect_robots.log import EvalLog, SceneResult

_STATUS_DISPLAY = {"success": "completed"}
_JSON_STRING_LIMIT = 2048
_FRAME_LABEL_RE = re.compile(r"camera '(?P<name>.*)' \(step (?P<step>\d+)\):")
_FRAME_PLACEHOLDER = "[image omitted: streamed camera frame]"
_FRAME_MAX_SIDE = 448


@dataclass
class _FrameBudget:
    """Shared mutable accounting for frame payloads in one document."""

    limit: int
    embedded: int = 0
    payload_bytes: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class _FrameContext:
    """The filesystem correlation state for one trial transcript."""

    frames_dir: Path
    trial_prefix: str
    budget: _FrameBudget


_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #20242b;
  --muted: #68707d;
  --line: #dfe3e8;
  --green: #19723b;
  --green-bg: #e9f6ed;
  --red: #a12a2a;
  --red-bg: #fbecec;
  --grey: #626a75;
  --grey-bg: #eef0f2;
  --neutral: #45546a;
  --neutral-bg: #edf1f6;
  --user: #3178c6;
  --assistant: #7a55b5;
  --tool: #36866a;
  --system: #7a828d;
  --amber: #8a5700;
  --amber-line: #d69b2d;
  --amber-bg: #fff5d9;
}
@media (prefers-color-scheme: light) {
  :root { color-scheme: light; }
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111419;
    --panel: #191d24;
    --text: #e7eaf0;
    --muted: #a4acb8;
    --line: #343a45;
    --green: #7ed99a;
    --green-bg: #193b27;
    --red: #ff9b9b;
    --red-bg: #492323;
    --grey: #c0c5cd;
    --grey-bg: #343943;
    --neutral: #b9c9df;
    --neutral-bg: #293342;
    --user: #73b7ff;
    --assistant: #bd9bed;
    --tool: #79c9ab;
    --system: #adb4be;
    --amber: #ffd484;
    --amber-line: #b77a16;
    --amber-bg: #3b2d12;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1120px, calc(100% - 32px)); margin: 0 auto 64px; }
header { border-bottom: 1px solid var(--line); background: var(--panel); }
.header-inner { width: min(1120px, calc(100% - 32px)); margin: auto; padding: 28px 0 22px; }
.header-top { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
h1 { margin: 0; font-size: 24px; font-weight: 650; min-width: 0; overflow-wrap: anywhere; }
h2 { margin: 0; font-size: 18px; font-weight: 650; min-width: 0; overflow-wrap: anywhere; }
h3 { margin: 24px 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }
.meta { color: var(--muted); margin-top: 8px; display: flex; gap: 16px; flex-wrap: wrap; }
.badge, .chip { display: inline-block; border-radius: 999px; padding: 2px 9px; font-size: 12px; }
.status-completed { color: var(--green); background: var(--green-bg); }
.status-error { color: var(--red); background: var(--red-bg); }
.status-cancelled { color: var(--grey); background: var(--grey-bg); }
.status-neutral { color: var(--neutral); background: var(--neutral-bg); }
.chip { color: var(--muted); background: var(--grey-bg); }
.spec-strip {
  margin: 24px 0; padding: 18px 20px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px;
}
dl {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 14px 24px; margin: 0;
}
dt { color: var(--muted); font-size: 12px; }
dd { margin: 2px 0 0; overflow-wrap: anywhere; }
.metrics {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  gap: 12px; margin-bottom: 28px;
}
.stat {
  padding: 14px 16px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px;
}
.stat-name { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.stat-value { margin-top: 3px; font-size: 19px; font-weight: 620; }
.scene {
  margin: 18px 0; padding: 22px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 9px;
}
.scene-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.instruction, .error { margin: 12px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.error { color: var(--red); }
.score-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 8px 0 0; }
.score-chip {
  border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; font-size: 12px;
  overflow-wrap: anywhere;
}
details { border-top: 1px solid var(--line); margin-top: 18px; padding-top: 12px; }
summary { cursor: pointer; color: var(--muted); font-weight: 600; }
.conversation { margin-top: 14px; }
.message { margin: 13px 0; padding: 2px 0 2px 13px; border-left: 3px solid var(--system); }
.message.user { border-color: var(--user); }
.message.assistant { border-color: var(--assistant); }
.message.tool { border-color: var(--tool); margin-left: 20px; }
.role { color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; }
.content { margin-top: 3px; white-space: pre-wrap; overflow-wrap: anywhere; }
img.frame {
  display: block; max-width: 100%; height: auto; margin: 6px 0;
  border: 1px solid var(--line); border-radius: 6px;
}
.system-message {
  margin: 13px 0; padding: 0 0 0 13px; border: 0;
  border-left: 3px solid var(--system);
}
.system-message summary { color: var(--muted); }
.call {
  margin-top: 8px; overflow-wrap: anywhere;
  font: 13px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
}
.agent-note {
  margin: 10px 0 6px; padding: 9px 11px; color: var(--amber);
  background: var(--amber-bg); border-left: 3px solid var(--amber-line);
}
.note-label {
  display: block; font-size: 10px; font-weight: 750;
  letter-spacing: .09em; text-transform: uppercase;
}
pre {
  padding: 14px; overflow: auto; background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px;
  font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
}
.none { color: var(--muted); margin: 28px 0; }
""".strip()


def _display_status(status: str) -> str:
    """Return the stable human-facing form of a persisted status value."""
    return _STATUS_DISPLAY.get(status, status)


def _chat_content(content: object) -> str | None:
    """Render text from an OpenAI-style content value, collapsing media parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        else:
            parts.append("[image]")
    return "\n".join(parts)


def _is_chat_transcript(transcript: object) -> bool:
    """Recognize a non-empty list of role-bearing message dictionaries."""
    return (
        isinstance(transcript, list)
        and bool(transcript)
        and all(isinstance(message, dict) and "role" in message for message in transcript)
    )


def _escape(value: object) -> str:
    """Escape one foreign value at its HTML interpolation boundary."""
    return html.escape(str(value), quote=True)


def _number(value: int | float) -> str:
    """Format numeric log values compactly before their interpolation boundary."""
    return f"{value:.4g}"


def _value(value: object) -> str:
    """Format a scalar or structured spec value without HTML escaping it."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _definition(label: object, value: object) -> str:
    """Render one escaped definition pair."""
    return f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"


def _status_class(status: str) -> str:
    """Map a persisted status to one of the fixed badge color classes."""
    displayed = _display_status(status)
    if displayed == "completed":
        return "status-completed"
    if status == "error":
        return "status-error"
    if status == "cancelled":
        return "status-cancelled"
    return "status-neutral"


def _status_badge(status: str) -> str:
    """Render a status with a fixed class and escaped display label."""
    return f'<span class="badge {_status_class(status)}">{_escape(_display_status(status))}</span>'


def _agent_notes(name: str, arguments: object) -> list[str]:
    """Extract non-empty agent notes from supported tool argument shapes."""
    parsed: object = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, dict):
        return []

    keys = ["note"]
    if name == "done":
        keys.append("summary")
    elif name == "give_up":
        keys.append("reason")
    notes: list[str] = []
    for key in keys:
        note = parsed.get(key)
        if isinstance(note, str) and note.strip():
            notes.append(note)
    return notes


def _render_tool_call(raw_call: object) -> str:
    """Render one tolerant OpenAI-style tool call, or nothing when malformed."""
    if not isinstance(raw_call, dict):
        return ""
    function = raw_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name", "unknown"))
    arguments = function.get("arguments", "")
    shown_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    notes = "".join(
        f'<div class="agent-note"><span class="note-label">agent note</span>{_escape(note)}</div>'
        for note in _agent_notes(name, arguments)
    )
    return f'{notes}<div class="call">{_escape(name)}({_escape(shown_arguments)})</div>'


def _load_frame(frame_ctx: _FrameContext, name: str, step: int) -> npt.NDArray[np.uint8] | None:
    """Load one exact-match stored frame, degrading every invalid artifact to ``None``."""
    if frame_ctx.budget.truncated:
        return None
    path = frame_ctx.frames_dir / f"{frame_ctx.trial_prefix}_{_safe(name)}_{step:06d}.npy"
    if not path.exists():
        return None
    try:
        array = cast("npt.NDArray[Any]", np.load(path, allow_pickle=False))
        if array.dtype != np.uint8 or array.size == 0:
            return None
        if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {1, 3, 4})):
            return None
    except Exception:
        return None
    longest = max(array.shape[0], array.shape[1])
    if longest > _FRAME_MAX_SIDE:
        stride = math.ceil(longest / _FRAME_MAX_SIDE)
        array = array[::stride, ::stride]
    return cast("npt.NDArray[np.uint8]", array)


def _frame_image(frame_ctx: _FrameContext, name: str, step: int) -> str | None:
    """Render one correlated frame if it is valid and fits the shared budget."""
    array = _load_frame(frame_ctx, name, step)
    if array is None:
        return None
    source = png_data_url(array)
    payload_size = len(source.partition(",")[2])
    budget = frame_ctx.budget
    if budget.limit and budget.payload_bytes + payload_size > budget.limit:
        budget.truncated = True
        return None
    budget.payload_bytes += payload_size
    budget.embedded += 1
    return (
        f'<img class="frame" loading="lazy" alt="camera {_escape(name)} step {step}" '
        f'src="{source}">'
    )


def _render_frame_parts(parts: list[object], frame_ctx: _FrameContext) -> str:
    """Render user content parts as text runs split only by successful frame embeds."""
    runs: list[str] = []
    buffered: list[str] = []
    pending: tuple[str, int] | None = None
    embedded = 0
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            raw_text = part.get("text", "")
            part_text = str(raw_text)
            if isinstance(raw_text, str):
                label = _FRAME_LABEL_RE.fullmatch(raw_text)
                if label is not None:
                    pending = (label.group("name"), int(label.group("step")))
                elif raw_text == _FRAME_PLACEHOLDER and pending is not None:
                    name, step = pending
                    pending = None
                    image = _frame_image(frame_ctx, name, step)
                    if image is not None:
                        text = _escape("\n".join(buffered))
                        runs.append(f'<div class="content">{text}</div>')
                        runs.append(image)
                        buffered = []
                        embedded += 1
                        continue
            buffered.append(part_text)
        else:
            buffered.append("[image]")
    if embedded == 0 or buffered:
        text = _escape("\n".join(buffered))
        runs.append(f'<div class="content">{text}</div>')
    return "".join(runs)


def _render_message(raw_message: object, frame_ctx: _FrameContext | None = None) -> str:
    """Render one tolerant chat message without trusting its role or content."""
    if not isinstance(raw_message, dict):
        return ""
    role = str(raw_message["role"])
    content = _chat_content(raw_message.get("content"))
    if role == "system":
        body = "" if content is None else f'<div class="content">{_escape(content)}</div>'
        return f'<details class="system-message"><summary>system</summary>{body}</details>'

    role_class = role if role in {"user", "assistant", "tool"} else "unknown"
    if frame_ctx is not None and role == "user" and isinstance(raw_message.get("content"), list):
        body = _render_frame_parts(cast(list[object], raw_message["content"]), frame_ctx)
    else:
        body = "" if content is None else f'<div class="content">{_escape(content)}</div>'
    if role == "tool":
        return (
            f'<div class="message {role_class}"><div class="role">{_escape(role)}</div>{body}</div>'
        )
    calls = ""
    tool_calls = raw_message.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = "".join(_render_tool_call(raw_call) for raw_call in tool_calls)
    return (
        f'<div class="message {role_class}"><div class="role">{_escape(role)}</div>'
        f"{body}{calls}</div>"
    )


def _render_chat_transcript(
    transcript: list[object], frame_ctx: _FrameContext | None = None
) -> str:
    """Render a defensive role-oriented conversation."""
    return (
        '<div class="conversation">'
        + "".join(_render_message(message, frame_ctx) for message in transcript)
        + "</div>"
    )


def _elide_json_values(value: Any) -> Any:
    """Recursively bound long JSON string values before serialization."""
    if isinstance(value, str):
        if len(value) <= _JSON_STRING_LIMIT:
            return value
        omitted = len(value) - _JSON_STRING_LIMIT
        return value[:_JSON_STRING_LIMIT] + f"[... {omitted} chars truncated]"
    if isinstance(value, list):
        return [_elide_json_values(item) for item in value]
    if isinstance(value, tuple):
        return [_elide_json_values(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _elide_json_values(item) for key, item in value.items()}
    return value


def _render_transcript(transcript: object, frame_ctx: _FrameContext | None = None) -> str:
    """Render chat-shaped records conversationally and all others as bounded JSON."""
    if _is_chat_transcript(transcript):
        return _render_chat_transcript(cast(list[object], transcript), frame_ctx)
    # Escaping happens on the dumped text below, so raw non-ASCII is safe and
    # far more readable than \uXXXX escapes.
    dumped = json.dumps(
        _elide_json_values(transcript), indent=2, sort_keys=True, ensure_ascii=False
    )
    return f"<pre>{_escape(dumped)}</pre>"


def _score_chips(values: Mapping[str, float], *, prefix: str = "") -> str:
    """Render sorted score values as compact escaped chips."""
    return "".join(
        f'<span class="score-chip">{_escape(prefix + name)}={_escape(_number(value))}</span>'
        for name, value in sorted(values.items())
    )


def _trial_frame_context(
    frame_ctx: _FrameContext | None, scene_id: str, trial: int
) -> _FrameContext | None:
    """Specialize a document frame context to one scene trial."""
    if frame_ctx is None:
        return None
    return _FrameContext(
        frame_ctx.frames_dir,
        _safe(f"{scene_id}-e{trial}"),
        frame_ctx.budget,
    )


def _render_trial_transcript(
    transcript: object,
    frame_ctx: _FrameContext | None,
    scene_id: str,
    trial: int,
) -> str:
    """Render one transcript with its enumerate-index frame correlation context."""
    return _render_transcript(transcript, _trial_frame_context(frame_ctx, scene_id, trial))


def _scene_section(
    scene: SceneResult, *, open_transcript: bool, frame_ctx: _FrameContext | None = None
) -> str:
    """Render one complete scene card and its available trial transcripts."""
    instruction = (
        ""
        if scene.instruction is None
        else f'<p class="instruction">{_escape(scene.instruction)}</p>'
    )
    error = "" if scene.error is None else f'<p class="error">{_escape(scene.error)}</p>'

    reduced = _score_chips(scene.reduced)
    reduced_block = (
        "" if not reduced else f'<h3>Reduced scores</h3><div class="score-row">{reduced}</div>'
    )
    epoch_chips = "".join(
        _score_chips(epoch, prefix=f"trial {index} ") for index, epoch in enumerate(scene.epochs)
    )
    epoch_block = (
        ""
        if not epoch_chips
        else f'<h3>Trial scores</h3><div class="score-row">{epoch_chips}</div>'
    )
    reasons = "".join(
        '<span class="score-chip">n/a</span>'
        if reason is None
        else f'<span class="score-chip">{_escape(reason)}</span>'
        for reason in scene.termination_reasons
    )
    reasons_block = (
        "" if not reasons else f'<h3>Termination reasons</h3><div class="score-row">{reasons}</div>'
    )
    judgements = "".join(
        '<span class="score-chip">n/a</span>'
        if judgement is None
        else f'<span class="score-chip">{_escape(judgement)}</span>'
        for judgement in scene.operator_judgements
    )
    judgements_block = (
        ""
        if not judgements
        else f'<h3>Operator judgements</h3><div class="score-row">{judgements}</div>'
    )

    transcripts = "".join(
        (
            f'<details class="transcript"{" open" if open_transcript else ""}>'
            f"<summary>Trial {trial} transcript</summary>"
            f"{_render_trial_transcript(transcript, frame_ctx, scene.scene_id, trial)}"
            "</details>"
        )
        for trial, transcript in enumerate(scene.policy_transcripts)
        if transcript is not None
    )
    return (
        '<section class="scene">'
        f'<div class="scene-head"><h2>{_escape(scene.scene_id)}</h2>'
        f"{_status_badge(scene.status)}</div>{instruction}{error}{reduced_block}{epoch_block}"
        f"{reasons_block}{judgements_block}{transcripts}</section>"
    )


def render_html(
    log: EvalLog,
    *,
    title: str,
    frames_dir: Path | None = None,
    frames_budget_bytes: int = 50_000_000,
) -> str:
    """Return one self-contained HTML document describing the complete evaluation log."""
    git = (
        'git <span class="chip">unknown</span>'
        if log.eval.git_commit is None
        else f"git {_escape(log.eval.git_commit)}"
    )
    definitions = [
        _definition("policy", log.eval.policy),
        _definition("embodiment", log.eval.embodiment),
    ]
    definitions.extend(
        _definition(key, _value(value)) for key, value in sorted(log.eval.policy_config.items())
    )
    if log.eval.seed is not None:
        definitions.append(_definition("seed", log.eval.seed))
    if log.eval.max_steps is not None:
        definitions.append(_definition("max steps", log.eval.max_steps))
    if log.stats.mean_inference_latency_s is not None:
        definitions.append(
            _definition(
                "mean inference latency", f"{_number(log.stats.mean_inference_latency_s)} s"
            )
        )
    definitions.extend(
        [
            _definition("duration", f"{_number(log.stats.duration_s)} s"),
            _definition("total steps", log.stats.total_steps),
        ]
    )

    metric_tiles = "".join(
        '<div class="stat">'
        f'<div class="stat-name">{_escape(name)}</div>'
        f'<div class="stat-value">{_escape(_number(value))}</div></div>'
        for name, value in sorted(log.results.metrics.items())
    )
    metric_tiles += "".join(
        '<div class="stat">'
        f'<div class="stat-name">{label}</div>'
        f'<div class="stat-value">{_escape(value)}</div></div>'
        for label, value in (
            ("scenes", log.results.total_scenes),
            ("trials", log.results.total_trials),
            ("errored", log.results.errored_trials),
        )
    )

    transcript_count = sum(
        transcript is not None for scene in log.samples for transcript in scene.policy_transcripts
    )
    budget = _FrameBudget(limit=frames_budget_bytes)
    frame_ctx = None if frames_dir is None else _FrameContext(frames_dir, "", budget)
    scenes = "".join(
        _scene_section(
            scene,
            open_transcript=transcript_count == 1,
            frame_ctx=frame_ctx,
        )
        for scene in log.samples
    )
    no_transcripts = (
        '<p class="none">no policy transcripts recorded</p>' if transcript_count == 0 else ""
    )
    frames_chip = (
        ""
        if not budget.truncated
        else '<span class="chip">frames truncated at '
        f"{frames_budget_bytes / 1_000_000:g} MB ({budget.embedded} embedded)</span>"
    )
    meta_tail = (
        f"<span>inspect-robots {_escape(log.eval.inspect_robots_version)}</span>"
        f"<span>{git}</span>{frames_chip}"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_STYLES}</style>
</head>
<body>
<header><div class="header-inner">
  <div class="header-top"><h1>{_escape(log.eval.task)}</h1>{_status_badge(log.status)}</div>
  <div class="meta"><span>{_escape(log.eval.created)}</span>
  {meta_tail}</div>
</div></header>
<main>
  <section class="spec-strip"><dl>{"".join(definitions)}</dl></section>
  <section class="metrics">{metric_tiles}</section>
  {scenes}
  {no_transcripts}
</main>
</body>
</html>
"""
    return document
