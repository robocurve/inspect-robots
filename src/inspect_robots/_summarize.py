"""Build deterministic or model-written learnings from a saved evaluation log."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from inspect_robots._html import _agent_notes, _chat_content, _display_status
from inspect_robots._pointers import resolve_log_pointer
from inspect_robots.cli import _OUTCOME_PHRASES
from inspect_robots.errors import ConfigError
from inspect_robots.log import EvalLog, SceneResult, read_eval_log

HttpPost = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]

_TRANSCRIPT_CHAR_BUDGET = 24_000
_RESPONSE_EXCERPT_LIMIT = 500
_TRUNCATION_MARKER = "[... beginning omitted; showing transcript tail ...]\n"

_SYSTEM_PROMPT = """Distill this robot evaluation into concise markdown.
Return exactly these three sections, in this order, with no other headings:
## What happened
## Failure modes
## Lessons for next attempt

Ground every claim in the supplied digest and transcripts. In the final section,
write direct imperatives to a future agent attempting the same task."""


@dataclass(frozen=True)
class TrialTranscript:
    """Associate one scene epoch with its transcript payload and selected source."""

    scene_id: str
    epoch: int
    transcript: object | None
    source: str


def _is_dropped(transcript: object) -> bool:
    # rollout._collect_transcript leaves either marker dict in place of the
    # conversation: transcript_dropped (size) or transcript_error (hook raised).
    return isinstance(transcript, dict) and (
        transcript.get("transcript_dropped") is True or "transcript_error" in transcript
    )


def _read_sidecar(scene: SceneResult, index: int, log_path: Path) -> list[object] | None:
    if index >= len(scene.trial_metadata):
        return None
    pointer = scene.trial_metadata[index].get("transcript")
    target = resolve_log_pointer(log_path, pointer)
    if target is None:
        return None
    try:
        with target.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def load_transcripts(log: EvalLog, log_path: Path) -> list[TrialTranscript]:
    """Load one transcript per recorded trial, preferring inline data over sidecars."""
    transcripts: list[TrialTranscript] = []
    for scene in log.samples:
        for index in range(len(scene.epochs)):
            inline = (
                scene.policy_transcripts[index] if index < len(scene.policy_transcripts) else None
            )
            if inline is not None and not _is_dropped(inline):
                transcripts.append(TrialTranscript(scene.scene_id, index, inline, "inline"))
                continue
            sidecar = _read_sidecar(scene, index, log_path)
            if sidecar is not None:
                transcripts.append(TrialTranscript(scene.scene_id, index, sidecar, "sidecar"))
            else:
                transcripts.append(TrialTranscript(scene.scene_id, index, None, "none"))
    return transcripts


def _one_line(value: object) -> str:
    return " ".join(str(value).split())


def _outcome(reason: object) -> str:
    text = "" if reason is None else str(reason)
    if not text:
        return "no reason recorded"
    return _OUTCOME_PHRASES.get(text, text)


def _parallel_value(values: tuple[Any, ...], index: int) -> Any:
    return values[index] if index < len(values) else None


def _step_count(scores: dict[str, float]) -> str:
    value = scores.get("episode_length")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    return "not recorded"


def _messages(transcript: object | None) -> list[object]:
    return transcript if isinstance(transcript, list) else []


def _tool_call_count(messages: list[object]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            count += len(calls)
    return count


def _last_assistant_note(messages: list[object]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in reversed(calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                notes = _agent_notes(
                    str(function.get("name", "unknown")), function.get("arguments", "")
                )
                if notes:
                    return _one_line(notes[-1])
        content = _chat_content(message.get("content"))
        if content is not None and content.strip():
            return _one_line(content)
    return None


def _plural(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def build_digest(log: EvalLog, transcripts: list[TrialTranscript]) -> str:
    """Render a stable markdown account of run identity, trial results, and transcript stats."""
    lines = [
        "# Evaluation digest",
        "",
        "## Run",
        f"- Task: {_one_line(log.eval.task)}",
        f"- Policy: {_one_line(log.eval.policy)}",
        f"- Embodiment: {_one_line(log.eval.embodiment)}",
        f"- Status: {_display_status(log.status)}",
    ]
    model = log.eval.policy_config.get("model")
    if model is not None:
        lines.append(f"- Model: {_one_line(model)}")

    lines.extend(["", "## Trials"])
    transcript_by_key = {
        (transcript.scene_id, transcript.epoch): transcript for transcript in transcripts
    }
    for scene in log.samples:
        for epoch, scores in enumerate(scene.epochs):
            reason = _parallel_value(scene.termination_reasons, epoch)
            fields = [
                f"outcome: {_outcome(reason)}",
                f"steps: {_step_count(scores)}",
                f"termination: {_one_line(reason) if reason is not None else 'not recorded'}",
            ]
            judgement = _parallel_value(scene.operator_judgements, epoch)
            if judgement is not None:
                fields.append(f"operator judgement: {_one_line(judgement)}")
            note = _parallel_value(scene.operator_notes, epoch)
            if note is not None:
                fields.append(f"operator notes: {_one_line(note)}")
            if not scores and scene.status == "error" and scene.error:
                fields.append(f"error: {_one_line(scene.error)}")
            lines.append(f"- `{scene.scene_id}` epoch {epoch}: {'; '.join(fields)}")

    lines.extend(["", "## Transcript stats"])
    for scene in log.samples:
        for epoch in range(len(scene.epochs)):
            transcript = transcript_by_key[(scene.scene_id, epoch)]
            if transcript.source == "none":
                lines.append(
                    f"- `{scene.scene_id}` epoch {epoch}: no transcript recorded; "
                    "0 messages; 0 tool calls; last assistant note: none"
                )
                continue
            messages = _messages(transcript.transcript)
            message_count = len(messages)
            tool_count = _tool_call_count(messages)
            last_note = _last_assistant_note(messages)
            note_text = "none" if last_note is None else last_note
            lines.append(
                f"- `{scene.scene_id}` epoch {epoch}: "
                f"{_plural(message_count, 'message')}; "
                f"{_plural(tool_count, 'tool call')}; "
                f"last assistant note: {note_text}"
            )
    return "\n".join(lines) + "\n"


def _bounded_transcript(transcript: TrialTranscript) -> str:
    if transcript.transcript is None:
        return "(no transcript recorded)"
    rendered = json.dumps(
        transcript.transcript,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    if len(rendered) <= _TRANSCRIPT_CHAR_BUDGET:
        return rendered
    tail_size = _TRANSCRIPT_CHAR_BUDGET - len(_TRUNCATION_MARKER)
    return _TRUNCATION_MARKER + rendered[-tail_size:]


def build_messages(digest: str, transcripts: list[TrialTranscript]) -> list[dict[str, Any]]:
    """Create a grounded chat request with a fixed response contract and bounded trial tails."""
    transcript_blocks = [
        f"### Transcript: {transcript.scene_id}, epoch {transcript.epoch}\n"
        f"{_bounded_transcript(transcript)}"
        for transcript in transcripts
    ]
    user_content = digest + "\n# Recorded transcripts\n\n" + "\n\n".join(transcript_blocks)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _response_excerpt(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return (text or "(empty response body)")[:_RESPONSE_EXCERPT_LIMIT]


def _urllib_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
    """Send one blocking HTTP POST and preserve HTTP error bodies for guided failures."""
    request = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise ConfigError(
            f"summary request failed: {exc.reason}.\n"
            "fix: check --base-url and network connectivity, then retry"
        ) from exc


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    http_post: HttpPost | None = None,
) -> str:
    """Return one OpenAI-compatible chat completion or raise a guided configuration error."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 8192}).encode("utf-8")
    post = _urllib_post if http_post is None else http_post
    status, response_body = post(url, headers, body)
    if not 200 <= status < 300:
        raise ConfigError(
            f"summary request failed with HTTP {status}: {_response_excerpt(response_body)}\n"
            "fix: check --base-url, --model, and the configured API key"
        )

    try:
        payload = cast(dict[str, Any], json.loads(response_body))
        choices = cast(list[Any], payload["choices"])
        message = cast(dict[str, Any], choices[0]["message"])
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
        raise ConfigError(
            f"summary endpoint returned a malformed reply: {_response_excerpt(response_body)}\n"
            "fix: use an OpenAI-compatible /chat/completions endpoint"
        ) from None
    return content


def summarize(
    log_path: Path,
    *,
    model: str | None,
    base_url: str,
    api_key_env: str,
    http_post: HttpPost | None = None,
) -> str:
    """Produce an offline digest or a model-written markdown summary for one saved log."""
    log = read_eval_log(str(log_path))
    if not log.samples:
        raise ConfigError(
            "the eval log has no samples to summarize.\n"
            "fix: pass a completed log containing at least one recorded trial"
        )
    transcripts = load_transcripts(log, log_path)
    digest = build_digest(log, transcripts)
    if model is None:
        return digest
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ConfigError(
            f"no API key found in ${api_key_env}.\n"
            f"fix: set ${api_key_env}, or choose the correct variable with --api-key-env"
        )
    return chat_completion(
        base_url,
        api_key,
        model,
        build_messages(digest, transcripts),
        http_post=http_post,
    )
