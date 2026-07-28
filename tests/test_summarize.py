"""Transcript summarization stays deterministic, bounded, and offline by default."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from email.message import Message
from pathlib import Path

import pytest

import inspect_robots._summarize as summarize_module
from inspect_robots._summarize import (
    _TRANSCRIPT_CHAR_BUDGET,
    TrialTranscript,
    _urllib_post,
    build_digest,
    build_messages,
    chat_completion,
    load_transcripts,
    summarize,
)
from inspect_robots.cli import build_parser, main
from inspect_robots.errors import ConfigError
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult


def _eval_log() -> EvalLog:
    return EvalLog(
        version=1,
        status="error",
        eval=EvalSpec(
            task="stack-blocks",
            policy="agent",
            embodiment="arm",
            created="2026-07-27T00:00:00Z",
            inspect_robots_version="0.8.0",
            policy_config={"model": "test-model", "temperature": 0.1},
            max_steps=20,
        ),
        results=EvalResults(
            total_scenes=2,
            total_trials=4,
            metrics={"success_at_end": 0.25},
            errored_trials=1,
        ),
        stats=EvalStats(
            started_at="start",
            completed_at="end",
            duration_s=3.0,
            total_steps=28,
        ),
        samples=(
            SceneResult(
                scene_id="scene-a",
                status="error",
                epochs=({"episode_length": 3.0, "success_at_end": 1.0}, {}),
                error="PolicyError: gripper stalled",
                operator_judgements=("yes", None),
                operator_notes=("cube was stable\nat the end", None),
                trial_metadata=(
                    {},
                    {"transcript": "transcripts/run/scene-a-e1.jsonl"},
                ),
                termination_reasons=("success", None),
                policy_transcripts=(
                    [
                        {"role": "user", "content": "stack the blocks"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "move",
                                        "arguments": json.dumps(
                                            {"note": "align over the red block"}
                                        ),
                                    }
                                }
                            ],
                        },
                    ],
                    {"transcript_dropped": True, "bytes": 2_100_000},
                ),
            ),
            SceneResult(
                scene_id="scene-b",
                status="success",
                epochs=({"success_at_end": 0.0}, {"episode_length": 5.0}),
                operator_judgements=(),
                operator_notes=(),
                trial_metadata=({},),
                termination_reasons=("max_steps", "collision"),
                policy_transcripts=(
                    None,
                    [{"role": "assistant", "content": ""}],
                ),
            ),
        ),
        error="one trial failed",
    )


def _write_log(tmp_path: Path, log: EvalLog | None = None) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps((log or _eval_log()).to_dict()), encoding="utf-8")
    return path


@pytest.fixture
def log_path(tmp_path: Path) -> Iterator[Path]:
    sidecar = tmp_path / "transcripts" / "run" / "scene-a-e1.jsonl"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps({"role": "assistant", "content": "could not recover"}) + "\n",
        encoding="utf-8",
    )
    yield _write_log(tmp_path)


def test_load_transcripts_prefers_inline_then_falls_back_to_sidecar(log_path: Path) -> None:
    log = _eval_log()
    transcripts = load_transcripts(log, log_path)

    assert [(item.scene_id, item.epoch, item.source) for item in transcripts] == [
        ("scene-a", 0, "inline"),
        ("scene-a", 1, "sidecar"),
        ("scene-b", 0, "none"),
        ("scene-b", 1, "inline"),
    ]
    assert transcripts[1].transcript == [{"role": "assistant", "content": "could not recover"}]
    assert transcripts[2].transcript is None


def test_missing_or_malformed_sidecar_degrades_to_no_transcript(tmp_path: Path) -> None:
    log = _eval_log()
    scene = replace(
        log.samples[1],
        epochs=({}, {}, {}),
        trial_metadata=(
            {"transcript": "missing.jsonl"},
            {"transcript": "broken.jsonl"},
        ),
        policy_transcripts=(None, None, None),
    )
    (tmp_path / "broken.jsonl").write_text("{not json}\n", encoding="utf-8")

    transcripts = load_transcripts(replace(log, samples=(scene,)), tmp_path / "run.json")

    assert [item.source for item in transcripts] == ["none", "none", "none"]


def test_sidecar_pointer_cannot_escape_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    secret = tmp_path / "secret.jsonl"
    secret.write_text(json.dumps({"role": "assistant", "content": "secret"}) + "\n")
    log = _eval_log()
    scene = replace(
        log.samples[1],
        trial_metadata=(
            {"transcript": str(secret)},
            {"transcript": "../secret.jsonl"},
        ),
        policy_transcripts=(None, None),
    )

    transcripts = load_transcripts(replace(log, samples=(scene,)), log_dir / "run.json")

    assert [item.source for item in transcripts] == ["none", "none"]


def test_transcript_error_marker_falls_back_like_dropped(log_path: Path) -> None:
    log = _eval_log()
    scene = replace(
        log.samples[0],
        policy_transcripts=(
            {"transcript_error": "policy transcript hook raised"},
            log.samples[0].policy_transcripts[1],
        ),
    )

    transcripts = load_transcripts(replace(log, samples=(scene,)), log_path)

    assert transcripts[0].source == "none"
    assert transcripts[1].source == "sidecar"


def test_digest_is_stable_and_covers_each_trial(log_path: Path) -> None:
    log = _eval_log()
    digest = build_digest(log, load_transcripts(log, log_path))

    assert digest.splitlines() == [
        "# Evaluation digest",
        "",
        "## Run",
        "- Task: stack-blocks",
        "- Policy: agent",
        "- Embodiment: arm",
        "- Status: error",
        "- Model: test-model",
        "",
        "## Trials",
        (
            "- `scene-a` epoch 0: outcome: succeeded; steps: 3; termination: success; "
            "operator judgement: yes; operator notes: cube was stable at the end"
        ),
        (
            "- `scene-a` epoch 1: outcome: no reason recorded; steps: not recorded; "
            "termination: not recorded; error: PolicyError: gripper stalled"
        ),
        (
            "- `scene-b` epoch 0: outcome: hit step limit; steps: not recorded; "
            "termination: max_steps"
        ),
        "- `scene-b` epoch 1: outcome: collision; steps: 5; termination: collision",
        "",
        "## Transcript stats",
        (
            "- `scene-a` epoch 0: 2 messages; 1 tool call; "
            "last assistant note: align over the red block"
        ),
        ("- `scene-a` epoch 1: 1 message; 0 tool calls; last assistant note: could not recover"),
        (
            "- `scene-b` epoch 0: no transcript recorded; "
            "0 messages; 0 tool calls; last assistant note: none"
        ),
        "- `scene-b` epoch 1: 1 message; 0 tool calls; last assistant note: none",
    ]


def test_digest_handles_old_parallel_fields_and_defensive_transcript_values() -> None:
    log = _eval_log()
    scene = replace(
        log.samples[0],
        status="success",
        epochs=({},),
        error=None,
        operator_judgements=(),
        operator_notes=(),
        trial_metadata=({},),
        termination_reasons=(),
        policy_transcripts=(
            [
                42,
                {"role": "user", "content": "goal", "tool_calls": "invalid"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        "invalid",
                        {"function": "invalid"},
                        {
                            "function": {
                                "name": "move",
                                "arguments": "not json",
                            }
                        },
                    ],
                },
                {"role": "assistant", "content": None, "tool_calls": "invalid"},
            ],
        ),
    )
    old_log = replace(
        log,
        status="success",
        eval=replace(log.eval, policy_config={}),
        samples=(scene,),
    )
    transcript = TrialTranscript("scene-a", 0, scene.policy_transcripts[0], "inline")

    digest = build_digest(old_log, [transcript])

    assert "- Status: completed" in digest
    assert "- Model:" not in digest
    assert "outcome: no reason recorded" in digest
    assert "4 messages; 3 tool calls; last assistant note: none" in digest


def test_build_messages_fixes_headings_and_keeps_truncated_tail() -> None:
    long_content = "HEAD" + ("x" * _TRANSCRIPT_CHAR_BUDGET) + "TAIL"
    transcripts = [
        TrialTranscript(
            "large",
            0,
            [{"role": "assistant", "content": long_content}],
            "inline",
        ),
        TrialTranscript("empty", 1, None, "none"),
    ]

    messages = build_messages("digest", transcripts)

    assert [message["role"] for message in messages] == ["system", "user"]
    system = str(messages[0]["content"])
    assert "## What happened" in system
    assert "## Failure modes" in system
    assert "## Lessons for next attempt" in system
    user = str(messages[1]["content"])
    large_block = user.split("### Transcript: large, epoch 0\n", 1)[1].split(
        "\n\n### Transcript: empty", 1
    )[0]
    assert len(large_block) == _TRANSCRIPT_CHAR_BUDGET
    assert "beginning omitted" in large_block
    assert "HEAD" not in large_block
    assert "TAIL" in large_block
    assert "(no transcript recorded)" in user


def test_build_messages_preserves_short_transcript() -> None:
    messages = build_messages(
        "digest",
        [TrialTranscript("short", 0, [{"role": "user", "content": "hello"}], "inline")],
    )

    assert '"content": "hello"' in str(messages[1]["content"])
    assert "beginning omitted" not in str(messages[1]["content"])


def test_llm_mode_sends_expected_request_and_returns_reply_verbatim(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("SUMMARY_KEY", "secret")

    def fake_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        captured.update(url=url, headers=headers, body=json.loads(body_bytes))
        return (
            200,
            json.dumps(
                {"choices": [{"message": {"content": "## What happened\n\nCanned reply"}}]}
            ).encode(),
        )

    document = summarize(
        log_path,
        model="summary-model",
        base_url="https://example.test/v1/",
        api_key_env="SUMMARY_KEY",
        http_post=fake_post,
    )

    assert document == "## What happened\n\nCanned reply"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "summary-model"
    assert body["max_tokens"] == 8192
    assert isinstance(body["messages"], list)


def test_digest_mode_needs_no_key_or_network(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fail_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        raise AssertionError("digest mode must not make a request")

    digest = summarize(
        log_path,
        model=None,
        base_url="https://example.test",
        api_key_env="ANTHROPIC_API_KEY",
        http_post=fail_post,
    )

    assert digest.startswith("# Evaluation digest\n")


def test_chat_completion_uses_default_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        return 200, b'{"choices":[{"message":{"content":"default transport"}}]}'

    monkeypatch.setattr(summarize_module, "_urllib_post", fake_post)

    assert (
        chat_completion("https://example.test", "key", "model", [], http_post=None)
        == "default transport"
    )


@pytest.mark.parametrize("status", [100, 500])
def test_chat_completion_rejects_non_2xx_with_body_excerpt(status: int) -> None:
    def fake_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        return status, b"provider denied the request"

    with pytest.raises(ConfigError, match=rf"HTTP {status}.*provider denied.*\nfix:"):
        chat_completion("https://example.test", "key", "model", [], http_post=fake_post)


@pytest.mark.parametrize(
    "reply",
    [
        b"",
        b'{"choices":[{"message":{"content":null}}]}',
        b"\xff\xfe not utf-8",
    ],
)
def test_chat_completion_rejects_malformed_replies(reply: bytes) -> None:
    def fake_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        return 200, reply

    with pytest.raises(ConfigError, match=r"malformed reply: .*\nfix:"):
        chat_completion("https://example.test", "key", "model", [], http_post=fake_post)


class _UrlopenResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def __enter__(self) -> _UrlopenResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_urllib_post_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _UrlopenResponse:
        captured.update(
            url=request.full_url,
            authorization=request.get_header("Authorization"),
            data=request.data,
            timeout=timeout,
        )
        return _UrlopenResponse(201, b"created")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _urllib_post("https://example.test", {"Authorization": "Bearer k"}, b"{}") == (
        201,
        b"created",
    )
    assert captured == {
        "url": "https://example.test",
        "authorization": "Bearer k",
        "data": b"{}",
        "timeout": 120.0,
    }


def test_urllib_post_returns_http_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _UrlopenResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "rate limited",
            Message(),
            io.BytesIO(b"slow down"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _urllib_post("https://example.test", {}, b"{}") == (429, b"slow down")


def test_urllib_post_turns_url_error_into_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _UrlopenResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ConfigError, match=r"offline.*\nfix:"):
        _urllib_post("https://example.test", {}, b"{}")


def test_summarize_requires_key_only_in_llm_mode(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_SUMMARY_KEY", raising=False)

    with pytest.raises(ConfigError, match=r"\$MISSING_SUMMARY_KEY.*\nfix:"):
        summarize(
            log_path,
            model="model",
            base_url="https://example.test",
            api_key_env="MISSING_SUMMARY_KEY",
        )


def test_summarize_rejects_log_without_samples(tmp_path: Path) -> None:
    log = replace(
        _eval_log(),
        results=EvalResults(total_scenes=0, total_trials=0),
        samples=(),
    )
    path = _write_log(tmp_path, log)

    with pytest.raises(ConfigError, match=r"no samples.*\nfix:"):
        summarize(
            path,
            model=None,
            base_url="https://example.test",
            api_key_env="UNUSED",
        )


def test_parser_exposes_summarize_defaults() -> None:
    args = build_parser().parse_args(["summarize", "run.json"])

    assert args.model is None
    assert args.base_url == "https://api.anthropic.com/v1"
    assert args.api_key_env == "ANTHROPIC_API_KEY"
    assert args.out is None


def test_cli_writes_default_learnings_path_atomically(
    log_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["summarize", str(log_path)]) == 0

    out_path = log_path.parent / "learnings" / "run.md"
    assert capsys.readouterr().out == f"wrote {out_path}\n"
    assert out_path.read_text(encoding="utf-8").startswith("# Evaluation digest\n")
    assert not list(out_path.parent.glob("*.tmp"))


def test_cli_llm_reply_lands_verbatim_in_explicit_output(
    log_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SUMMARY_KEY", "secret")

    def fake_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        return 200, b'{"choices":[{"message":{"content":"verbatim reply"}}]}'

    monkeypatch.setattr(summarize_module, "_urllib_post", fake_post)
    out_path = tmp_path / "nested" / "custom.md"

    assert (
        main(
            [
                "summarize",
                str(log_path),
                "--model",
                "model",
                "--api-key-env",
                "SUMMARY_KEY",
                "-o",
                str(out_path),
            ]
        )
        == 0
    )

    assert out_path.read_text(encoding="utf-8") == "verbatim reply"
    assert capsys.readouterr().out == f"wrote {out_path}\n"


def test_cli_stdout_prints_only_document(
    log_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["summarize", str(log_path), "-o", "-"]) == 0

    output = capsys.readouterr().out
    assert output.startswith("# Evaluation digest\n")
    assert "wrote " not in output
    assert not (log_path.parent / "learnings").exists()


def test_cli_refuses_output_directory(log_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"is a directory; pass a Markdown file path"):
        main(["summarize", str(log_path), "-o", str(log_path.parent)])


def test_cli_refuses_to_overwrite_input_log(log_path: Path) -> None:
    with pytest.raises(SystemExit, match="would overwrite the input log"):
        main(["summarize", str(log_path), "-o", str(log_path)])


def test_cli_renders_config_error_without_traceback(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_SUMMARY_KEY", raising=False)

    with pytest.raises(SystemExit, match=r"no API key.*\nfix:"):
        main(
            [
                "summarize",
                str(log_path),
                "--model",
                "model",
                "--api-key-env",
                "MISSING_SUMMARY_KEY",
            ]
        )
