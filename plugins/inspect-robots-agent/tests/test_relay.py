"""Exercise optional Relay capture through the real policy and evaluation loop."""

from __future__ import annotations

import builtins
import gc
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import httpx
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots_agent import LLMAgentPolicy


def _reply(name: str, arguments: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def _run(
    path: Path,
    *,
    relay: bool = True,
    fail: bool = False,
    scene: str = "scene",
    wire: str = "chat",
    cancel: bool = False,
    max_llm_calls: int = 100,
    scorer: Any = None,
) -> Any:
    if relay:
        pytest.importorskip("nemo_relay")
    replies = [
        _reply("unknown", {"secret": "private-argument"}, 0),
        _reply("move_by", {"deltas": {"dx": 0.1, "dy": 0.1}, "note": "move"}, 1),
        _reply("done", {"summary": "private-summary"}, 2),
    ]
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if cancel:
            raise KeyboardInterrupt()
        if fail:
            raise httpx.ReadError("private-provider-detail")
        reply = replies[len(calls) - 1]
        if wire == "messages":
            call = reply["choices"][0]["message"]["tool_calls"][0]
            reply = {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"]),
                    }
                ],
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                },
            }
        return httpx.Response(200, json=reply)

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://provider.test/v1",
        env={},
        transport=httpx.MockTransport(respond),
        max_retries=1,
        relay=relay,
        wire=wire,
        max_llm_calls=max_llm_calls,
    )
    task = Task(
        name="relay-demo",
        scenes=[Scene(id=scene, instruction="private-goal")],
        scorer=scorer or success_at_end(),
        max_steps=20,
    )
    (log,) = ir_eval(task, policy, CubePickEmbodiment(), seed=0, log_dir=str(path))
    return log, len(calls)


def _events(path: Path, log: Any) -> list[dict[str, Any]]:
    receipt = log.samples[0].trial_metadata[0]["relay"]
    assert receipt["status"] == "captured"
    return [json.loads(line) for line in (path / receipt["path"]).read_text().splitlines()]


def test_real_calls_are_balanced_correlated_and_payload_free(tmp_path: Path) -> None:
    log, calls = _run(tmp_path)
    rows = _events(tmp_path, log)
    starts = {r["uuid"]: r for r in rows if r.get("scope_category") == "start"}
    ends = {r["uuid"]: r for r in rows if r.get("scope_category") == "end"}
    assert starts.keys() == ends.keys()
    assert calls == 3
    assert len(starts) == 7  # one trial, three logical model calls, three dispatches
    root = next(r for r in starts.values() if r["category"] == "agent")
    assert all(r["parent_uuid"] == root["uuid"] for r in starts.values() if r is not root)
    tools = [r for r in ends.values() if r["category"] == "tool"]
    assert [r["name"] for r in tools] == ["unknown", "move_by", "done"]
    assert [r["metadata"]["status"] for r in tools] == ["rejected", "accepted", "accepted"]
    assert [r["metadata"]["model_call_index"] for r in tools] == [0, 1, 2]
    llms = [r for r in ends.values() if r["category"] == "llm"]
    # ChatClient does not currently expose usage in AssistantMessage. Unknown
    # usage must stay absent, not turn into a misleading zero-token estimate.
    assert all("usage" not in r["category_profile"]["annotated_response"] for r in llms)
    text = json.dumps(rows)
    assert "private-" not in text
    assert "data:image" not in text
    assert log.eval.policy_config["relay"] is True


def test_capture_preserves_calls_transcripts_and_scores(tmp_path: Path) -> None:
    baseline, base_calls = _run(tmp_path / "off", relay=False)
    traced, traced_calls = _run(tmp_path / "on")
    assert base_calls == traced_calls == 3
    assert baseline.results.metrics == traced.results.metrics
    assert baseline.samples[0].policy_transcripts == traced.samples[0].policy_transcripts
    assert baseline.samples[0].termination_reasons == traced.samples[0].termination_reasons
    assert "relay" not in baseline.samples[0].trial_metadata[0]
    assert not (tmp_path / "off" / "relay").exists()


def test_provider_failure_closes_spans_without_retrying(tmp_path: Path) -> None:
    log, calls = _run(tmp_path, fail=True)
    assert log.status == "error"
    assert calls == 1
    rows = _events(tmp_path, log)
    assert len(rows) == 4  # trial + failed model lifecycle
    end = next(r for r in rows if r["category"] == "llm" and r["scope_category"] == "end")
    assert end["metadata"]["status"] == "error"
    assert end["metadata"]["otel.status_code"] == "ERROR"
    assert "private-provider-detail" not in json.dumps(rows)


def test_export_setup_failure_preserves_policy_execution(tmp_path: Path) -> None:
    (tmp_path / "relay").write_text("block directory creation")
    log, calls = _run(tmp_path)
    assert log.status == "success"
    assert calls == 3
    assert log.samples[0].trial_metadata[0]["relay"]["status"] == "error"


def test_parallel_trials_do_not_capture_each_other(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run, tmp_path / str(i), scene=f"scene-{i}") for i in range(2)]
        logs = [future.result()[0] for future in futures]
    roots = []
    for i, log in enumerate(logs):
        rows = _events(tmp_path / str(i), log)
        assert len(rows) == 14
        assert {r["metadata"]["scene_id"] for r in rows} == {f"scene-{i}"}
        roots.append(next(r["uuid"] for r in rows if r["category"] == "agent"))
    assert roots[0] != roots[1]


def test_disabled_mode_does_not_import_relay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("nemo_relay"):
            raise AssertionError("disabled tracing imported Relay")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    log, calls = _run(tmp_path, relay=False)
    assert log.status == "success" and calls == 3


def test_non_boolean_opt_in_is_rejected() -> None:
    with pytest.raises(ConfigError, match="relay must be a boolean"):
        LLMAgentPolicy(model="test", base_url="http://provider.test", env={}, relay="false")


def test_reported_usage_is_exported(tmp_path: Path) -> None:
    log, calls = _run(tmp_path, wire="messages")
    llms = [
        r for r in _events(tmp_path, log) if r["category"] == "llm" and r["scope_category"] == "end"
    ]
    assert calls == len(llms) == 3
    assert (
        sum(r["category_profile"]["annotated_response"]["usage"]["prompt_tokens"] for r in llms)
        == 21
    )
    assert (
        sum(r["category_profile"]["annotated_response"]["usage"]["completion_tokens"] for r in llms)
        == 9
    )
    assert (
        sum(r["category_profile"]["annotated_response"]["usage"]["cache_read_tokens"] for r in llms)
        == 15
    )
    assert (
        sum(
            r["category_profile"]["annotated_response"]["usage"]["cache_write_tokens"] for r in llms
        )
        == 6
    )


def test_sdk_finish_failure_does_not_repeat_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = pytest.importorskip("nemo_relay")
    baseline, _ = _run(tmp_path / "off", relay=False)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("private-export-error")

    monkeypatch.setattr(sdk.tools, "call_end", fail)
    traced, calls = _run(tmp_path / "on")
    assert calls == 3
    assert traced.status == baseline.status == "success"
    assert traced.samples[0].policy_transcripts == baseline.samples[0].policy_transcripts
    assert traced.results.metrics == baseline.results.metrics
    receipt = traced.samples[0].trial_metadata[0]["relay"]
    assert receipt["status"] == "error"
    assert "private-export-error" not in json.dumps(receipt)


def test_missing_extra_has_install_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def unavailable(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "inspect_robots_agent._relay":
            raise ImportError("unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ConfigError, match=r"Python 3.11\+ and inspect-robots-agent\[relay\]"):
        LLMAgentPolicy(model="test", base_url="http://provider.test", env={}, relay=True)


def test_sdk_stack_failure_preserves_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sdk = pytest.importorskip("nemo_relay")

    def fail() -> None:
        raise RuntimeError("private-sdk-error")

    monkeypatch.setattr(sdk, "create_scope_stack", fail)
    log, calls = _run(tmp_path)
    assert log.status == "success" and calls == 3
    assert log.samples[0].trial_metadata[0]["relay"]["status"] == "error"


def test_cancellation_preserves_exception_and_closes_trace(tmp_path: Path) -> None:
    from inspect_robots.errors import _CancelledTrial

    with pytest.raises(_CancelledTrial) as cancelled:
        _run(tmp_path, cancel=True)
    receipt = cancelled.value.record.metadata["relay"]
    assert receipt["status"] == "captured"
    rows = [json.loads(line) for line in (tmp_path / receipt["path"]).read_text().splitlines()]
    assert len(rows) == 4
    root_end = rows[-1]
    assert root_end["category"] == "agent"
    assert root_end["metadata"]["status"] == "cancelled"


def test_budget_stop_is_distinguished_from_model_tool(tmp_path: Path) -> None:
    log, calls = _run(tmp_path, max_llm_calls=1)
    rows = _events(tmp_path, log)
    tool_ends = [r for r in rows if r["category"] == "tool" and r["scope_category"] == "end"]
    assert calls == 1
    assert [r["name"] for r in tool_ends] == ["unknown", "give_up"]
    assert tool_ends[-1]["metadata"]["source"] == "policy"
    assert tool_ends[-1]["metadata"]["model_call_index"] is None


def test_stalled_writer_does_not_block_trial_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sdk = pytest.importorskip("nemo_relay")
    from inspect_robots_agent import _relay

    entered, release = Event(), Event()
    original = Path.open

    class StalledFile:
        def __init__(self, file: Any) -> None:
            self.file = file

        def write(self, text: str) -> Any:
            entered.set()
            release.wait(5)
            return self.file.write(text)

        def close(self) -> None:
            self.file.close()

    def opened(path: Path, *args: Any, **kwargs: Any) -> Any:
        file = original(path, *args, **kwargs)
        return StalledFile(file) if path.name == "events.jsonl" else file

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(_relay, "_DRAIN_TIMEOUT_S", 0.05)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run, tmp_path / "stalled")
        try:
            assert entered.wait(2)
            log, calls = future.result(timeout=1)
            assert log.status == "success" and calls == 3
            assert log.samples[0].trial_metadata[0]["relay"]["status"] == "error"
        finally:
            release.set()
    # The deferred callback/cleanup must not contaminate a later capture.
    sdk.subscribers.flush()
    monkeypatch.setattr(Path, "open", original)
    log, calls = _run(tmp_path / "next")
    assert calls == 3 and len(_events(tmp_path / "next", log)) == 14


def test_scoring_interrupt_does_not_retain_discarded_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sdk = pytest.importorskip("nemo_relay")
    from inspect_robots_agent._relay import RelayTrace

    registrations: list[str] = []
    traces: list[Any] = []
    original_register = sdk.subscribers.register
    original_init = RelayTrace.__init__

    def register(name: str, callback: Any) -> None:
        registrations.append(name)
        original_register(name, callback)

    def init(self: Any, *args: Any) -> None:
        original_init(self, *args)
        traces.append(weakref.ref(self))

    class InterruptedScore:
        name = "interrupted"

        def __call__(self, *args: Any) -> None:
            raise KeyboardInterrupt()

    monkeypatch.setattr(sdk.subscribers, "register", register)
    monkeypatch.setattr(RelayTrace, "__init__", init)

    def run_and_discard() -> None:
        try:
            _run(tmp_path, scorer=InterruptedScore())
        except KeyboardInterrupt:
            return
        pytest.fail("scoring cancellation was swallowed")

    run_and_discard()
    gc.collect()
    assert len(traces) == len(registrations) == 1
    assert traces[0]() is None
    assert sdk.subscribers.deregister(registrations[0]) is False
