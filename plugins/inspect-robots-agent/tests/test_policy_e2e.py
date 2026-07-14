"""LLMAgentPolicy end-to-end through real eval() on the mock cubepick world.

Conversations are scripted with httpx.MockTransport: no network, no LLM, no
hardware — the whole loop (bind, observation payloads, tool synthesis,
guardrails, policy-stop, budgets, error taxonomy) runs for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots_agent import LLMAgentPolicy

# --- scripted-conversation harness ---------------------------------------------


def _tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def _text_response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class _Script:
    """Serves queued responses; repeats the last one when the queue runs dry."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return httpx.Response(200, json=payload)


def _policy(script: _Script, **kwargs: Any) -> LLMAgentPolicy:
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(script),
        env={},
        **kwargs,
    )


def _task(max_steps: int = 40) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


class _RecordingSink(NullSink):
    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.records.append(record)


# --- tests -----------------------------------------------------------------------


def test_goal_runs_to_done_and_config_lands_in_log(tmp_path: Path) -> None:
    script = _Script(
        [
            _tool_response("move_by", {"deltas": {"dx": 0.1, "dy": 0.1}, "duration_s": 0.5}),
            _tool_response("done", {"summary": "close enough"}),
        ]
    )
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "success"
    (record,) = sink.records
    assert record.truncated is True
    assert record.termination_reason == "done"
    # 5 interpolated steps + 1 hold-still stop action.
    assert len(record.steps) == 6
    assert logs[0].eval.policy_config["model"] == "test/model"
    assert logs[0].eval.policy_config["max_llm_calls"] == 50


def test_outbound_messages_carry_state_images_and_tools(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "looked around"})])
    ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))
    first = script.requests[0]
    assert first["model"] == "test/model"
    assert [t["function"]["name"] for t in first["tools"]] == ["move_by", "done", "give_up"]
    system, goal, observation = first["messages"]
    assert "cubepick" in system["content"]
    assert goal["content"] == "Goal: reach the cube"
    text_parts = [p["text"] for p in observation["content"] if p["type"] == "text"]
    assert any("state[eef_pos]" in t for t in text_parts)
    image_parts = [p for p in observation["content"] if p["type"] == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_wild_swing_is_clamped_by_guardrails_but_not_without(tmp_path: Path) -> None:
    def run(approver: Any) -> TrialRecord:
        script = _Script(
            [
                _tool_response("move_by", {"deltas": {"dx": 5.0}, "duration_s": 0.5}),
                _tool_response("done", {"summary": "stop"}),
            ]
        )
        sink = _RecordingSink()
        ir_eval(
            _task(),
            _policy(script),
            CubePickEmbodiment(),
            log_dir=str(tmp_path),
            sinks=[sink],
            approver=approver,
        )
        return sink.records[0]

    space = CubePickEmbodiment().info.action_space
    guarded = run(ChainApprover(ClampApprover(space), DeltaLimitApprover(space)))
    approvals = [e for e in guarded.events if e.kind == "approval"]
    assert approvals, "the 1.0-per-step swing must be clamped to the 0.1 bound"
    assert all(abs(float(np.asarray(s.action.data)[0])) <= 0.1 for s in guarded.steps)

    unguarded = run(None)  # eval()'s Python-API default is the permissive AutoApprover
    assert not [e for e in unguarded.events if e.kind == "approval"]
    assert any(abs(float(np.asarray(s.action.data)[0])) > 0.1 for s in unguarded.steps)


def test_llm_call_budget_forces_give_up(tmp_path: Path) -> None:
    script = _Script([_tool_response("move_by", {"deltas": {"dx": 0.05}, "duration_s": 0.5})])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(),
        _policy(script, max_llm_calls=1),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        sinks=[sink],
    )
    assert logs[0].status == "success"
    (record,) = sink.records
    assert record.termination_reason == "give_up"


def test_persistent_non_tool_output_becomes_policy_error(tmp_path: Path) -> None:
    script = _Script([_text_response("I would rather write a poem about the cube.")])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    # A PolicyError marks the trial errored (never scored); the eval survives.
    (sample,) = logs[0].samples
    assert sample.status == "error"
    (record,) = sink.records
    assert record.error is not None and "no tool call" in record.error


def test_recoverable_tool_error_is_fed_back_and_corrected(tmp_path: Path) -> None:
    script = _Script(
        [
            _tool_response("move_by", {"deltas": {"dz": 0.1}, "duration_s": 0.5}),  # bad dim
            _tool_response("move_by", {"deltas": {"dx": 0.1}, "duration_s": 0.5}),
            _tool_response("done", {"summary": "recovered"}),
        ]
    )
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "success"
    assert sink.records[0].termination_reason == "done"
    # The error went back to the model as a tool message naming the offender.
    tool_messages = [
        m for request in script.requests for m in request["messages"] if m.get("role") == "tool"
    ]
    assert any("unknown dimension 'dz'" in str(m["content"]) for m in tool_messages)


def test_effort_defaults_low_and_is_tunable(tmp_path: Path) -> None:
    script = _Script([_tool_response("done", {"summary": "ok"})])
    logs = ir_eval(_task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path))
    # Robot control is latency-sensitive: low effort by default (guardrails
    # sit below the model, so this trades thinking time, not safety).
    assert script.requests[0]["reasoning_effort"] == "low"
    assert logs[0].eval.policy_config["effort"] == "low"

    script = _Script([_tool_response("done", {"summary": "ok"})])
    ir_eval(_task(), _policy(script, effort="high"), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert script.requests[0]["reasoning_effort"] == "high"

    script = _Script([_tool_response("done", {"summary": "ok"})])
    ir_eval(_task(), _policy(script, effort=None), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert "reasoning_effort" not in script.requests[0]

    with pytest.raises(ValueError, match="effort"):
        _policy(_Script([]), effort="turbo")


def test_registry_resolves_agent_policy() -> None:
    from inspect_robots.registry import resolve

    policy = resolve("policy", "agent", model="m", base_url="http://x/v1")
    assert isinstance(policy, LLMAgentPolicy)


def test_unbound_act_raises_clear_error() -> None:
    from inspect_robots.types import Observation

    policy = _policy(_Script([_text_response("hi")]))
    with pytest.raises(RuntimeError, match="bind"):
        policy.act(Observation())


def test_on_trial_end_writes_transcript_and_strips_images(tmp_path: Path) -> None:
    # A test for the on_trial_end hook in the agent policy.
    # It must strip images out but leave everything else, and store the path
    # in the trial metadata.
    script = _Script([_tool_response("done", {"summary": "done"})])
    sink = _RecordingSink()
    logs = ir_eval(
        _task(), _policy(script), CubePickEmbodiment(), log_dir=str(tmp_path), sinks=[sink]
    )
    assert logs[0].status == "success"

    # Metadata should contain transcript path
    (record,) = sink.records
    assert "transcript" in record.metadata
    transcript_rel = record.metadata["transcript"]
    assert transcript_rel.startswith("transcripts/")

    # Transcript should exist on disk
    transcript_path = tmp_path / transcript_rel
    assert transcript_path.is_file()

    # Verify the JSONL content and image stripping
    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # System, Goal, Observation, ToolCall, ToolResponse

    system, goal, obs, asst, tool_resp = [json.loads(line) for line in lines]
    assert system["role"] == "system"
    assert goal["role"] == "user"
    assert obs["role"] == "user"
    assert asst["role"] == "assistant"
    assert tool_resp["role"] == "tool"

    # Observation parts should not contain 'image_url' (they were stripped)
    obs_parts = obs["content"]
    assert any(p["type"] == "text" for p in obs_parts)
    assert all(p["type"] != "image_url" for p in obs_parts)
