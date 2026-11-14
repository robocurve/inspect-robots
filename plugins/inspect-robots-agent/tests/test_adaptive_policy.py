"""Exercise both adaptive flags through the Anthropic Messages transport."""

from __future__ import annotations

import base64
import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.rollout import Record
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Observation
from inspect_robots_agent.policy import LLMAgentPolicy

_CONFIG: dict[str, Any] = {
    "model": "anthropic/claude-fable-5-2",
    "env": {"ANTHROPIC_API_KEY": "test-only"},
    "wire": "messages",
    "effort": "high",
    "image_horizon": None,
    "wire_capture": False,
}


def _without_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **m,
            "content": [
                {k: v for k, v in block.items() if k != "cache_control"} for block in m["content"]
            ],
        }
        for m in messages
    ]


def _image_widths(message: dict[str, Any]) -> list[int]:
    return [
        struct.unpack(">I", base64.b64decode(block["source"]["data"])[16:20])[0]
        for block in message["content"]
        if block["type"] == "image"
    ]


def test_adaptive_history_remains_append_only_across_recovery_and_reset(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []
    scripted_targets = [0.9, 0.2, 0.2, 7.0, None, 0.2, 0.2]

    def handle(request: httpx.Request) -> httpx.Response:
        assert "mid-conversation-output-config-2026-07-01" in request.headers["anthropic-beta"]
        body = json.loads(request.content)
        i = len(requests)
        requests.append(body)
        thinking = {"type": "thinking", "thinking": "", "signature": f"signature-{i}"}
        target = scripted_targets[i]
        output = (
            [{"type": "text", "text": "Let me check the grasp."}]
            if target is None
            else [
                {
                    "type": "tool_use",
                    "id": f"toolu_{i}",
                    "name": "move_joints",
                    "input": {"targets": {"left_gripper": target}, "note": "Check move."},
                }
            ]
        )
        return httpx.Response(
            200,
            json={
                "content": [thinking, *output],
                "stop_reason": "end_turn" if target is None else "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    policy = LLMAgentPolicy(
        **_CONFIG,
        effort_policy="event",
        camera_history_policy="transition_witness",
        transport=httpx.MockTransport(handle),
    )
    policy.bind(
        EmbodimentInfo(
            name="paired-grasp",
            action_space=Box(
                shape=(2,),
                low=np.zeros(2),
                high=np.ones(2),
                semantics=ActionSemantics("joint_pos", dim_labels=("joint", "left_gripper")),
            ),
            observation_space=ObservationSpace(
                state=StateSpec(fields=(StateField(key="q", shape=(2,)),))
            ),
            control_hz=10.0,
            is_simulated=True,
        )
    )
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    images = dict.fromkeys(("top_cam", "left_cam", "right_cam"), pixels)

    def observation(step: int, gripper: float) -> Observation:
        return Observation(
            images=images, state={"q": np.array([0.0, gripper])}, extra={"env_step": step}
        )

    scene = Scene(id="s0", instruction="Grasp the object")
    policy.reset(scene)
    policy.act(observation(0, 0.9))  # initial: high/full
    policy.act(observation(100, 0.9))  # routine: medium/smaller; commands a close
    policy.act(observation(200, 0.2))  # gripper event: high, full pre/post
    policy.act(observation(300, 0.2))  # routine; invalid call escalates, then a tool-less retry

    messages = [_without_breakpoints(r["messages"]) for r in requests]
    for earlier, later in pairwise(messages):
        assert earlier == later[: len(earlier)]
    assert all(r["output_config"] == {"effort": "high"} for r in requests)
    assert _image_widths(messages[0][-1]) == [224, 224, 224]
    assert _image_widths(messages[1][-2]) == [224, 112, 112]
    assert _image_widths(messages[2][-2]) == [224, 224, 224, 224, 224]
    assert _image_widths(messages[4][-2]) == [224, 224]
    assert messages[4][-3]["content"][0]["type"] == "tool_result"
    assert any(
        block.get("signature") == "signature-4"
        for message in messages[-1]
        for block in message["content"]
    )
    assert all(image.shape == (224, 224, 3) for image in images.values())
    record = Record(scene_id="s0", epoch=0, seed=0)
    policy.on_trial_end(record, str(tmp_path), "test-run")
    assert [
        (event["reason"], event["effort"]) for event in record.metadata["observation_events"]
    ] == [
        ("initial", "high"),
        ("routine", "medium"),
        ("gripper_command", "high"),
        ("routine", "medium"),
        ("recovery", "high"),
    ]
    transcript = policy.transcript()
    assert transcript and "signature-" not in json.dumps(transcript)

    policy.reset(scene)
    policy.act(observation(0, 0.2))
    first = _without_breakpoints(requests[-1]["messages"])
    assert [m["role"] for m in first] == ["user", "user"]
    assert _image_widths(first[-1]) == [224, 224, 224]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"effort_policy": "event", "effort": "low"}, "effort=high"),
        ({"effort_policy": "event", "image_horizon": 2}, "image_horizon=none"),
        ({"effort_policy": "event", "model": "anthropic/claude-opus-5-5"}, "Fable 5.2"),
        ({"camera_history_policy": "transition_witness", "images": "on_demand"}, "images=always"),
    ],
)
def test_adaptive_flags_reject_unsupported_configurations(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        LLMAgentPolicy(**(_CONFIG | overrides))
