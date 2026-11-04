"""Golden request bodies: what the agent sends on each HTTP wire for one fixed trial.

The trial covers a camera frame, joint state, an embodiment motion report and
motor temperatures, and the move, sleep and done tools. Images are replaced by
a short hash and the clock is fixed, so a diff in the stored files is a real
change to the prompt, the tool schemas or the observation text. Regenerate
deliberately with ``INSPECT_ROBOTS_AGENT_UPDATE_GOLDEN=1`` and review the diff.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import Task
from inspect_robots import eval as ir_eval
from inspect_robots.embodiment import EmbodimentInfo, StepResult
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation
from inspect_robots_agent import LLMAgentPolicy

_GOLDEN_DIR = Path(__file__).parent / "golden"
_UPDATE = os.environ.get("INSPECT_ROBOTS_AGENT_UPDATE_GOLDEN") == "1"
_DATA_URL = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")
_BASE64_FIELDS = frozenset({"data"})

_TURNS: list[tuple[str, dict[str, Any]]] = [
    ("move_joints", {"targets": {"joint": 0.25}, "note": "Moving the joint to a quarter."}),
    ("sleep", {"seconds": 2, "note": "Resting before the last check."}),
    ("done", {"summary": "The joint reached its target.", "hindsight": "none"}),
]


class _GoldenEmbodiment:
    """One absolute joint with a camera and embodiment-published extras."""

    def __init__(self) -> None:
        self._q = np.array([0.0])
        self.info = EmbodimentInfo(
            name="golden-test",
            action_space=Box(
                shape=(1,),
                low=np.array([-1.0]),
                high=np.array([1.0]),
                semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="top", height=2, width=2, channels=3),),
                state=StateSpec(fields=(StateField(key="q", shape=(1,)),)),
            ),
            control_hz=10.0,
            is_simulated=True,
            docs="A single test joint. Positions are radians in [-1, 1].",
        )

    def _observation(self, instruction: str | None = None) -> Observation:
        level = round(float(self._q[0]) * 100) % 256
        frame = np.full((2, 2, 3), level, dtype=np.uint8)
        return Observation(
            images={"top": frame},
            state={"q": self._q.copy()},
            instruction=instruction,
            extra={
                "motion_report": {"lines": ["joint: on target."]},
                "motor_temps": {"lines": ["motor temperatures: joint 31 C of a 60 C limit."]},
            },
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._q = np.array([0.0])
        return self._observation(scene.instruction)

    def step(self, action: Action) -> StepResult:
        self._q = np.asarray(action.data, dtype=np.float64).copy()
        return StepResult(observation=self._observation())

    def close(self) -> None:
        return None


def _call_payload(wire: str, index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    call_id = f"call_{index}"
    if wire == "chat":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    }
                }
            ]
        }
    if wire == "responses":
        return {
            "id": f"resp_{index}",
            "status": "completed",
            "output": [
                {
                    "id": f"fc_{index}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            ],
        }
    if wire == "messages":
        return {
            "id": f"msg_{index}",
            "type": "message",
            "content": [{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
            "stop_reason": "tool_use",
        }
    return {
        "id": f"interaction_{index}",
        "status": "completed",
        "steps": [{"type": "function_call", "id": call_id, "name": name, "arguments": arguments}],
    }


class _Script:
    def __init__(self, wire: str) -> None:
        self.wire = wire
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        index = min(len(self.requests) - 1, len(_TURNS) - 1)
        name, arguments = _TURNS[index]
        return httpx.Response(200, json=_call_payload(self.wire, index, name, arguments))


def _digest(value: str) -> str:
    return "<png " + hashlib.sha256(value.encode()).hexdigest()[:12] + ">"


def _normalize(value: Any, key: str | None = None) -> Any:
    """Replace image payloads with short hashes so the files stay small and stable."""
    if isinstance(value, str):
        if key in _BASE64_FIELDS and len(value) > 64:
            return _digest(value)
        return _DATA_URL.sub(lambda match: _digest(match.group(0)), value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {name: _normalize(item, name) for name, item in value.items()}
    return value


def _fake_clock() -> Any:
    ticks = iter(float(second) for second in range(0, 10_000, 7))
    return lambda: next(ticks)


_MODELS = {
    "chat": ("test/model", "http://llm.test/v1"),
    "responses": ("openai/test-model", "http://llm.test/v1"),
    "messages": ("anthropic/test-model", "http://llm.test/v1"),
    "interactions": ("google/test-model", "http://llm.test/v1beta"),
}


@pytest.mark.parametrize("wire", sorted(_MODELS))
def test_request_bodies_match_the_golden_files(wire: str, tmp_path: Path) -> None:
    model, base_url = _MODELS[wire]
    script = _Script(wire)
    policy = LLMAgentPolicy(
        model=model,
        base_url=base_url,
        api_key_env="TEST_KEY",
        wire=wire,
        depth="off",
        transport=httpx.MockTransport(script),
        clock=_fake_clock(),
        env={"TEST_KEY": "sk-test"},
    )
    task = Task(
        name="golden",
        scenes=[Scene(id="s0", instruction="Move the joint to a quarter.", init_seed=0)],
        scorer=success_at_end(),
        max_steps=200,
    )

    ir_eval(task, policy, _GoldenEmbodiment(), log_dir=str(tmp_path))

    assert len(script.requests) == len(_TURNS)
    actual = _normalize(script.requests)
    golden = _GOLDEN_DIR / f"wire-{wire}.json"
    if _UPDATE:
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(golden.read_text(encoding="utf-8"))
    assert actual == expected, (
        f"{wire} request bodies changed; if intended, regenerate with "
        "INSPECT_ROBOTS_AGENT_UPDATE_GOLDEN=1 and review the diff of "
        f"{golden.relative_to(Path(__file__).parent)}"
    )


def test_normalize_hashes_data_urls_and_long_base64_fields() -> None:
    url = "data:image/png;base64," + "A" * 80
    out = _normalize({"image_url": {"url": url}, "data": "B" * 80, "text": f"see {url}", "n": 1})
    assert out["image_url"]["url"].startswith("<png ")
    assert out["data"].startswith("<png ")
    assert out["text"].startswith("see <png ")
    assert out["n"] == 1
    assert _normalize({"data": "short"}) == {"data": "short"}
