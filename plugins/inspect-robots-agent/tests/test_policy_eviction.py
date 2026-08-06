from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import httpx
import pytest

from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent.policy import _evicted_view


def _image_parts(*names: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for name in names:
        parts.extend(
            [
                {"type": "text", "text": f"Camera {name}:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{name}"},
                },
            ]
        )
    return parts


def _observation_message(*names: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "state[q]: 0"},
            *_image_parts(*names),
        ],
    }


def _policy(**kwargs: Any) -> LLMAgentPolicy:
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        env={},
        **kwargs,
    )


def test_fewer_image_messages_than_horizon_returns_new_list_without_copies() -> None:
    messages = [
        {"role": "system", "content": "system"},
        _observation_message("top"),
        {"role": "assistant", "content": "next"},
    ]

    view = _evicted_view(messages, 2)

    assert view == messages
    assert view is not messages
    assert all(actual is original for actual, original in zip(view, messages, strict=True))


def test_aged_observation_replaces_labels_and_images_without_mutating_history() -> None:
    aged = _observation_message("top", "left", "right")
    kept = _observation_message("top")
    messages = [aged, kept]
    original = json.loads(json.dumps(messages))

    view = _evicted_view(messages, 1)

    assert view[0] is not aged
    assert view[0]["content"] == [
        {"type": "text", "text": "state[q]: 0"},
        {"type": "text", "text": "[3 camera frame(s) elided]"},
    ]
    assert messages == original


def test_kept_image_messages_and_non_image_messages_retain_identity() -> None:
    plain = {"role": "user", "content": "plain"}
    aged = _observation_message("old")
    kept = _observation_message("new")

    view = _evicted_view([plain, aged, kept], 1)

    assert view[0] is plain
    assert view[2] is kept


def test_anchor_marks_only_newest_stubbed_message() -> None:
    oldest = _observation_message("oldest")
    newest_stub = _observation_message("middle")
    kept = _observation_message("newest")

    view = _evicted_view([oldest, newest_stub, kept], 1, mark_anchor=True)

    assert "cache_anchor" not in view[0]
    assert view[1]["cache_anchor"] is True
    assert "cache_anchor" not in view[2]


def test_on_demand_image_only_message_stubs_to_one_text_part() -> None:
    image_only = {"role": "user", "content": _image_parts("wrist")}

    view = _evicted_view([image_only, _observation_message("new")], 1)

    assert view[0]["content"] == [{"type": "text", "text": "[1 camera frame(s) elided]"}]


def test_unlabelled_and_nontext_image_predecessors_are_preserved() -> None:
    first_image = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,first"}}],
    }
    nontext_label = {
        "role": "user",
        "content": [
            {"type": "vendor_label", "value": 1},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,second"}},
        ],
    }

    view = _evicted_view(
        [first_image, nontext_label, _observation_message("new")],
        1,
    )

    assert view[0]["content"] == [{"type": "text", "text": "[1 camera frame(s) elided]"}]
    assert view[1]["content"] == [
        {"type": "vendor_label", "value": 1},
        {"type": "text", "text": "[1 camera frame(s) elided]"},
    ]


def test_non_list_content_passes_through_untouched() -> None:
    plain = {"role": "user", "content": "Respond with exactly one tool call."}

    view = _evicted_view([plain], 1, mark_anchor=True)

    assert view[0] is plain


@pytest.mark.parametrize("value", [True, 0, -1, "2", ""])
def test_image_horizon_rejects_invalid_values(value: object) -> None:
    expected = (
        "image_horizon must be an int >= 1, or None to send full image history.\n"
        "fix: pass -P image_horizon=N or -P image_horizon=none"
    )

    with pytest.raises(ConfigError) as exc_info:
        _policy(image_horizon=value)

    assert str(exc_info.value) == expected


@pytest.mark.parametrize("value", [None, 2])
def test_image_horizon_accepts_none_and_positive_int(value: int | None) -> None:
    policy = _policy(image_horizon=value)

    assert policy.config.image_horizon == value


def test_none_horizon_sends_all_observation_images_across_three_cycles() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_move",
                                    "type": "function",
                                    "function": {
                                        "name": "move_by",
                                        "arguments": json.dumps(
                                            {
                                                "deltas": {"dx": 0.01},
                                                "note": "I see the cube and move toward it.",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    embodiment = CubePickEmbodiment()
    policy = _policy(
        image_horizon=None,
        transport=httpx.MockTransport(handler),
    )
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="reach")
    policy.reset(scene)
    observation = embodiment.reset(scene, seed=0)

    for _ in range(3):
        policy.act(observation)

    for cycle, body in enumerate(requests, start=1):
        image_messages = [
            message
            for message in body["messages"]
            if isinstance(message.get("content"), list)
            and any(part.get("type") == "image_url" for part in message["content"])
        ]
        assert len(image_messages) == cycle

    assert (
        sum(
            1
            for message in policy._messages
            if isinstance(message.get("content"), list)
            and any(part.get("type") == "image_url" for part in message["content"])
        )
        == 3
    )


def test_eviction_boundary_and_already_stubbed_prefix_are_byte_stable() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_move",
                                    "type": "function",
                                    "function": {
                                        "name": "move_by",
                                        "arguments": json.dumps(
                                            {
                                                "deltas": {"dx": 0.01},
                                                "note": "I see the cube and move toward it.",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    embodiment = CubePickEmbodiment()
    policy = _policy(
        image_horizon=2,
        transport=httpx.MockTransport(handler),
    )
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="reach")
    policy.reset(scene)
    observation = embodiment.reset(scene, seed=0)

    for _ in range(4):
        policy.act(observation)

    def stub_indices(body: dict[str, Any]) -> set[int]:
        return {
            index
            for index, message in enumerate(body["messages"])
            if isinstance(message.get("content"), list)
            and any(
                part.get("text", "").endswith("camera frame(s) elided]")
                for part in message["content"]
            )
        }

    for previous, current in pairwise(requests):
        newly_stubbed = stub_indices(current) - stub_indices(previous)
        boundary = min(newly_stubbed) if newly_stubbed else len(previous["messages"])
        previous_prefix = json.dumps(
            previous["messages"][:boundary],
            separators=(",", ":"),
        )
        current_prefix = json.dumps(
            current["messages"][:boundary],
            separators=(",", ":"),
        )
        assert current_prefix == previous_prefix

    all_stubbed_indices = set().union(*(stub_indices(body) for body in requests))
    assert len(all_stubbed_indices) == 2
    for message_index in all_stubbed_indices:
        stable_forms = {
            json.dumps(body["messages"][message_index], separators=(",", ":"))
            for body in requests
            if message_index in stub_indices(body)
        }
        assert len(stable_forms) == 1


def test_on_demand_allows_re_reveal_after_eviction() -> None:
    responses = [
        # Call 1: take_pic top
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "take_pic",
                                    "arguments": json.dumps(
                                        {"cameras": ["top"], "note": "Check top"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        # Call 2: take_pic front
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "take_pic",
                                    "arguments": json.dumps(
                                        {"cameras": ["front"], "note": "Check front"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        # Call 3: take_pic left
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_3",
                                "type": "function",
                                "function": {
                                    "name": "take_pic",
                                    "arguments": json.dumps(
                                        {"cameras": ["left"], "note": "Check left"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        # Call 4: take_pic top again (was evicted since image_horizon=2)
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_4",
                                "type": "function",
                                "function": {
                                    "name": "take_pic",
                                    "arguments": json.dumps(
                                        {"cameras": ["top"], "note": "Check top again"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        # Call 5: take_pic left again (still visible; should be refused)
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_5",
                                "type": "function",
                                "function": {
                                    "name": "take_pic",
                                    "arguments": json.dumps(
                                        {"cameras": ["left"], "note": "Check left again"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        # Call 6: move_by
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_6",
                                "type": "function",
                                "function": {
                                    "name": "move_by",
                                    "arguments": json.dumps(
                                        {"deltas": {"dx": 0.01}, "note": "Move"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
    ]

    response_iter = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(response_iter))

    embodiment = CubePickEmbodiment()
    policy = _policy(
        images="on_demand",
        image_horizon=2,
        transport=httpx.MockTransport(handler),
    )
    policy.bind(embodiment.info)
    scene = Scene(id="s0", instruction="reach")
    policy.reset(scene)

    import numpy as np

    from inspect_robots.types import Observation

    images = {
        name: np.full((2, 2, 3), i, dtype=np.uint8)
        for i, name in enumerate(["top", "front", "left"], start=1)
    }
    observation = Observation(state={"q": np.array([0.0])}, images=images, extra={"env_step": 0})

    chunk = policy.act(observation)
    assert chunk is not None

    tool_results = [
        message["content"] for message in policy._messages if message.get("role") == "tool"
    ]
    assert tool_results[:4] == [
        "captured 1 frame(s): 'top'",
        "captured 1 frame(s): 'front'",
        "captured 1 frame(s): 'left'",
        "captured 1 frame(s): 'top'",
    ]
    assert tool_results[4].startswith("already shown for this observation: 'left'")
