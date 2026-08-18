"""Automatic task generation stays deterministic, visual, lazy, and fail-fast."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval as run_eval
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import derive_seed
from inspect_robots.scene import Scene
from inspect_robots.taskgen import _DEFAULT_INSTRUCTIONS, _REPLY_CONTRACT, generate_scene
from inspect_robots.types import Observation


def _reply_body(reply: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": reply}}]}).encode()


def _success_post(
    captured: dict[str, Any],
    reply: str = "TASK: Reach the green cube.\nRUBRIC:\nThe end effector touches the cube.",
) -> Any:
    def post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        captured.update(url=url, headers=headers, body=json.loads(body_bytes))
        return 200, _reply_body(reply)

    return post


class _MultiCameraEmbodiment(CubePickEmbodiment):
    def __init__(self, images: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.images = images
        self.reset_scene: Scene | None = None
        self.reset_seed: int | None = None

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.reset_scene = scene
        self.reset_seed = seed
        return Observation(images=self.images)


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: str = "TASK: Reach the green cube.\nRUBRIC:\nThe end effector touches the cube.",
    embodiment: CubePickEmbodiment | None = None,
    **kwargs: Any,
) -> tuple[Scene, dict[str, Any]]:
    monkeypatch.setenv("TASKGEN_KEY", "secret")
    captured: dict[str, Any] = {}
    scene = generate_scene(
        embodiment or CubePickEmbodiment(),
        model="vision-model",
        api_key_env="TASKGEN_KEY",
        base_url="https://example.test/v1/",
        http_post=_success_post(captured, reply),
        **kwargs,
    )
    return scene, captured


def _assert_fix(error: pytest.ExceptionInfo[ConfigError]) -> str:
    message = str(error.value)
    assert "\nfix:" in message
    return message


@pytest.mark.parametrize("model", [None, "", " \t "])
def test_model_is_required_with_guidance(model: str | None) -> None:
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(CubePickEmbodiment(), model=model)
    assert "model" in _assert_fix(exc_info)


def test_prompt_overrides_are_mutually_exclusive() -> None:
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(
            CubePickEmbodiment(),
            model="vision-model",
            instructions="custom",
            instructions_file="prompt.txt",
        )
    assert "both instructions" in _assert_fix(exc_info)


@pytest.mark.parametrize("max_cameras", [0, -1])
def test_max_cameras_must_be_positive(max_cameras: int) -> None:
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(CubePickEmbodiment(), model="vision-model", max_cameras=max_cameras)
    assert "max_cameras" in _assert_fix(exc_info)


def test_scene_id_must_be_non_empty() -> None:
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(CubePickEmbodiment(), model="vision-model", scene_id="")
    assert "scene_id" in _assert_fix(exc_info)


def test_none_seed_cannot_match_eval_seed() -> None:
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(CubePickEmbodiment(), model="vision-model", seed=None)
    assert "same integer seed" in _assert_fix(exc_info)


def test_unreadable_instructions_file_has_guidance(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(
            CubePickEmbodiment(),
            model="vision-model",
            instructions_file=str(missing),
        )
    message = _assert_fix(exc_info)
    assert "instructions_file" in message
    # The message interpolates the path with !r, so match its repr form
    # (Windows backslashes are escaped there, not raw).
    assert repr(str(missing)) in message


def test_api_key_must_be_set_and_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMPTY_TASKGEN_KEY", raising=False)
    for value in (None, ""):
        if value is not None:
            monkeypatch.setenv("EMPTY_TASKGEN_KEY", value)
        with pytest.raises(ConfigError) as exc_info:
            generate_scene(
                CubePickEmbodiment(),
                model="vision-model",
                api_key_env="EMPTY_TASKGEN_KEY",
            )
        assert "$EMPTY_TASKGEN_KEY" in _assert_fix(exc_info)


def test_peek_requires_at_least_one_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKGEN_KEY", "secret")
    embodiment = _MultiCameraEmbodiment({})
    with pytest.raises(ConfigError) as exc_info:
        generate_scene(
            embodiment,
            model="vision-model",
            api_key_env="TASKGEN_KEY",
        )
    assert "no camera images" in _assert_fix(exc_info)


def test_default_prompt_cube_frame_request_and_scene_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, captured = _generate(monkeypatch)

    assert scene.id == "auto-0"
    assert scene.instruction == "Reach the green cube."
    assert scene.metadata == {
        "rubric": "The end effector touches the cube.",
        "taskgen": {
            "model": "vision-model",
            "base_url": "https://example.test/v1/",
        },
    }
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "vision-model"
    messages = body["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    parts = messages[0]["content"]
    assert parts[0] == {
        "type": "text",
        "text": f"{_DEFAULT_INSTRUCTIONS}\n\n{_REPLY_CONTRACT}",
    }
    assert parts[1] == {"type": "text", "text": "camera 'top'"}
    image_url = parts[2]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_instructions_file_is_read_and_fixed_contract_is_appended(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt = tmp_path / "designer.txt"
    prompt.write_text("Use only red objects.", encoding="utf-8")

    _scene, captured = _generate(monkeypatch, instructions_file=str(prompt))

    text = captured["body"]["messages"][0]["content"][0]["text"]
    assert text == f"Use only red objects.\n\n{_REPLY_CONTRACT}"


def test_custom_instructions_cannot_override_reply_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scene, captured = _generate(monkeypatch, instructions="Describe a careful task.")

    text = captured["body"]["messages"][0]["content"][0]["text"]
    assert text.startswith("Describe a careful task.\n\n")
    assert text.endswith(_REPLY_CONTRACT)


def test_cameras_are_sorted_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    embodiment = _MultiCameraEmbodiment({"side": frame, "front": frame, "top": frame})

    _scene, captured = _generate(monkeypatch, embodiment=embodiment, max_cameras=2)

    parts = captured["body"]["messages"][0]["content"]
    assert [part["text"] for part in parts if part["type"] == "text"][1:] == [
        "camera 'front'",
        "camera 'side'",
    ]
    assert sum(part["type"] == "image_url" for part in parts) == 2


def test_peek_scene_and_seed_match_eval_first_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    embodiment = _MultiCameraEmbodiment({"top": frame})

    _generate(monkeypatch, embodiment=embodiment, scene_id="generated", seed=37)

    assert embodiment.reset_scene == Scene(
        id="generated-peek",
        instruction="Hold still and observe the scene.",
    )
    assert embodiment.reset_seed == derive_seed(37, None, 0)
    assert inspect.signature(generate_scene).parameters["seed"].default == 0
    assert inspect.signature(run_eval).parameters["seed"].default == 0


@pytest.mark.parametrize(
    ("reply", "diagnosis", "extend"),
    [
        ("", "RUBRIC:", False),
        ("RUBRIC:\nTouch the cube.", "TASK:", True),
        ("TASK: Touch the cube.", "RUBRIC:", True),
        ("TASK: Touch the cube.\nRUBRIC:   \n \t", "empty rubric", False),
    ],
)
def test_reply_contract_failures_include_bounded_excerpt(
    monkeypatch: pytest.MonkeyPatch, reply: str, diagnosis: str, extend: bool
) -> None:
    long_reply = reply + ("x" * 700 if extend else "")
    with pytest.raises(ConfigError) as exc_info:
        _generate(monkeypatch, reply=long_reply)
    message = _assert_fix(exc_info)
    assert diagnosis in message
    assert "x" * 501 not in message


def test_inline_rubric_is_joined_to_following_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    scene, _captured = _generate(
        monkeypatch,
        reply="TASK: Stack the blocks.\nRUBRIC: The red block is on top.\nThe stack is stable.",
    )
    assert scene.metadata["rubric"] == "The red block is on top.\nThe stack is stable."


def test_last_rubric_marker_anchors_last_prior_task(monkeypatch: pytest.MonkeyPatch) -> None:
    scene, _captured = _generate(
        monkeypatch,
        reply=(
            "TASK: Ignore this draft.\n"
            "RUBRIC:\nIgnore this draft rubric.\n"
            "TASK: Put the cube in the corner.\n"
            "RUBRIC:\n"
            "The cube is in the corner.\n"
            "TASK: Put the cube in the corner."
        ),
    )
    assert scene.instruction == "Put the cube in the corner."
    assert scene.metadata["rubric"] == (
        "The cube is in the corner.\nTASK: Put the cube in the corner."
    )


@pytest.mark.parametrize(
    ("injected", "expected"),
    [
        (
            "summary request failed with HTTP 503: injected outage\n"
            "fix: check --base-url and --model",
            "task generation request failed with HTTP 503: injected outage",
        ),
        (
            "summary endpoint returned a malformed reply: injected bytes\n"
            "fix: use an OpenAI-compatible endpoint",
            "task generation endpoint returned a malformed reply: injected bytes",
        ),
        (
            "chat request failed: injected outage\n"
            "fix: check the base URL and network connectivity",
            "task generation request failed: injected outage",
        ),
    ],
)
def test_wire_config_errors_are_reworded_from_injected_messages(
    monkeypatch: pytest.MonkeyPatch, injected: str, expected: str
) -> None:
    monkeypatch.setenv("CUSTOM_TASK_KEY", "secret")

    def fail_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        raise ConfigError(injected)

    with pytest.raises(ConfigError) as exc_info:
        generate_scene(
            CubePickEmbodiment(),
            model="vision-model",
            api_key_env="CUSTOM_TASK_KEY",
            http_post=fail_post,
        )
    message = str(exc_info.value)
    assert expected in message
    assert "injected" in message
    assert "-A k=v" in message
    assert "$CUSTOM_TASK_KEY" in message
    assert "--base-url" not in message
    assert message.count("fix:") == 1


def test_timeout_is_reworded_as_task_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TASKGEN_KEY", "secret")

    def timeout_post(url: str, headers: dict[str, str], body_bytes: bytes) -> tuple[int, bytes]:
        raise TimeoutError("injected timeout")

    with pytest.raises(ConfigError) as exc_info:
        generate_scene(
            CubePickEmbodiment(),
            model="vision-model",
            api_key_env="TASKGEN_KEY",
            http_post=timeout_post,
        )
    message = _assert_fix(exc_info)
    assert "task generation request failed: injected timeout" in message
    assert "-A k=v" in message


def test_bare_package_import_does_not_import_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import inspect_robots; print('inspect_robots.cli' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_unencodable_frame_is_a_guided_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame the PNG encoder rejects fails fast naming the camera, never a raw TypeError."""
    embodiment = _MultiCameraEmbodiment({"wrist": np.zeros((4, 4, 3), dtype=np.float32)})
    with pytest.raises(ConfigError) as error:
        _generate(monkeypatch, embodiment=embodiment)
    message = _assert_fix(error)
    assert "camera 'wrist'" in message
    assert "uint8" in message
