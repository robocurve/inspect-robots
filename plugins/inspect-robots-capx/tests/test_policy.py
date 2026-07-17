"""CapxPolicy profile enforcement, codegen protocol, registry, and rollout integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from conftest import CapxStub
from inspect_robots import eval as ir_eval
from inspect_robots.embodiment import EmbodimentInfo
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
from inspect_robots.task import Task
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_capx import CapxPolicy, CapxPolicyConfig, capx_policy
from inspect_robots_capx.policy import (
    _EXECUTION_REPORT_CHAR_LIMIT,
    _REPORT_TRUNCATION_MARKER,
)


def _completion(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class _ScriptedTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.llm_requests: list[dict[str, Any]] = []
        self.capx = CapxStub()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path in {"/v1/chat/completions", "/v1/responses"}:
            self.llm_requests.append(json.loads(request.content))
            if not self.responses:
                raise AssertionError("scripted LLM response queue exhausted")
            return httpx.Response(200, json=self.responses.pop(0))
        return self.capx.handler(request)


def _info(
    *,
    mode: str = "joint_pos",
    gripper: str = "continuous",
    labels: tuple[str, ...] | None = ("j0", "j1", "gripper"),
    control_hz: float | None = 10.0,
    cameras: tuple[CameraSpec, ...] = (CameraSpec("front", 2, 2),),
    state: StateSpec | None = StateSpec(fields=(StateField("joint_pos", (3,)),)),
) -> EmbodimentInfo:
    return EmbodimentInfo(
        name="joint-test",
        action_space=Box(
            shape=(3,),
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            semantics=ActionSemantics(
                mode,  # type: ignore[arg-type]
                gripper=gripper,  # type: ignore[arg-type]
                dim_labels=labels,
            ),
        ),
        observation_space=ObservationSpace(cameras=cameras, state=state),
        control_hz=control_hz,
        is_simulated=True,
        docs="Joint state includes the gripper as its final labeled dimension.",
    )


def _observation(*, image: bool = False) -> Observation:
    images = {"front": np.zeros((2, 2, 3), dtype=np.uint8)} if image else {}
    return Observation(
        images=images,
        state={"joint_pos": np.array([0.0, 0.0, 1.0])},
        instruction="pick the cube",
    )


def _policy(script: _ScriptedTransport, **kwargs: Any) -> CapxPolicy:
    return CapxPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        sam3_url="http://sam.test",
        graspnet_url="http://grasp.test",
        pyroki_url="http://ik.test",
        transport=httpx.MockTransport(script),
        env={},
        **kwargs,
    )


def _bound_policy(script: _ScriptedTransport, **kwargs: Any) -> CapxPolicy:
    policy = _policy(script, **kwargs)
    policy.bind(_info())
    policy.reset(Scene(id="s0", instruction="pick the cube"))
    return policy


@pytest.mark.parametrize(
    "info",
    [
        _info(mode="eef_abs_pose"),
        _info(gripper="none"),
        _info(control_hz=None),
        _info(labels=("left_j0", "right_j0", "right_gripper")),
    ],
    ids=["ee_mode", "no_gripper_semantics", "missing_hz", "no_gripper_label"],
)
def test_bind_rejects_out_of_profile_embodiments(info: EmbodimentInfo) -> None:
    policy = _policy(_ScriptedTransport([]))

    with pytest.raises(ValueError, match="plan 0021 CaP-X v1 profile"):
        policy.bind(info)


def test_bind_requires_unambiguous_state_and_camera() -> None:
    policy = _policy(_ScriptedTransport([]))
    ambiguous_state = StateSpec(fields=(StateField("a", (3,)), StateField("b", (3,))))
    with pytest.raises(ValueError, match="exactly one state field"):
        policy.bind(_info(state=ambiguous_state))

    with pytest.raises(ValueError, match="one camera when camera=None"):
        policy.bind(_info(cameras=(CameraSpec("front", 2, 2), CameraSpec("wrist", 2, 2))))


def test_bind_uses_last_dimension_fallback_when_labels_are_absent() -> None:
    script = _ScriptedTransport([_completion("close_gripper()")])
    policy = _policy(script)
    policy.bind(_info(labels=None))
    policy.reset(Scene(id="s0", instruction="close"))

    chunk = policy.act(_observation())

    assert np.array_equal(chunk.actions[-1].data, np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    ("reply", "expected_code"),
    [
        (
            "import numpy as np\nmove_to_joints(np.array([0.1, -0.1]))",
            "import numpy as np\nmove_to_joints(np.array([0.1, -0.1]))",
        ),
        (
            "```python\nimport numpy as np\nmove_to_joints(np.array([0.1, -0.1]))\n```",
            "import numpy as np\nmove_to_joints(np.array([0.1, -0.1]))",
        ),
        (
            "REGENERATE\n```python\nimport numpy as np\nmove_to_joints(np.array([0.1, -0.1]))\n```",
            "import numpy as np\nmove_to_joints(np.array([0.1, -0.1]))",
        ),
    ],
    ids=["raw", "fenced", "regenerate_fenced"],
)
def test_raw_and_fenced_code_paths(reply: str, expected_code: str) -> None:
    policy = _bound_policy(_ScriptedTransport([_completion(reply)]))

    chunk = policy.act(_observation())

    assert chunk.actions[0].meta["code"] == expected_code
    assert np.array_equal(chunk.actions[-1].data, np.array([0.1, -0.1, 1.0]))


@pytest.mark.parametrize("word", ["FINISH", "GIVE_UP"])
def test_control_words_return_one_action_stop_hold(word: str) -> None:
    policy = _bound_policy(_ScriptedTransport([_completion(word)]))

    chunk = policy.act(_observation())

    assert len(chunk.actions) == 1
    assert np.array_equal(chunk.actions[0].data, np.array([0.0, 0.0, 1.0]))
    assert chunk.actions[0].meta == {"request_stop": True, "stop_reason": word}


@pytest.mark.parametrize(
    ("reply", "reason"),
    [("FINISH.", "FINISH"), ("GIVE_UP!", "GIVE_UP"), ("finish", "FINISH")],
)
def test_control_words_tolerate_punctuation_and_case(reply: str, reason: str) -> None:
    policy = _bound_policy(_ScriptedTransport([_completion(reply)]))

    chunk = policy.act(_observation())

    assert chunk.actions[0].meta == {"request_stop": True, "stop_reason": reason}


def test_prose_wrapped_fenced_code_executes_the_fenced_block() -> None:
    script = _ScriptedTransport(
        [
            _completion(
                "Sure, here is the corrected code:\n"
                "```python\nimport numpy as np\nmove_to_joints(np.array([0.1, -0.1]))\n```\n"
                "Let me know how it goes."
            ),
        ]
    )
    policy = _bound_policy(script)

    chunk = policy.act(_observation())

    assert np.array_equal(chunk.actions[-1].data, np.array([0.1, -0.1, 1.0]))


def test_perception_only_turn_feeds_report_back_then_returns_motion() -> None:
    script = _ScriptedTransport(
        [
            _completion("print('need another look')"),
            _completion("import numpy as np\nmove_to_joints(np.array([0.2, 0.0]))"),
        ]
    )
    policy = _bound_policy(script)

    chunk = policy.act(_observation())

    assert len(script.llm_requests) == 2
    second_messages = script.llm_requests[1]["messages"]
    assert "need another look" in second_messages[-1]["content"]
    assert np.array_equal(chunk.actions[-1].data, np.array([0.2, 0.0, 1.0]))


def test_failure_counter_persists_across_act_even_when_error_queued_actions() -> None:
    script = _ScriptedTransport(
        [
            _completion(
                "import numpy as np\nmove_to_joints(np.array([0.1, 0.0]))\n"
                "raise RuntimeError('after queue')"
            ),
            _completion("raise ValueError('next turn')"),
        ]
    )
    policy = _bound_policy(script, max_code_failures=2)

    first = policy.act(_observation())
    assert first.actions

    with pytest.raises(RuntimeError, match="2 consecutive turns"):
        policy.act(_observation())
    transcript = policy.transcript()
    assert transcript is not None and "next turn" in transcript[-1]["content"]


def test_clean_perception_turn_resets_consecutive_failure_counter() -> None:
    script = _ScriptedTransport(
        [
            _completion(
                "import numpy as np\nmove_to_joints(np.array([0.1, 0.0]))\nraise ValueError('one')"
            ),
            _completion("print('clean turn')"),
            _completion("raise ValueError('one again')"),
            _completion("FINISH"),
        ]
    )
    policy = _bound_policy(script, max_code_failures=2)

    policy.act(_observation())
    stopped = policy.act(_observation())

    assert stopped.actions[0].meta["stop_reason"] == "FINISH"


def test_call_budget_forces_give_up_after_clean_empty_turn() -> None:
    policy = _bound_policy(
        _ScriptedTransport([_completion("print('nothing queued')")]),
        max_llm_calls=1,
    )

    chunk = policy.act(_observation())

    assert chunk.actions[0].meta == {"request_stop": True, "stop_reason": "GIVE_UP"}


def test_execution_report_is_eager_and_code_lands_on_first_action() -> None:
    code = "import numpy as np\nprint('terminal feedback')\nmove_to_joints(np.array([0.1, 0.0]))"
    policy = _bound_policy(_ScriptedTransport([_completion(code)]))

    chunk = policy.act(_observation())
    transcript = policy.transcript()

    assert transcript is not None
    assert "terminal feedback" in transcript[-1]["content"]
    assert transcript[-1]["role"] == "user"
    assert chunk.actions[0].meta["code"] == code


def test_execution_report_is_tail_first_truncated_to_documented_cap() -> None:
    script = _ScriptedTransport(
        [
            _completion("print('x' * 20000)"),
            _completion("import numpy as np\nmove_to_joints(np.array([0.1, 0.0]))"),
        ]
    )
    policy = _bound_policy(script)

    policy.act(_observation())
    transcript = policy.transcript()
    assert transcript is not None
    reports = [
        message["content"]
        for message in transcript
        if isinstance(message.get("content"), str)
        and message["content"].startswith(_REPORT_TRUNCATION_MARKER)
    ]
    assert len(reports) == 1
    assert len(reports[0]) == _EXECUTION_REPORT_CHAR_LIMIT
    assert reports[0].endswith("Respond with FINISH, GIVE_UP, or REGENERATE followed by Python.")


def test_transcript_and_delta_are_deep_sanitized_and_reset_rewinds_cursor() -> None:
    policy = _bound_policy(_ScriptedTransport([_completion("FINISH")]))

    policy.act(_observation(image=True))
    first = policy.transcript_delta()
    assert first is not None
    assert "data:image" not in json.dumps(first)
    assert policy.transcript_delta() is None

    first[0]["content"] = "mutated"
    full = policy.transcript()
    assert full is not None and full[0]["content"] != "mutated"

    policy.reset(Scene(id="s1", instruction="again"))
    reset_delta = policy.transcript_delta()
    assert reset_delta is not None
    assert [message["role"] for message in reset_delta] == ["system", "user"]


def test_reset_clears_pyroki_full_config_warm_start() -> None:
    code = (
        "import numpy as np\n"
        "q = solve_ik(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))\n"
        "move_to_joints(q)"
    )
    script = _ScriptedTransport([_completion(code), _completion(code)])
    policy = _bound_policy(script)

    policy.act(_observation())
    policy.reset(Scene(id="s1", instruction="repeat"))
    policy.act(_observation())

    ik_bodies = [body for path, body in script.capx.requests if path == "/ik"]
    assert [body["prev_cfg"] for body in ik_bodies] == [None, None]


def test_config_and_close_lifecycle() -> None:
    policy = _policy(
        _ScriptedTransport([]),
        max_llm_calls=7,
        max_code_failures=4,
        gripper_open_is_high=False,
        transcript_echo=True,
    )

    assert isinstance(policy.config, CapxPolicyConfig)
    assert policy.config.max_llm_calls == 7
    assert policy.config.max_code_failures == 4
    assert policy.config.gripper_open_is_high is False
    policy.close()
    policy.close()
    assert policy._servers._http.is_closed


def test_unbound_act_is_actionable() -> None:
    policy = _policy(_ScriptedTransport([]))
    with pytest.raises(RuntimeError, match="before bind"):
        policy.act(_observation())


class _JointEmbodiment:
    def __init__(self) -> None:
        self._q = np.array([0.0, 0.0, 1.0])
        self.info = _info(control_hz=2.0)

    def _observation(self, instruction: str | None = None) -> Observation:
        return Observation(
            images={"front": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={"joint_pos": self._q.copy()},
            instruction=instruction,
            extra={
                "depth": lambda: np.ones((2, 2), dtype=np.float32),
                "intrinsics": np.eye(3),
                "extrinsics": np.eye(4),
            },
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._q = np.array([0.0, 0.0, 1.0])
        return self._observation(scene.instruction)

    def step(self, action: Action) -> StepResult:
        self._q = np.asarray(action.data, dtype=np.float64).copy()
        return StepResult(observation=self._observation())

    def close(self) -> None:
        return None


def test_end_to_end_joint_embodiment_rollout_carries_transcript(tmp_path: Path) -> None:
    code = (
        "import numpy as np\n"
        "objects = segment('red cube')\n"
        "q = solve_ik(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))\n"
        "move_to_joints(q)\n"
        "close_gripper()"
    )
    script = _ScriptedTransport([_completion(code), _completion("FINISH")])
    policy = _policy(script)
    task = Task(
        name="capx-e2e",
        scenes=[Scene(id="s0", instruction="pick the red cube")],
        scorer=success_at_end(),
        max_steps=40,
    )

    logs = ir_eval(task, policy, _JointEmbodiment(), log_dir=str(tmp_path))

    assert logs[0].status == "success"
    sample = logs[0].samples[0]
    assert sample.status == "success"
    (transcript,) = sample.policy_transcripts
    assert transcript is not None
    serialized = json.dumps(transcript)
    assert "segment('red cube')" in serialized
    assert "FINISH" in serialized
    assert "data:image" not in serialized
    policy.close()


def test_registry_entry_point_resolves_and_factory_forwards_kwargs() -> None:
    from inspect_robots.registry import resolve

    direct = capx_policy(
        model="test/model",
        base_url="http://llm.test/v1",
        max_llm_calls=7,
        env={},
    )
    resolved = resolve(
        "policy",
        "capx",
        model="test/model",
        base_url="http://llm.test/v1",
        max_llm_calls=9,
        env={},
    )

    assert isinstance(direct, CapxPolicy)
    assert isinstance(direct.config, CapxPolicyConfig)
    assert direct.config.max_llm_calls == 7
    assert isinstance(resolved, CapxPolicy)
    assert isinstance(resolved.config, CapxPolicyConfig)
    assert resolved.config.max_llm_calls == 9
    direct.close()
    resolved.close()
