"""The execution namespace persists per trial and turns helper failures into feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest

from conftest import CapxStub
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Observation
from inspect_robots_capx._codec import npy_b64_decode
from inspect_robots_capx._motion import MotionQueue
from inspect_robots_capx._sandbox import CodeSandbox
from inspect_robots_capx._servers import CapxServerClients


def _sandbox(capx_stub: CapxStub) -> tuple[CodeSandbox, MotionQueue, CapxServerClients]:
    space = Box(
        shape=(3,),
        low=np.array([-1.0, -1.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            gripper="continuous",
            dim_labels=("j0", "j1", "gripper"),
        ),
    )
    motion = MotionQueue(space, control_hz=10.0, max_speed_frac=0.1, gripper_index=2)
    servers = CapxServerClients(
        sam3_url="http://sam.test",
        graspnet_url="http://grasp.test",
        pyroki_url="http://ik.test",
        transport=httpx.MockTransport(capx_stub.handler),
    )
    sandbox = CodeSandbox(
        servers=servers,
        motion=motion,
        camera="front",
        state_key="joint_pos",
    )
    return sandbox, motion, servers


def _observation(**extra: Any) -> Observation:
    return Observation(
        images={"front": np.zeros((2, 2, 3), dtype=np.uint8)},
        state={"joint_pos": np.array([0.0, 0.0, 1.0])},
        extra=extra,
    )


def test_namespace_persists_within_trial_and_resets_across_trials(capx_stub: CapxStub) -> None:
    sandbox, _, _ = _sandbox(capx_stub)
    sandbox.set_observation(_observation())

    first = sandbox.execute("counter = 1")
    second = sandbox.execute("counter += 1\nprint(counter)")
    sandbox.reset()
    sandbox.set_observation(_observation())
    after_reset = sandbox.execute("print(counter)")

    assert first.raised is False
    assert second.stdout == "2\n"
    assert second.stderr == ""
    assert after_reset.raised is True
    assert "NameError" in after_reset.stderr


def test_stdout_stderr_and_traceback_are_captured(capx_stub: CapxStub) -> None:
    sandbox, _, _ = _sandbox(capx_stub)
    sandbox.set_observation(_observation())

    result = sandbox.execute(
        "import sys\nprint('out')\nprint('warning', file=sys.stderr)\nraise ValueError('boom')"
    )

    assert result.stdout == "out\n"
    assert "warning\n" in result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    assert "ValueError: boom" in result.stderr
    assert result.raised is True


def test_helper_error_becomes_stderr_instead_of_crashing(capx_stub: CapxStub) -> None:
    sandbox, motion, _ = _sandbox(capx_stub)
    sandbox.set_observation(_observation())

    result = sandbox.execute("plan_grasp(segment('cube')[0]['mask'])")

    assert result.raised is True
    assert "observation.extra is missing 'depth'" in result.stderr
    assert "zero-argument callable" in result.stderr
    assert motion.has_actions() is False


@pytest.mark.parametrize(
    "depth_factory",
    [
        lambda depth: depth,
        lambda depth: lambda: depth,
    ],
    ids=["raw_array", "zero_arg_callable"],
)
def test_depth_accepts_raw_array_and_zero_arg_callable(
    capx_stub: CapxStub,
    depth_factory: Callable[[np.ndarray], Any],
) -> None:
    sandbox, _, _ = _sandbox(capx_stub)
    depth = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    sandbox.set_observation(
        _observation(depth=depth_factory(depth), intrinsics=np.eye(3), extrinsics=np.eye(4))
    )

    result = sandbox.execute("mask = segment('cube')[0]['mask']\nposes, scores = plan_grasp(mask)")

    assert result.raised is False
    plan_body = next(body for path, body in capx_stub.requests if path == "/plan")
    assert np.array_equal(npy_b64_decode(plan_body["depth_base64"]), depth)


def test_motion_helpers_share_one_cursor_and_queue(capx_stub: CapxStub) -> None:
    sandbox, motion, _ = _sandbox(capx_stub)
    sandbox.set_observation(_observation())

    result = sandbox.execute("move_to_joints(np.array([0.1, -0.1]))")
    assert result.raised is True
    assert "NameError: name 'np' is not defined" in result.stderr

    recovered = sandbox.execute(
        "import numpy as np\nmove_to_joints(np.array([0.1, -0.1]))\nclose_gripper()"
    )
    chunk = motion.take_chunk()

    assert recovered.raised is False
    assert np.array_equal(chunk.actions[-1].data, np.array([0.1, -0.1, 0.0]))

    sandbox.execute("move_to_joints(np.array([0.2, 0.2]))")
    assert motion.has_actions() is True
    sandbox.reset()
    assert motion.has_actions() is False
    assert motion.cursor is None


def test_obs_access_resolves_callable_values(capx_stub: CapxStub) -> None:
    sandbox, _, _ = _sandbox(capx_stub)
    sandbox.set_observation(_observation(extrinsics=lambda: np.eye(4)))

    result = sandbox.execute(
        "import numpy as np\n"
        "pose = obs['extrinsics'] @ np.eye(4)\n"
        "assert pose.shape == (4, 4)\n"
        "assert obs.get('extrinsics').shape == (4, 4)\n"
        "assert obs.get('absent') is None"
    )

    assert result.raised is False, result.stderr
