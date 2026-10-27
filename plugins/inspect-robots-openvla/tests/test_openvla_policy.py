"""Tests for the OpenVLA policy adapter.

These run anywhere — no GPU, no checkpoint, no torch. The stub processor/model
in ``_stub_model.py`` reproduce the exact three call shapes the adapter uses.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pytest
from _stub_model import StubOpenVLAModel, StubProcessor
from PIL import Image

from inspect_robots import (
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentInfo,
    ObservationSpace,
    Policy,
    Scene,
)
from inspect_robots.compat import check_compatibility
from inspect_robots.errors import CompatibilityError, PolicyError
from inspect_robots.registry import registered, resolve
from inspect_robots.types import Observation
from inspect_robots_openvla import OpenVLAPolicy, openvla_policy
from inspect_robots_openvla._images import prepare_image

_SCENE = Scene(id="s0", instruction="Pick up the cube")
_CAMERA = "agentview"


def _observation(
    *, camera: str = _CAMERA, instruction: str | None = _SCENE.instruction
) -> Observation:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return Observation(images={camera: image}, instruction=instruction)


def _policy(*, model: StubOpenVLAModel, processor: StubProcessor, **kwargs: Any) -> OpenVLAPolicy:
    return OpenVLAPolicy(model=model, processor=processor, **kwargs)


# --------------------------------------------------------------------------- #
# Static contract: offline construction + registry discovery
# --------------------------------------------------------------------------- #


def test_construction_never_imports_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # `None` in sys.modules makes `import torch` raise; construction must not do it.
    monkeypatch.setitem(sys.modules, "torch", None)
    policy = OpenVLAPolicy()

    assert policy.info.name == "openvla_local"
    assert policy.info.checkpoint == "openvla/openvla-7b"
    assert policy.info.action_space.shape == (7,)
    assert policy.info.observation_space.camera_names == frozenset({_CAMERA})


def test_declares_the_eef_action_contract() -> None:
    semantics = OpenVLAPolicy().info.action_space.semantics

    assert semantics is not None
    assert semantics.control_mode == "eef_delta_pose"
    assert semantics.rotation_repr == "euler_xyz"
    assert semantics.gripper == "binary"
    assert semantics.frame == "base"
    assert semantics.dim_labels == ("dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper")


def test_satisfies_the_runtime_checkable_policy_protocol() -> None:
    assert isinstance(OpenVLAPolicy(), Policy)


def test_requires_model_and_processor_together(stub_model: StubOpenVLAModel) -> None:
    with pytest.raises(ValueError, match="together"):
        OpenVLAPolicy(model=stub_model)


def test_rejects_an_empty_camera_name() -> None:
    with pytest.raises(ValueError, match="camera must be"):
        OpenVLAPolicy(camera="  ")


def test_rejects_a_non_positive_camera_size() -> None:
    with pytest.raises(ValueError, match="camera_height"):
        OpenVLAPolicy(camera_height=0)


def test_registry_discovers_the_entry_point(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    assert "openvla_local" in registered("policy")

    resolved = resolve("policy", "openvla_local", model=stub_model, processor=stub_processor)
    assert isinstance(resolved, OpenVLAPolicy)


def test_factory_returns_the_adapter(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = openvla_policy(model=stub_model, processor=stub_processor)
    assert isinstance(policy, OpenVLAPolicy)


# --------------------------------------------------------------------------- #
# act(): prompt, image, action normalisation
# --------------------------------------------------------------------------- #


def test_act_before_reset_raises(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)

    with pytest.raises(PolicyError, match="before reset"):
        policy.act(_observation())


def test_act_returns_one_action_chunk_with_closed_gripper(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    stub_model.action = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.0], dtype=np.float32)
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    chunk = policy.act(_observation())

    assert len(chunk.actions) == 1
    data = np.asarray(chunk.actions[0].data)
    assert data.shape == (7,)
    assert data.dtype == np.float32
    assert data[-1] == pytest.approx(1.0)  # RLDS 0 (closed) -> +1
    assert chunk.control_hz is None
    assert chunk.inference_latency_s is not None and chunk.inference_latency_s >= 0.0


def test_act_lowercases_the_instruction_into_the_openvla_prompt(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    policy.act(_observation())

    prompt, image = stub_processor.calls[0]
    assert prompt == "In: What action should the robot take to pick up the cube?\nOut:"
    assert isinstance(image, Image.Image)
    assert image.size == (640, 480)  # no crop: resolution untouched


def test_open_gripper_maps_to_minus_one(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    stub_model.action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    chunk = policy.act(_observation())

    assert np.asarray(chunk.actions[0].data)[-1] == pytest.approx(-1.0)


def test_non_gripper_dims_pass_through_untouched(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    raw = np.array([0.11, -0.22, 0.33, 0.04, -0.05, 0.06, 0.9], dtype=np.float32)
    stub_model.action = raw
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    data = np.asarray(policy.act(_observation()).actions[0].data)

    np.testing.assert_array_equal(data[:6], raw[:6])


def test_reset_rebinds_the_instruction_to_the_new_scene(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)
    policy.act(_observation(instruction=None))

    policy.reset(Scene(id="s1", instruction="Open the drawer"))
    policy.act(_observation(instruction=None))

    assert "pick up the cube" in stub_processor.calls[0][0]
    assert "open the drawer" in stub_processor.calls[1][0]


def test_observation_instruction_overrides_the_scene_instruction(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    policy.act(_observation(instruction="Open the drawer"))
    policy.act(_observation(instruction="Stack the blocks"))

    # The goal moved twice without reset() in between; the model must follow
    # the observation, not the stale scene instruction.
    assert "open the drawer" in stub_processor.calls[0][0]
    assert "stack the blocks" in stub_processor.calls[1][0]


def test_three_consecutive_acts_are_independent(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    chunks = [policy.act(_observation()) for _ in range(3)]

    # 每一步都重新观测并重新推理; 不复用上一步的输入或输出
    assert len(stub_processor.calls) == 3
    assert len(stub_model.calls) == 3
    assert {prompt for prompt, _ in stub_processor.calls} == {
        "In: What action should the robot take to pick up the cube?\nOut:"
    }

    datas = [np.asarray(chunk.actions[0].data) for chunk in chunks]
    assert all(len(chunk.actions) == 1 for chunk in chunks)
    assert len({id(data) for data in datas}) == 3  # 每步返回独立数组
    np.testing.assert_array_equal(datas[0], datas[2])

    datas[0][-1] = -1.0  # 改写第 1 步不应波及第 3 步
    assert datas[2][-1] == pytest.approx(1.0)


def test_unnorm_key_is_forwarded_and_sampling_is_greedy(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor, unnorm_key="bridge_orig")
    policy.reset(_SCENE)

    policy.act(_observation())

    assert stub_model.calls[0]["unnorm_key"] == "bridge_orig"
    assert stub_model.calls[0]["do_sample"] is False


def test_unnorm_key_is_omitted_when_unset(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    policy.act(_observation())

    assert "unnorm_key" not in stub_model.calls[0]


def test_missing_camera_names_the_available_ones(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    with pytest.raises(KeyError, match="available cameras"):
        policy.act(_observation(camera="wrist"))


def test_oft_sized_action_is_rejected_with_a_hint(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    stub_model.action = np.zeros(8, dtype=np.float32)
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    with pytest.raises(PolicyError, match="OpenVLA-OFT"):
        policy.act(_observation())


def test_empty_action_is_rejected(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    stub_model.action = np.zeros((0,), dtype=np.float32)
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    with pytest.raises(PolicyError, match="empty action"):
        policy.act(_observation())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_action_is_rejected(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor, bad: float
) -> None:
    stub_model.action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bad], dtype=np.float32)
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    # NaN must not be thresholded into a valid gripper command.
    with pytest.raises(PolicyError, match="non-finite"):
        policy.act(_observation())


def test_chunked_output_is_flattened(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    stub_model.action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]], dtype=np.float32)
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    data = np.asarray(policy.act(_observation()).actions[0].data)

    assert data.shape == (7,)


def test_injected_inputs_are_cast_to_the_requested_device(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor, device="cpu")
    policy.reset(_SCENE)

    policy.act(_observation())

    assert stub_processor.inputs[0].to_calls == [(("cpu",), {"dtype": None})]


def test_missing_openvla_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deterministic stand-in for "torch is not installed" on a CPU-only box.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    policy = OpenVLAPolicy()
    policy.reset(_SCENE)

    with pytest.raises(PolicyError, match="extra"):
        policy.act(_observation())


# --------------------------------------------------------------------------- #
# Image preprocessing (Pillow only)
# --------------------------------------------------------------------------- #


def test_center_crop_resamples_to_the_processor_size() -> None:
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)

    cropped = prepare_image(image, center_crop=True)

    assert cropped.size == (224, 224)
    assert not np.array_equal(np.asarray(cropped), image)


def test_center_crop_matches_the_upstream_resample_spec() -> None:
    # Reference: openvla/experiments/robot/openvla_utils.py:crop_and_resize --
    # a box at (1 - sqrt(scale)) / 2 offsets on both axes, cropped in one go and
    # resampled straight to 224 px (tf.image.crop_and_resize is bilinear).
    scale = 0.9
    side = math.sqrt(scale)
    for width, height in ((640, 480), (96, 64), (97, 65)):
        image = np.random.default_rng(width * height).integers(
            0, 255, (height, width, 3), dtype=np.uint8
        )
        keep_w = max(1, round(width * side))
        keep_h = max(1, round(height * side))
        left = (width - keep_w) // 2
        top = (height - keep_h) // 2
        reference = (
            Image.fromarray(image)
            .crop((left, top, left + keep_w, top + keep_h))
            .resize((224, 224), Image.Resampling.BILINEAR)
        )

        np.testing.assert_array_equal(
            np.asarray(prepare_image(image, center_crop=True, crop_scale=scale)),
            np.asarray(reference),
        )


def test_center_crop_keeps_sqrt_of_each_axis_on_non_square_frames() -> None:
    # openvla/experiments/robot/openvla_utils.py:crop_and_resize builds its box
    # from relative offsets (1 - sqrt(crop_scale)) / 2 per axis. Using 0.9 as a
    # linear factor instead would leave black borders on the result.
    scale = 0.9
    side = math.sqrt(scale)
    for width, height in ((640, 480), (96, 64), (97, 65)):
        keep_w = max(1, round(width * side))
        keep_h = max(1, round(height * side))
        left = (width - keep_w) // 2
        top = (height - keep_h) // 2
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[top : top + keep_h, left : left + keep_w] = 255

        cropped = np.asarray(prepare_image(image, center_crop=True, crop_scale=scale))

        assert cropped.shape == (224, 224, 3)
        # The white block fills the frame edge to edge, so both corners survive.
        assert cropped[0, 0, 0] > 200
        assert cropped[-1, -1, 0] > 200


def test_center_crop_rejects_out_of_range_scale() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="crop_scale"):
        prepare_image(image, center_crop=True, crop_scale=1.5)


def test_non_uint8_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="uint8"):
        prepare_image(np.zeros((8, 8, 3), dtype=np.float32))


def test_gray_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        prepare_image(np.zeros((8, 8), dtype=np.uint8))


def test_pil_inputs_are_converted_to_rgb() -> None:
    converted = prepare_image(Image.new("L", (32, 24), color=128))

    assert converted.mode == "RGB"
    assert converted.size == (32, 24)


# --------------------------------------------------------------------------- #
# Compatibility with an embodiment that shares the contract
# --------------------------------------------------------------------------- #


class _ArmEmbodiment:
    """Static info only; never stepped in these tests."""

    def __init__(self, semantics: ActionSemantics) -> None:
        self.info = EmbodimentInfo(
            name="stub-arm",
            action_space=Box(shape=(7,), semantics=semantics),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(_CAMERA, 480, 640),),
            ),
            is_simulated=True,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Any:  # pragma: no cover
        raise NotImplementedError

    def step(self, action: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass


def test_compatible_when_semantics_and_camera_match() -> None:
    report = check_compatibility(
        OpenVLAPolicy(),
        _ArmEmbodiment(
            ActionSemantics(
                control_mode="eef_delta_pose",
                rotation_repr="euler_xyz",
                gripper="binary",
                frame="base",
            )
        ),
    )

    assert report.ok, report.errors


def test_position_only_embodiment_fails_fast() -> None:
    report = check_compatibility(
        OpenVLAPolicy(),
        _ArmEmbodiment(
            ActionSemantics(
                control_mode="eef_delta_pos",
                rotation_repr="euler_xyz",
                gripper="binary",
                frame="base",
            )
        ),
    )

    assert not report.ok
    assert "control_mode" in {issue.code for issue in report.errors}
    with pytest.raises(CompatibilityError):
        report.raise_for_errors()


def test_rotation_mismatch_fails_fast() -> None:
    report = check_compatibility(
        OpenVLAPolicy(),
        _ArmEmbodiment(
            ActionSemantics(
                control_mode="eef_delta_pose",
                rotation_repr="axis_angle",
                gripper="binary",
                frame="base",
            )
        ),
    )

    assert not report.ok
    assert "rotation_repr" in {issue.code for issue in report.errors}
    with pytest.raises(CompatibilityError):
        report.raise_for_errors()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_close_is_idempotent_and_blocks_further_calls(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    policy = _policy(model=stub_model, processor=stub_processor)
    policy.reset(_SCENE)

    policy.close()
    policy.close()

    with pytest.raises(RuntimeError, match="closed"):
        policy.act(_observation())
    with pytest.raises(RuntimeError, match="closed"):
        policy.reset(_SCENE)


def test_context_manager_closes_on_exit(
    stub_model: StubOpenVLAModel, stub_processor: StubProcessor
) -> None:
    with _policy(model=stub_model, processor=stub_processor) as policy:
        policy.reset(_SCENE)

    with pytest.raises(RuntimeError, match="closed"):
        policy.act(_observation())
