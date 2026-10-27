"""The ``Policy`` adapter that drives base OpenVLA (``openvla/openvla-7b``).

This is the "brain" half of an Inspect Robots eval, backed by OpenVLA's
7-D end-effector action head. It conforms to :class:`inspect_robots.Policy`
(the runtime-checkable protocol), so once installed it pairs with any
compatible :class:`inspect_robots.Embodiment`.

Action convention (published through ``PolicyInfo.action_space`` so
``check_compatibility`` can verify it):

======  ===========================================================
dim     meaning
======  ===========================================================
0..2    end-effector position delta (metres, base frame)
3..5    end-effector rotation delta, Euler XYZ (radians)
6       gripper, binary: ``+1`` closed / ``-1`` open
======  ===========================================================

Upstream OpenVLA emits the RLDS gripper convention (``0`` closed, ``1`` open);
this adapter flips it once, at the boundary, so downstream embodiments can use
the robosuite-style ``±1`` value directly.

Laziness: constructing the adapter and reading ``.info`` never import torch,
touch the network, or read the checkpoint — so ``inspect-robots list policies``
and fail-fast compatibility checks work before the 7B weights exist locally.
Weights load on the first ``act()``. ``model`` / ``processor`` may be injected
to test the adapter without downloading anything.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    CameraSpec,
    Observation,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.errors import PolicyError
from inspect_robots_openvla._images import prepare_image

_ACTION_DIM = 7
_DIM_LABELS = ("dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper")
_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


class OpenVLAPolicy:
    """An Inspect Robots ``Policy`` backed by base OpenVLA."""

    def __init__(
        self,
        *,
        model_path: str = "openvla/openvla-7b",
        unnorm_key: str | None = None,
        camera: str = "agentview",
        camera_height: int = 224,
        camera_width: int = 224,
        center_crop: bool = False,
        name: str = "openvla_local",
        control_hz: float | None = None,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be provided together")
        if not camera.strip():
            raise ValueError("camera must be a non-empty observation key")
        if camera_height <= 0 or camera_width <= 0:
            raise ValueError("camera_height and camera_width must be > 0")

        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.camera = camera
        self.center_crop = center_crop
        self._dtype_name = torch_dtype

        self.info = PolicyInfo(
            name=name,
            action_space=Box(
                shape=(_ACTION_DIM,),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pose",
                    rotation_repr="euler_xyz",
                    gripper="binary",
                    frame="base",
                    dim_labels=_DIM_LABELS,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(camera, camera_height, camera_width),)
            ),
            control_hz=control_hz,
            checkpoint=model_path,
        )
        # Base OpenVLA predicts one action per forward pass; chunking (if any) is
        # the runner's job, not the adapter's.
        self.config = PolicyConfig(action_horizon=1)

        self._device = device
        self._torch_device: str | None = device if model is not None else None
        self._dtype: Any | None = None
        self._model = model
        self._processor = processor
        self._instruction: str | None = None
        self._closed = False

    # ------------------------------------------------------------------ #
    # Policy protocol
    # ------------------------------------------------------------------ #

    def reset(self, scene: Scene) -> None:
        """Bind the scene's instruction; the policy itself is stateless."""
        self._require_open()
        self._instruction = scene.instruction

    def act(self, observation: Observation) -> ActionChunk:
        """One forward pass: ``Observation`` in, a one-action chunk out."""
        self._require_open()
        if self._instruction is None:
            raise PolicyError(
                f"{self.info.name!r}: act() called before reset(); call reset(scene) first"
            )

        image = observation.images.get(self.camera)
        if image is None:
            raise KeyError(
                f"observation has no camera {self.camera!r}; available cameras: "
                f"{sorted(observation.images)}. Adjust the camera= keyword, "
                "e.g. -P camera=base_rgb"
            )

        processor, model = self._ensure_loaded()
        # ``Observation.instruction`` is the changing-goal channel; the scene
        # instruction bound by reset() is the fallback for datasets that set it
        # once.
        instruction = observation.instruction or self._instruction
        prompt = _PROMPT.format(instruction=instruction.lower())
        inputs = processor(prompt, prepare_image(image, center_crop=self.center_crop))
        if self._torch_device is not None:
            inputs = inputs.to(self._torch_device, dtype=self._dtype)

        kwargs: dict[str, Any] = {"do_sample": False}
        if self.unnorm_key is not None:
            kwargs["unnorm_key"] = self.unnorm_key

        start = time.perf_counter()
        raw = model.predict_action(**inputs, **kwargs)
        latency_s = time.perf_counter() - start

        return ActionChunk(
            actions=[Action(data=self._normalize_action(raw))],
            control_hz=self.info.control_hz,
            inference_latency_s=latency_s,
        )

    def close(self) -> None:
        """Drop the model references; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._processor = None

    def __enter__(self) -> OpenVLAPolicy:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OpenVLAPolicy is closed")

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load the processor/model on first use; never at construction time."""
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        return self._load_from_hub()

    def _load_from_hub(self) -> tuple[Any, Any]:  # pragma: no cover - needs GPU + weights
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise PolicyError(
                "OpenVLAPolicy needs the 'openvla' extra (torch + transformers): "
                "pip install 'inspect-robots-openvla[openvla]'"
            ) from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, self._dtype_name)
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        self._torch_device = device
        self._dtype = dtype
        self._processor = processor
        self._model = model
        return processor, model

    def _normalize_action(self, raw: Any) -> npt.NDArray[np.float32]:
        """Flatten to ``(7,)`` float32 and publish the binary gripper convention."""
        action = np.asarray(raw, dtype=np.float32).reshape(-1)
        if action.size == 0:
            raise PolicyError(f"{self.info.name!r} returned an empty action array")
        if action.size != _ACTION_DIM:
            raise PolicyError(
                f"{self.info.name!r} returned a {action.size}-D action; this adapter speaks "
                f"the {_ACTION_DIM}-D base-OpenVLA layout "
                "[dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]. "
                "OpenVLA-OFT has a different action space and needs its own adapter."
            )
        if not np.isfinite(action).all():
            raise PolicyError(
                f"{self.info.name!r} returned a non-finite action {action.tolist()}; "
                "refusing to threshold it into a gripper command"
            )
        # RLDS gripper (0 = closed, 1 = open) -> binary (+1 = closed, -1 = open).
        action[-1] = 1.0 if float(action[-1]) < 0.5 else -1.0
        return action
