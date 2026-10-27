"""Duck-typed stand-ins for the Hugging Face processor and the OpenVLA model.

Torch is deliberately absent: the adapter only relies on ``processor(...)``,
``.to(device, dtype)`` and ``model.predict_action(**inputs)``, so these three
shapes are all a CPU-only test needs to pin down.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class StubInputs(dict[str, Any]):
    """Processor output: a mapping that also supports ``.to(device, dtype)``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.to_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def to(self, *args: Any, **kwargs: Any) -> StubInputs:
        self.to_calls.append((args, kwargs))
        return self


class StubProcessor:
    """Records ``(prompt, image)`` calls and returns fake batched inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.inputs: list[StubInputs] = []

    def __call__(self, prompt: str, image: Any) -> StubInputs:
        self.calls.append((prompt, image))
        inputs = StubInputs(
            input_ids=[[0]],
            pixel_values=np.zeros((1, 3, 224, 224), dtype=np.float32),
        )
        self.inputs.append(inputs)
        return inputs


class StubOpenVLAModel:
    """Returns a fixed action vector and records every ``predict_action`` kwarg."""

    def __init__(self, action: Any = None) -> None:
        self.action = np.asarray(
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0] if action is None else action,
            dtype=np.float32,
        )
        self.calls: list[dict[str, Any]] = []

    def predict_action(self, **kwargs: Any) -> npt.NDArray[np.float32]:
        self.calls.append(kwargs)
        return self.action.copy()
