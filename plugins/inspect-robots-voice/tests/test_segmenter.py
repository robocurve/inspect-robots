"""Adaptive energy-gate behavior without audio hardware."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots_voice._segmenter import EnergyGate


def _block(value: float, *, samples: int = 1_600) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


def test_silence_yields_no_utterance() -> None:
    gate = EnergyGate()

    assert all(gate.push(_block(0.0)) is None for _ in range(20))
    assert not gate.is_open


def test_speech_burst_includes_pre_roll_and_closes_after_hangover() -> None:
    gate = EnergyGate()
    for _ in range(3):
        assert gate.push(_block(0.0)) is None
    assert gate.push(_block(0.5)) is None
    assert gate.is_open
    assert gate.push(_block(0.5)) is None
    for _ in range(6):
        assert gate.push(_block(0.0)) is None

    utterance = gate.push(_block(0.0))

    assert utterance is not None
    assert len(utterance) == 19_200
    assert np.all(utterance[:4_800] == 0.0)
    assert np.all(utterance[4_800:8_000] == 0.5)
    assert not gate.is_open


def test_burst_shorter_than_minimum_open_is_discarded() -> None:
    gate = EnergyGate()

    assert gate.push(_block(0.5, samples=800)) is None
    assert gate.push(_block(0.0, samples=800)) is None
    assert gate.push(_block(0.0)) is None
    assert not gate.is_open


def test_hangover_does_not_close_early() -> None:
    gate = EnergyGate()
    assert gate.push(_block(0.5)) is None
    for _ in range(6):
        assert gate.push(_block(0.0)) is None
        assert gate.is_open

    assert gate.push(_block(0.0)) is not None
    assert not gate.is_open


def test_thirty_second_cap_force_closes_and_gate_continues() -> None:
    gate = EnergyGate(sample_rate=100)
    short = 10
    for _ in range(3):
        gate.push(_block(0.0, samples=short))
    assert gate.push(_block(0.5, samples=short)) is None

    capped = None
    for _ in range(400):
        capped = gate.push(_block(0.5, samples=short))
        if capped is not None:
            break

    assert capped is not None
    assert len(capped) == 3_000
    assert not gate.is_open

    for _ in range(3):
        gate.push(_block(0.0, samples=short))
    gate.push(_block(0.5, samples=short))
    continued = None
    for _ in range(7):
        continued = gate.push(_block(0.0, samples=short))
    assert continued is not None


def test_noise_floor_adapts_on_silence_and_freezes_during_speech() -> None:
    gate = EnergyGate(ema_alpha=0.5)
    initial = gate.noise_floor
    for _ in range(4):
        gate.push(_block(0.02))
    adapted = gate.noise_floor

    assert adapted > initial
    gate.push(_block(0.5))
    assert gate.is_open
    assert gate.noise_floor == adapted
    gate.push(_block(0.5))
    assert gate.noise_floor == adapted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"open_ratio": 1.0},
        {"hangover_s": 0.0},
        {"ema_alpha": 0.0},
        {"ema_alpha": 1.1},
    ],
)
def test_invalid_gate_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EnergyGate(**kwargs)


def test_empty_blocks_and_reset_clear_all_state() -> None:
    gate = EnergyGate()
    assert gate.push(np.array([], dtype=np.float32)) is None
    gate.push(_block(0.5))
    assert gate.is_open

    gate.reset()

    assert not gate.is_open
    assert gate.noise_floor == 1e-2
