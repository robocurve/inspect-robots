"""Pure NumPy adaptive energy-gate segmentation for microphone audio."""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]


class EnergyGate:
    """Turn fixed-rate mono audio blocks into bounded speech utterances.

    Consecutive energy above the adaptive noise threshold opens an utterance after
    ``min_open_s``. An open utterance closes after ``hangover_s`` below threshold or at the
    ``max_utterance_s`` hard cap. Noise adapts only while the gate is closed and below threshold.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        open_ratio: float = 3.0,
        hangover_s: float = 0.7,
        pre_roll_s: float = 0.3,
        min_open_s: float = 0.1,
        max_utterance_s: float = 30.0,
        ema_alpha: float = 0.05,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if open_ratio <= 1:
            raise ValueError("open_ratio must be greater than one")
        if min(hangover_s, pre_roll_s, min_open_s, max_utterance_s) <= 0:
            raise ValueError("all duration settings must be positive")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.sample_rate = sample_rate
        self.open_ratio = open_ratio
        self.hangover_samples = round(hangover_s * sample_rate)
        self.pre_roll_samples = round(pre_roll_s * sample_rate)
        self.min_open_samples = round(min_open_s * sample_rate)
        self.max_utterance_samples = round(max_utterance_s * sample_rate)
        self.ema_alpha = ema_alpha
        self._initial_noise_floor = 1e-2
        self._noise_floor = self._initial_noise_floor
        self._pre_roll: deque[AudioArray] = deque()
        self._pre_roll_size = 0
        self._candidate: list[AudioArray] = []
        self._candidate_size = 0
        self._utterance: list[AudioArray] | None = None
        self._utterance_size = 0
        self._below_size = 0

    @property
    def is_open(self) -> bool:
        """Return whether a speech utterance is currently accumulating."""
        return self._utterance is not None

    @property
    def noise_floor(self) -> float:
        """Return the current RMS estimate used to derive the opening threshold."""
        return self._noise_floor

    def reset(self) -> None:
        """Discard all audio state and restore the initial noise estimate."""
        self._noise_floor = self._initial_noise_floor
        self._pre_roll.clear()
        self._pre_roll_size = 0
        self._candidate.clear()
        self._candidate_size = 0
        self._utterance = None
        self._utterance_size = 0
        self._below_size = 0

    def push(self, block: npt.ArrayLike) -> AudioArray | None:
        """Consume one mono block and return an utterance only when the gate closes."""
        audio = np.asarray(block, dtype=np.float32).reshape(-1).copy()
        if audio.size == 0:
            return None
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        threshold = self._noise_floor * self.open_ratio
        if self._utterance is None:
            return self._push_closed(audio, rms, threshold)
        return self._push_open(audio, rms, threshold)

    def _push_closed(self, audio: AudioArray, rms: float, threshold: float) -> AudioArray | None:
        if rms > threshold:
            self._candidate.append(audio)
            self._candidate_size += len(audio)
            if self._candidate_size >= self.min_open_samples:
                self._utterance = [*self._pre_roll, *self._candidate]
                self._utterance_size = self._pre_roll_size + self._candidate_size
                self._below_size = 0
                self._pre_roll.clear()
                self._pre_roll_size = 0
                self._candidate.clear()
                self._candidate_size = 0
                if self._utterance_size >= self.max_utterance_samples:
                    return self._finish()
            return None

        for candidate_block in self._candidate:
            self._append_pre_roll(candidate_block)
        self._candidate.clear()
        self._candidate_size = 0
        self._append_pre_roll(audio)
        self._noise_floor = (1.0 - self.ema_alpha) * self._noise_floor + self.ema_alpha * rms
        return None

    def _push_open(self, audio: AudioArray, rms: float, threshold: float) -> AudioArray | None:
        assert self._utterance is not None
        remaining = self.max_utterance_samples - self._utterance_size
        accepted = audio[:remaining]
        self._utterance.append(accepted)
        self._utterance_size += len(accepted)
        if self._utterance_size >= self.max_utterance_samples:
            return self._finish()
        if rms > threshold:
            self._below_size = 0
        else:
            self._below_size += len(audio)
        if self._below_size >= self.hangover_samples:
            return self._finish()
        return None

    def _append_pre_roll(self, audio: AudioArray) -> None:
        self._pre_roll.append(audio)
        self._pre_roll_size += len(audio)
        while self._pre_roll_size > self.pre_roll_samples and self._pre_roll:
            excess = self._pre_roll_size - self.pre_roll_samples
            first = self._pre_roll[0]
            if len(first) <= excess:
                self._pre_roll.popleft()
                self._pre_roll_size -= len(first)
            else:
                self._pre_roll[0] = first[excess:]
                self._pre_roll_size -= excess

    def _finish(self) -> AudioArray:
        assert self._utterance is not None
        result = np.concatenate(self._utterance)[: self.max_utterance_samples].astype(
            np.float32, copy=False
        )
        self._utterance = None
        self._utterance_size = 0
        self._below_size = 0
        self._candidate_size = 0
        return result
