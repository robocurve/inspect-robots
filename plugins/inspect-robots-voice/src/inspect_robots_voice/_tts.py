"""Lazy Kokoro synthesis and verified model-file acquisition."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

_KOKORO_FILENAME = "kokoro-v1.0.onnx"
_VOICES_FILENAME = "voices-v1.0.bin"
_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_KOKORO_URL = f"{_RELEASE_BASE}/{_KOKORO_FILENAME}"
_VOICES_URL = f"{_RELEASE_BASE}/{_VOICES_FILENAME}"
_KOKORO_SHA256 = "PENDING_PIN"
_VOICES_SHA256 = "PENDING_PIN"


class TtsEngine(Protocol):
    """Synthesize text as float32 mono samples and their sample rate."""

    def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]: ...


class KokoroEngine:
    """Provide local Kokoro synthesis behind the plugin's engine seam."""

    def __init__(
        self,
        model: str,
        voices: str,
        *,
        voice: str,
        speed: float,
        lang: str,
    ) -> None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise ImportError(
                "--speak needs kokoro-onnx, which is not installed or does not yet support "
                "this Python"
            ) from exc
        self._kokoro: Any = Kokoro(model, voices)
        self._voice = voice
        self._speed = speed
        self._lang = lang

    def synthesize(self, text: str) -> tuple[npt.NDArray[np.float32], int]:
        """Return one Kokoro utterance as normalized float32 samples and a sample rate."""
        samples, sample_rate = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)


def _fetch(url: str, destination: Path) -> None:
    print(f"speaker: downloading {destination.name} from {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected}, actual {actual}")


def _resolve_file(
    override: str | None,
    cache_dir: Path,
    filename: str,
    url: str,
    expected: str,
) -> str:
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        return str(path)

    path = cache_dir / filename
    if path.is_file():
        _verify(path, expected)
        return str(path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        _fetch(url, part)
        _verify(part, expected)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return str(path)


def resolve_model_files(model: str | None, voices: str | None) -> tuple[str, str]:
    """Resolve explicit model paths or acquire verified release files in the user cache."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = cache_home / "inspect-robots-voice"
    return (
        _resolve_file(model, cache_dir, _KOKORO_FILENAME, _KOKORO_URL, _KOKORO_SHA256),
        _resolve_file(voices, cache_dir, _VOICES_FILENAME, _VOICES_URL, _VOICES_SHA256),
    )
