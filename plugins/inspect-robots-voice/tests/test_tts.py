"""Verified model acquisition and the lazy Kokoro engine seam."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import inspect_robots_voice._tts as tts
from inspect_robots_voice._tts import KokoroEngine, resolve_model_files


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_explicit_paths_pass_through_without_cache_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "offline.onnx"
    voices = tmp_path / "offline.bin"
    model.write_bytes(b"model")
    voices.write_bytes(b"voices")
    monkeypatch.setattr(tts, "_fetch", lambda *_args: pytest.fail("unexpected download"))

    assert resolve_model_files(str(model), str(voices)) == (str(model), str(voices))


@pytest.mark.parametrize("kind", ["model", "voices"])
def test_missing_explicit_path_names_the_path(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = tmp_path / "present"
    present.write_bytes(b"present")
    missing = tmp_path / "missing"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import re

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        resolve_model_files(
            str(missing) if kind == "model" else str(present),
            str(missing) if kind == "voices" else str(present),
        )


def test_cache_hit_is_verified_and_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "inspect-robots-voice"
    cache.mkdir()
    model = cache / tts._KOKORO_FILENAME
    voices = cache / tts._VOICES_FILENAME
    model.write_bytes(b"cached model")
    voices.write_bytes(b"cached voices")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(tts, "_KOKORO_SHA256", _digest(model.read_bytes()))
    monkeypatch.setattr(tts, "_VOICES_SHA256", _digest(voices.read_bytes()))
    monkeypatch.setattr(tts, "_fetch", lambda *_args: pytest.fail("unexpected download"))

    assert resolve_model_files(None, None) == (str(model), str(voices))


def test_stale_cache_entry_is_discarded_and_redownloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "inspect-robots-voice"
    cache.mkdir()
    stale = cache / tts._KOKORO_FILENAME
    stale.write_bytes(b"stale model")
    fresh_payload = b"fresh model"

    def fake_fetch(url: str, destination: Path) -> None:
        del url
        destination.write_bytes(fresh_payload)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(tts, "_fetch", fake_fetch)
    monkeypatch.setattr(tts, "_KOKORO_SHA256", _digest(fresh_payload))

    model, _voices = resolve_model_files(None, str(_existing(tmp_path / "voices.bin")))

    assert Path(model) == stale
    assert Path(model).read_bytes() == fresh_payload
    assert "failed sha256 check; redownloading" in capsys.readouterr().err


def test_download_uses_part_files_then_atomically_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        tts._KOKORO_FILENAME: b"downloaded model",
        tts._VOICES_FILENAME: b"downloaded voices",
    }
    destinations: list[Path] = []

    def fake_fetch(url: str, destination: Path) -> None:
        del url
        destinations.append(destination)
        assert destination.name.endswith(".part")
        final_name = destination.name.removesuffix(".part")
        destination.write_bytes(payloads[final_name])

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(tts, "_fetch", fake_fetch)
    monkeypatch.setattr(tts, "_KOKORO_SHA256", _digest(payloads[tts._KOKORO_FILENAME]))
    monkeypatch.setattr(tts, "_VOICES_SHA256", _digest(payloads[tts._VOICES_FILENAME]))

    model, voices = resolve_model_files(None, None)

    assert Path(model).read_bytes() == payloads[tts._KOKORO_FILENAME]
    assert Path(voices).read_bytes() == payloads[tts._VOICES_FILENAME]
    assert len(destinations) == 2
    assert all(not destination.exists() for destination in destinations)


def test_sha256_mismatch_deletes_part_and_reports_both_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"wrong model"
    expected = _digest(b"expected model")
    actual = _digest(payload)
    part_paths: list[Path] = []

    def fake_fetch(url: str, destination: Path) -> None:
        del url
        part_paths.append(destination)
        destination.write_bytes(payload)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(tts, "_fetch", fake_fetch)
    monkeypatch.setattr(tts, "_KOKORO_SHA256", expected)

    with pytest.raises(RuntimeError) as exc_info:
        resolve_model_files(None, str(_existing(tmp_path / "voices.bin")))

    message = str(exc_info.value)
    assert expected in message
    assert actual in message
    assert len(part_paths) == 1
    assert not part_paths[0].exists()


def _existing(path: Path) -> Path:
    path.write_bytes(b"present")
    return path


def test_module_import_does_not_import_kokoro(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "kokoro_onnx":
            pytest.fail("kokoro_onnx imported at module load")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(tts)


def test_guarded_kokoro_import_has_speak_and_python_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "kokoro_onnx":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "kokoro_onnx", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(ImportError) as exc_info:
        KokoroEngine("model", "voices", voice="af_sarah", speed=1.0, lang="en-us")

    message = str(exc_info.value)
    assert "--speak needs kokoro-onnx" in message
    assert "does not yet support this Python" in message


def test_kokoro_engine_constructs_and_synthesizes_via_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeKokoro:
        def __init__(self, model: str, voices: str) -> None:
            calls.append(("init", model, voices))

        def create(self, text: str, **kwargs: object) -> tuple[list[float], int]:
            calls.append(("create", text, kwargs))
            return [0.25, -0.5], 24_000

    module = types.ModuleType("kokoro_onnx")
    module.__dict__["Kokoro"] = FakeKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)

    engine = KokoroEngine(
        "/models/kokoro.onnx",
        "/models/voices.bin",
        voice="bf_emma",
        speed=1.25,
        lang="en-gb",
    )
    samples, sample_rate = engine.synthesize("hello")

    assert calls == [
        ("init", "/models/kokoro.onnx", "/models/voices.bin"),
        (
            "create",
            "hello",
            {"voice": "bf_emma", "speed": 1.25, "lang": "en-gb"},
        ),
    ]
    assert samples.dtype == np.float32
    assert np.array_equal(samples, np.array([0.25, -0.5], dtype=np.float32))
    assert sample_rate == 24_000
