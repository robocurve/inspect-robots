"""The vlm grader fails fast on configuration and degrades after the rollout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots.errors import ConfigError
from inspect_robots.frames import FrameStore
from inspect_robots.grader import Grader, vlm_grader
from inspect_robots.registry import resolve
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult


def _frame(fill: int) -> npt.NDArray[np.uint8]:
    return np.full((2, 2, 3), fill, dtype=np.uint8)


def _framed_record(
    *,
    initial: dict[str, npt.NDArray[np.uint8]] | None = None,
    final: dict[str, npt.NDArray[np.uint8]] | None = None,
) -> TrialRecord:
    """One-step record with inline first/last frames (final defaults to one camera)."""
    steps = [
        StepRecord(
            t=0,
            observation=Observation(images=initial or {}),
            action=Action(data=np.zeros(2)),
            result=StepResult(
                observation=Observation(images={"cam": _frame(9)} if final is None else final)
            ),
        )
    ]
    return TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)


class _CapturePost:
    """An HttpPost double that records requests and returns a canned reply."""

    def __init__(self, reply: str = "looks done\nGRADE: success") -> None:
        content = json.dumps({"choices": [{"message": {"content": reply}}]})
        self.response: tuple[int, bytes] = (200, content.encode())
        self.requests: list[dict[str, object]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        self.requests.append({"url": url, "headers": headers, "body": json.loads(body)})
        return self.response

    def message_parts(self) -> list[dict[str, object]]:
        body = self.requests[-1]["body"]
        assert isinstance(body, dict)
        parts = body["messages"][0]["content"]
        assert isinstance(parts, list)
        return parts


def _vlm(post: _CapturePost, monkeypatch: pytest.MonkeyPatch, **kwargs: str | int) -> Grader:
    monkeypatch.setenv("VLM_TEST_KEY", "secret")
    return vlm_grader("judge-model", api_key_env="VLM_TEST_KEY", http_post=post, **kwargs)  # type: ignore[arg-type]


def _scene() -> Scene:
    return Scene(id="s", instruction="stack the cubes")


def _labels(parts: list[dict[str, object]]) -> list[object]:
    return [p["text"] for p in parts if p["type"] == "text"][1:]


def test_vlm_grader_registry_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_TEST_KEY", "secret")
    grader = resolve("grader", "vlm", model="m", api_key_env="VLM_TEST_KEY")
    assert isinstance(grader, Grader)
    assert grader.name == "vlm"


def test_vlm_grader_requires_model_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match=r"model id.\nfix:"):
        vlm_grader("")
    monkeypatch.delenv("VLM_MISSING_KEY", raising=False)
    with pytest.raises(ConfigError, match=r"VLM_MISSING_KEY.\nfix:"):
        vlm_grader("m", api_key_env="VLM_MISSING_KEY")


def test_vlm_grader_rejects_a_useless_camera_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_TEST_KEY", "secret")
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ConfigError, match=r"max_cameras must be a positive integer"):
            vlm_grader("m", api_key_env="VLM_TEST_KEY", max_cameras=bad)  # type: ignore[arg-type]


def test_vlm_grader_rubric_sources_are_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLM_TEST_KEY", "secret")
    with pytest.raises(ConfigError, match=r"mutually exclusive.\nfix:"):
        vlm_grader("m", rubric="a", rubric_file="b", api_key_env="VLM_TEST_KEY")
    with pytest.raises(ConfigError, match=r"cannot read rubric_file"):
        vlm_grader("m", rubric_file=str(tmp_path / "missing.md"), api_key_env="VLM_TEST_KEY")


def test_vlm_grader_reads_rubric_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rubric_path = tmp_path / "rubric.md"
    rubric_path.write_text("cube must sit on cube", encoding="utf-8")
    post = _CapturePost()
    grader = _vlm(post, monkeypatch, rubric_file=str(rubric_path))
    rubric_path.unlink()

    grader.grade(_framed_record(), _scene())

    text = post.message_parts()[0]["text"]
    assert isinstance(text, str)
    assert "cube must sit on cube" in text


def test_vlm_grader_prompt_covers_instruction_default_rubric_and_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _CapturePost()
    grader = _vlm(post, monkeypatch)
    record = _framed_record(initial={"cam": _frame(1)}, final={"cam": _frame(9)})

    grader.grade(record, _scene())

    parts = post.message_parts()
    text = parts[0]["text"]
    assert isinstance(text, str)
    assert "stack the cubes" in text
    assert "Grade success if the system completed the task instruction" in text
    assert "initial frames (before the robot acted) and the final" in text
    assert _labels(parts) == ["initial frame, camera 'cam'", "final frame, camera 'cam'"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 2
    for image in images:
        url = image["image_url"]
        assert isinstance(url, dict)
        assert str(url["url"]).startswith("data:image/png;base64,")
    assert record.operator_judgement == "success"
    assert record.operator_note == "looks done\nGRADE: success"
    event = record.events[-1]
    assert event.kind == "operator"
    assert event.data["source"] == "vlm"
    assert event.data["verdict"] == "success"


def test_vlm_grader_prefers_a_scene_metadata_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _CapturePost()
    grader = _vlm(post, monkeypatch, rubric="run-level rubric")
    scene = Scene(id="s", instruction="stack", metadata={"rubric": "generated: cube on cube"})

    grader.grade(_framed_record(), scene)

    text = post.message_parts()[0]["text"]
    assert isinstance(text, str)
    assert "generated: cube on cube" in text
    assert "run-level rubric" not in text


def test_vlm_grader_ignores_blank_or_nonstring_scene_rubrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in ("   ", 7):
        post = _CapturePost()
        grader = _vlm(post, monkeypatch, rubric="run-level rubric")
        scene = Scene(id="s", instruction="stack", metadata={"rubric": bad})

        grader.grade(_framed_record(), scene)

        text = post.message_parts()[0]["text"]
        assert isinstance(text, str)
        assert "run-level rubric" in text


def test_vlm_grader_without_initial_frames_reworks_the_frames_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _CapturePost()
    grader = _vlm(post, monkeypatch)

    grader.grade(_framed_record(), _scene())

    parts = post.message_parts()
    text = parts[0]["text"]
    assert isinstance(text, str)
    assert "You will see the final frames (after the trial ended)." in text
    assert "initial frames" not in text
    assert _labels(parts) == ["final frame, camera 'cam'"]


def test_vlm_grader_loads_frame_refs_sorted_and_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FrameStore(str(tmp_path / "frames"))
    refs = {
        camera: store.put("trial", 0, camera, _frame(i))
        for i, camera in enumerate(["wrist", "b_cam", "a_cam", "c_cam", "z_cam"])
    }
    steps = [
        StepRecord(
            t=0,
            observation=Observation(),
            action=Action(data=np.zeros(2)),
            result=StepResult(observation=Observation()),
            image_refs=refs,
            result_image_refs={"cam": store.put("trial", 1, "cam", _frame(7))},
        )
    ]
    record = TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)
    post = _CapturePost()
    grader = _vlm(post, monkeypatch, max_cameras=2)

    grader.grade(record, _scene())

    assert _labels(post.message_parts()) == [
        "initial frame, camera 'a_cam'",
        "initial frame, camera 'b_cam'",
        "final frame, camera 'cam'",
    ]
    assert record.operator_judgement == "success"


def test_vlm_grader_never_loads_frames_beyond_the_camera_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FrameStore(str(tmp_path / "frames"))
    kept = store.put("trial", 1, "a_cam", _frame(7))
    corrupt = store.put("trial", 1, "z_cam", _frame(8))
    Path(corrupt.path).unlink()
    steps = [
        StepRecord(
            t=0,
            observation=Observation(),
            action=Action(data=np.zeros(2)),
            result=StepResult(observation=Observation()),
            result_image_refs={"a_cam": kept, "z_cam": corrupt},
        )
    ]
    record = TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)
    post = _CapturePost()

    _vlm(post, monkeypatch, max_cameras=1).grade(record, _scene())

    assert _labels(post.message_parts()) == ["final frame, camera 'a_cam'"]
    assert record.operator_judgement == "success"


def test_vlm_grader_adopts_existing_judgement_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _CapturePost()
    grader = _vlm(post, monkeypatch)
    record = _framed_record()
    record.operator_judgement = "y"

    grader.grade(record, _scene())

    assert post.requests == []
    assert record.operator_judgement == "y"


def test_vlm_grader_adopts_definitive_embodiment_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for reason in ("success", "failure"):
        post = _CapturePost()
        grader = _vlm(post, monkeypatch)
        record = _framed_record()
        record.terminated = True
        record.termination_reason = reason

        grader.grade(record, _scene())

        assert post.requests == []
        assert record.operator_judgement == reason
        assert record.events[-1].data["source"] == "embodiment"


def test_vlm_grader_degrades_without_final_frames(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    post = _CapturePost()
    grader = _vlm(post, monkeypatch)
    empty = TrialRecord(scene_id="s", epoch=0, seed=0)
    framed_but_bare = _framed_record(final={})

    grader.grade(empty, _scene())
    grader.grade(framed_but_bare, _scene())

    assert post.requests == []
    assert empty.operator_judgement is None
    assert framed_but_bare.operator_judgement is None
    err = capsys.readouterr().err
    assert err.count("vlm grader: no final frames recorded; trial left ungraded") == 2


def test_vlm_grader_degrades_on_frame_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FrameStore(str(tmp_path / "frames"))
    ref = store.put("trial", 0, "cam", _frame(1))
    Path(ref.path).unlink()
    steps = [
        StepRecord(
            t=0,
            observation=Observation(),
            action=Action(data=np.zeros(2)),
            result=StepResult(observation=Observation()),
            result_image_refs={"cam": ref},
        )
    ]
    record = TrialRecord(scene_id="s", epoch=0, seed=0, steps=steps)
    post = _CapturePost()
    grader = _vlm(post, monkeypatch)

    grader.grade(record, _scene())

    assert post.requests == []
    assert record.operator_judgement is None
    assert "trial left ungraded" in capsys.readouterr().err


def test_vlm_grader_parses_verdicts_leniently(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [
        ("GRADE: failure", "failure"),
        ("**GRADE: Success**", "success"),
        ("grade: SUCCESS", "success"),
        ("GRADE: success\nwait, no.\nGRADE: failure", "failure"),
    ]
    for reply, expected in cases:
        post = _CapturePost(reply=reply)
        record = _framed_record()

        _vlm(post, monkeypatch).grade(record, _scene())

        assert record.operator_judgement == expected, reply


def test_vlm_grader_degrades_on_unparseable_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    post = _CapturePost(reply="the trial went fine")
    record = _framed_record()

    _vlm(post, monkeypatch).grade(record, _scene())

    assert record.operator_judgement is None
    err = capsys.readouterr().err
    assert "no GRADE line in the reply" in err
    assert "the trial went fine" in err


def test_vlm_grader_caps_the_note(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _CapturePost(reply=("x" * 3000) + "\nGRADE: success")
    record = _framed_record()

    _vlm(post, monkeypatch).grade(record, _scene())

    assert record.operator_judgement == "success"
    assert record.operator_note is not None
    assert len(record.operator_note) == 2000


def test_vlm_grader_degrades_on_transport_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def http_boom(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        return 500, b"upstream exploded"

    def timeout(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        raise TimeoutError("read timed out")

    monkeypatch.setenv("VLM_TEST_KEY", "secret")
    failures = (
        (http_boom, "grading request failed with HTTP 500"),
        (timeout, "read timed out"),
    )
    for post_fn, marker in failures:
        grader = vlm_grader("m", api_key_env="VLM_TEST_KEY", http_post=post_fn)
        record = _framed_record()

        grader.grade(record, _scene())

        assert record.operator_judgement is None
        assert marker in capsys.readouterr().err
