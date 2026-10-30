from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pytest
from _dream_fakes import HIGH, HOME, STATE, FakeClient, FakeSession, FakeYam, observation, yam_info

from inspect_robots import ActionChunk, ActionSemantics, Box, Observation, Policy, Scene
from inspect_robots.compat import check_compatibility
from inspect_robots_dream import policy as policy_module
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession


def bound(
    session: FakeSession | None = None, *, control_hz: float = 20.0, **kwargs: Any
) -> DreamPolicy:
    policy = DreamPolicy(session=session or FakeSession(), **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        policy.bind(yam_info(control_hz=control_hz))
    return policy


def run(policy: DreamPolicy, steps: int, state: np.ndarray = HOME) -> list[ActionChunk]:
    return [policy.act(observation(state)) for _ in range(steps)]


def played(chunks: list[ActionChunk]) -> list[list[float]]:
    return [chunk.actions[0].data.tolist() for chunk in chunks]


def test_info_declares_the_bimanual_yam_contract() -> None:
    policy = DreamPolicy(model="dreamlabs/dream-1-beta-flash", rows_per_request=24)
    assert isinstance(policy, Policy)
    assert policy.info.name == "dream" and policy.info.checkpoint == "dreamlabs/dream-1-beta-flash"
    assert policy.info.action_space.shape == (14,) and policy.info.control_hz == 20.0
    assert policy.info.observation_space.camera_names == {"top_cam", "left_cam", "right_cam"}
    assert policy.info.observation_space.state_keys == {"joint_pos"}
    assert (policy.config.action_horizon, policy.config.replan_interval) == (1, None)
    assert check_compatibility(policy, FakeYam()).issues == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows_per_request": 0},
        {"rows_per_request": 49},
        {"rows_per_request": True},
        {"rows_per_request": 2.5},
        {"preroll_steps": -1},
        {"preroll_steps": True},
        {"dry_run": "yes"},
        {"model": ""},
        {"timeout_s": 0},
        {"record": "no"},
    ],
)
def test_constructor_rejects_bad_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DreamPolicy(**kwargs)


@pytest.mark.parametrize(
    "space",
    [
        Box(shape=(7,)),
        Box(shape=(14,)),
        Box(shape=(14,), semantics=ActionSemantics(control_mode="joint_delta")),
        Box(
            shape=(14,),
            semantics=ActionSemantics(control_mode="joint_pos", dim_labels=tuple("abcdefghijklmn")),
        ),
    ],
)
def test_bind_refuses_anything_but_the_bimanual_yam(space: Box) -> None:
    session = FakeSession()
    with pytest.raises(ValueError, match="bimanual YAM"):
        DreamPolicy(session=session).bind(yam_info(action_space=space))
    assert session.checked == []


def test_bind_adopts_the_rate_and_checks_dream_can_serve_the_model() -> None:
    session = FakeSession()
    policy = DreamPolicy(session=session, model="dreamlabs/dream-1-beta-flash", timeout_s=60)
    with pytest.warns(UserWarning, match=r"10 Hz.*-E control_hz=20"):
        policy.bind(yam_info(control_hz=10))
    assert policy.preroll_steps == 16
    [checked] = session.checked
    assert (checked.model, checked.timeout_s) == ("dreamlabs/dream-1-beta-flash", 60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy.bind(yam_info(control_hz=None))
        policy.bind(yam_info(camera=(360, 640)))
    assert policy.preroll_steps == 32


@pytest.mark.parametrize(("hz", "steps"), [(5, 8), (7, 12), (12, 20), (20, 32), (30, 48)])
def test_the_preroll_spans_the_whole_frame_window(hz: float, steps: int) -> None:
    assert bound(control_hz=hz).preroll_steps == steps
    assert steps / hz >= 1.6


def test_bind_warns_when_frames_would_be_stretched() -> None:
    with pytest.warns(UserWarning, match=r"top_cam \(224x224\), left_cam .* stretched"):
        DreamPolicy(session=FakeSession()).bind(yam_info(camera=(224, 224)))


def test_reset_needs_an_instruction_and_act_needs_a_reset() -> None:
    policy = DreamPolicy(session=FakeSession())
    with pytest.raises(RuntimeError, match="reset"):
        policy.act(observation())
    with pytest.raises(ValueError, match="instruction"):
        policy.reset(Scene(id="s", instruction=""))


def test_preroll_holds_the_first_pose_while_the_window_fills(scene: Scene) -> None:
    session = FakeSession()
    policy = bound(session)
    policy.reset(scene)
    assert played(run(policy, 32, state=HOME)) == [HOME.tolist()] * 32
    assert session.commanded == []
    timestamps = [timestamp for *_, timestamp in session.ticks]
    assert timestamps == pytest.approx([step / 20 for step in range(32)])
    frames, state, prompt, _ = session.ticks[0]
    assert set(frames) == {"top_cam", "left_cam", "right_cam"}
    assert state.tolist() == HOME.tolist() and prompt == scene.instruction


def test_chunks_play_row_by_row_and_continue_from_the_last_row(scene: Scene) -> None:
    first, second = np.tile(STATE, (48, 1)), np.tile(HOME, (48, 1))
    first[-1, 0] = 0.25
    session = FakeSession(first, second)
    policy = bound(session, preroll_steps=0)
    policy.reset(scene)
    chunks = run(policy, 49)
    assert chunks[0].inference_latency_s == 0.4 and chunks[1].inference_latency_s is None
    assert played(chunks[:48]) == first.tolist() and played(chunks[48:]) == [HOME.tolist()]
    assert session.commanded[0] is None
    assert session.commanded[1] is not None and session.commanded[1].tolist() == first[-1].tolist()
    assert all(chunk.control_hz == 20.0 for chunk in chunks)


def test_rows_per_request_sets_how_often_dream_is_asked(scene: Scene) -> None:
    policy = bound(rows_per_request=8, preroll_steps=0)
    policy.reset(scene)
    run(policy, 24)
    assert [request["step"] for request in policy.transcript()["requests"]] == [0, 8, 16]


def test_dry_run_asks_dream_but_holds_still(scene: Scene) -> None:
    session = FakeSession()
    policy = bound(session, preroll_steps=2, dry_run=True)
    policy.reset(scene)
    assert played(run(policy, 52, state=HOME)) == [HOME.tolist()] * 52
    assert session.commanded == [None, None]
    assert policy.transcript()["tracking_error_rad"] is None


def test_an_observation_instruction_overrides_the_scene(scene: Scene) -> None:
    session = FakeSession()
    policy = bound(session)
    policy.reset(scene)
    policy.act(observation(instruction="now stack the cups"))
    assert session.ticks[-1][2] == "now stack the cups"


def test_transcript_reports_requests_limits_and_tracking(scene: Scene) -> None:
    rows = np.tile(STATE, (48, 1))
    rows[:10, 3] = HIGH[3] + 0.5
    rows[:10, 10] = HIGH[10] + 0.5
    policy = bound(FakeSession(rows), preroll_steps=0)
    policy.reset(scene)
    target = policy.act(observation(HOME)).actions[0].data
    lagging = target.copy()
    lagging[1] -= 0.1
    policy.act(observation(lagging))
    policy.act(observation(target))
    transcript = policy.transcript()
    assert transcript["requests"] == [
        {
            "step": 0,
            "rtt_s": 0.4,
            "infer_ms": 300.0,
            "attempts": 1,
            "rows": 48,
            "rows_beyond_limits": 10,
            "session_id": "s",
        }
    ]
    tracking = transcript["tracking_error_rad"]
    assert (tracking["mean"], tracking["max"]) == pytest.approx((0.05, 0.1))
    assert transcript["model"] == "dreamlabs/dream-1-beta" and transcript["server_url"] is None
    json.dumps(transcript)
    transcript["requests"][0]["rows"] = 0
    assert policy.transcript()["requests"][0]["rows"] == 48


def test_without_embodiment_bounds_no_row_is_beyond_limits(scene: Scene) -> None:
    unbounded = Box(shape=(14,), semantics=ActionSemantics(control_mode="joint_pos"))
    policy = DreamPolicy(session=FakeSession(np.tile(HIGH + 1, (4, 1))), preroll_steps=0)
    policy.bind(yam_info(action_space=unbounded))
    policy.reset(scene)
    policy.act(observation())
    assert policy.transcript()["requests"][0]["rows_beyond_limits"] == 0


def test_a_new_trial_starts_from_an_empty_plan(scene: Scene) -> None:
    session = FakeSession()
    policy = bound(session, preroll_steps=0)
    policy.reset(scene)
    run(policy, 3)
    policy.reset(scene)
    run(policy, 1)
    assert len(session.episodes) == 2 and session.commanded[-1] is None
    assert len(policy.transcript()["requests"]) == 1


@pytest.mark.parametrize(
    ("broken", "match"),
    [
        (Observation(images=observation().images, state={}), "joint_pos"),
        (Observation(images={}, state=observation().state), "no image"),
    ],
)
def test_act_names_what_the_observation_lacks(
    scene: Scene, broken: Observation, match: str
) -> None:
    policy = bound()
    policy.reset(scene)
    with pytest.raises(ValueError, match=match):
        policy.act(broken)


def test_the_policy_opens_its_own_session_and_reopens_after_close(
    monkeypatch: pytest.MonkeyPatch, scene: Scene
) -> None:
    clients: list[FakeClient] = []

    def dream_session() -> DreamSession:
        clients.append(FakeClient())
        return DreamSession(lambda: clients[-1])

    monkeypatch.setattr(policy_module, "DreamSession", dream_session)
    with DreamPolicy() as policy:
        policy.reset(scene)
    policy.close()
    policy.reset(scene)
    policy.close()
    assert [client.names() for client in clients] == [["get_models", "policy", "close"]] * 2


def test_an_injected_session_is_closed_but_kept() -> None:
    session = FakeSession()
    policy = DreamPolicy(session=session)
    policy.close()
    policy.close()
    assert session.closed == 2


def test_close_before_any_session_does_nothing() -> None:
    DreamPolicy().close()


def test_a_server_url_selects_the_bridge(monkeypatch: pytest.MonkeyPatch, scene: Scene) -> None:
    opened: list[str] = []

    def bridge_session(url: str) -> FakeSession:
        opened.append(url)
        return FakeSession()

    monkeypatch.setattr(policy_module, "BridgeSession", bridge_session)
    policy = DreamPolicy(server_url="http://127.0.0.1:8210")
    policy.reset(scene)
    assert opened == ["http://127.0.0.1:8210"]
    assert policy.remedy is not None and "inspect-robots-dream-bridge" in policy.remedy
    policy.close()
