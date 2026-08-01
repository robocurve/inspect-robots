# Rerun per-arm blueprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Rerun viewer shows one time-series plot per labeled action-dim group (YAM: "left" with dims 0-6, "right" with dims 7-13, commanded and measured series together) instead of the heuristic subset it picks today, by sending an explicit blueprint at eval start (#265).

**Architecture:** A new duck-typed sink hook (`bind_spaces`, mirroring the existing `bind_task`/`log_policy_messages` getattr precedents) hands sinks the resolved `Box` action space and `ObservationSpace` before `on_eval_start`. `RerunSink` distills them into plain fields (dim labels, action dim count, camera names, 1-D state field lengths), derives groups by first-underscore label prefix, and sends one blueprint right after `init`/`spawn`/`connect_grpc`/`save` in `on_eval_start` (caller thread — blueprint sends are timeline-independent, so the worker-thread rule is untouched). View contents use `trial/**/...` wildcard expressions so a single send covers every trial namespace. Every failure mode (hook never called, unlabeled dims, SDK without blueprint support, build/send exception) degrades to today's heuristic layout.

**Tech Stack:** Python 3.10+, stdlib + numpy only in core; rerun-sdk stays a lazy optional import; pytest with the existing fake-SDK conventions in tests/test_rerun_sink.py.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict, src + tests), `uv run pytest --cov -q` at **100% branch coverage**.
- D1 docstrings on public defs; state the contract. Line length 100.
- Core never imports rerun-sdk at module scope (`core-only-import` CI job).
- No behavior change when the hook is absent, when a non-RerunSink sink lacks `bind_spaces`, or when rerun-sdk lacks blueprint support: every existing test passes untouched.
- Worktree: `~/robocurve/ir-wt-rerun-blueprint`; run everything via `uv run ...` there. Reference #265 in commit messages; end each with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

## Reference: current wiring (main @ 66b50ed7)

- `src/inspect_robots/eval.py:82-99` — `_Broadcast.__init__` collects optional `log_policy_messages` hooks via `getattr`; the class is the home for the new fan-out method. `bus = _Broadcast(sink_list)` at :271; `bus.on_eval_start(spec)` at :301; `assert_compatible` at :246; the `bind_task` duck-typed precedent at :250-255. The embodiment is fully resolved by :246 with `embodiment.action_space: Box` and `embodiment.observation_space: ObservationSpace` (`embodiment.py:56-57`).
- `src/inspect_robots/spaces.py:87-94` — `Box.shape`, `.semantics: ActionSemantics | None`; `ActionSemantics.dim_labels: tuple[str, ...] | None` (:48-68). `ObservationSpace.cameras: tuple[CameraSpec, ...]`, `.state: StateSpec | None`, `.state_keys` (:183+); `StateField(key, shape, ...)`.
- `src/inspect_robots/logging/rerun_sink.py` — module docstring states the threading contract (startup calls on caller, everything after on the worker); `__init__` at :147; `on_eval_start` at :470 (the `try` block ends after `rr.save`); `_emit` logs `{pre}/camera/{cam}`, `{pre}/state/{key}/{i}`, `{pre}/action/{i}`, `{pre}/reward`; transcripts at `{prefix}/llm` and `{prefix}/llm/latest`; trial prefixes are `trial/<scene_id>/e<epoch>` (:499).
- `tests/test_rerun_sink.py` — fake-SDK conventions: `_StartupRR` (:70) records startup calls; richer fakes below it record `log` calls; tests drive `eval()` end-to-end with `CubePickEmbodiment`/`ScriptedPolicy` and a monkeypatched `sys.modules["rerun"]`. Mirror these idioms; find the fake used by on_eval_start tests with `grep -n "_StartupRR\|sys.modules" tests/test_rerun_sink.py`.
- `CubePickEmbodiment` (mock) — check `src/inspect_robots/mock/` for whether its action space declares `dim_labels`; the eval-level test must know whether to expect groups or the single-view fallback, and a labeled fake embodiment can be built inline if needed.

---

### Task 1: `bind_spaces` fan-out in eval

**Files:**
- Modify: `src/inspect_robots/eval.py` (`_Broadcast`; the eval body between :271 and :301)
- Test: `tests/test_eval.py` (or wherever `_Broadcast`/sink-lifecycle tests live — `grep -n "log_policy_messages" tests/*.py`)

**Interfaces:**
- Produces: `_Broadcast.bind_spaces(action_space: Box, observation_space: ObservationSpace) -> None` — fans to each sink's callable `bind_spaces` attribute (getattr at call time; a one-shot call needs no precollected hook list). `eval()` calls `bus.bind_spaces(embodiment.action_space, embodiment.observation_space)` immediately before `bus.on_eval_start(spec)`.

- [ ] **Step 1: failing tests** — a sink WITH `bind_spaces` receives exactly the embodiment's spaces before its `on_eval_start` (record call order in a list); a sink WITHOUT the attribute is untouched and the eval completes; a sink whose `bind_spaces` is a non-callable attribute is skipped. Drive through `eval()` with the mock task/embodiment/policy, following the harness of the existing `log_policy_messages` tests.
- [ ] **Step 2: run, confirm FAIL** (`uv run pytest tests/<file> -k bind_spaces -v`).
- [ ] **Step 3: implement**

```python
    def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
        """Offer the resolved spaces to sinks that declare a bind_spaces hook.

        Duck-typed like ``log_policy_messages``: sinks without the attribute
        are unaffected, so the sink Protocol is unchanged.
        """
        for sink in self._sinks:
            hook = getattr(sink, "bind_spaces", None)
            if callable(hook):
                hook(action_space, observation_space)
```

Call site in `eval()` (directly before `bus.on_eval_start(spec)`): `bus.bind_spaces(embodiment.action_space, embodiment.observation_space)`. Import types under `TYPE_CHECKING` if not already imported.

- [ ] **Step 4: run to green; Step 5: commit** `eval: offer resolved spaces to sinks via duck-typed bind_spaces (#265)`

---

### Task 2: blueprint construction and send in RerunSink

**Files:**
- Modify: `src/inspect_robots/logging/rerun_sink.py`
- Test: `tests/test_rerun_sink.py`

**Interfaces:**
- Produces on `RerunSink`: `bind_spaces(action_space: Box, observation_space: ObservationSpace) -> None` (stores distilled fields), module-level `_joint_groups(labels: tuple[str, ...] | None, dim: int) -> list[tuple[str, list[int]]]`, and private `_send_blueprint(rr: Any) -> None` invoked at the end of `on_eval_start`'s `try` block.

- [ ] **Step 1: failing tests** (fake `rr` namespace per existing conventions, extended with a fake `blueprint` submodule whose view/container classes record their kwargs, plus a recording `send_blueprint`):

```python
def test_joint_groups_split_on_label_prefix() -> None:
    labels = tuple(f"{s}_{p}" for s in ("left", "right") for p in ("j0", "j1", "gripper"))
    assert _joint_groups(labels, 6) == [("left", [0, 1, 2]), ("right", [3, 4, 5])]
    assert _joint_groups(None, 3) == [("joints", [0, 1, 2])]
    assert _joint_groups(("a_x", "a_y"), 2) == [("joints", [0, 1])]  # single prefix
    assert _joint_groups(("ax", "by"), 2) == [("joints", [0, 1])]  # no underscore
    assert _joint_groups(("l_a", "r_b"), 3) == [("joints", [0, 1, 2])]  # label/dim mismatch


def test_blueprint_sent_with_per_group_views(...) -> None:
    # bind_spaces with a 4-dim Box labeled left_j0,left_gripper,right_j0,right_gripper,
    # an ObservationSpace with one camera "top" and StateSpec field joint_pos shape (4,);
    # on_eval_start must call send_blueprint exactly once, after init; the recorded
    # TimeSeriesView kwargs must include a "left" view whose contents contain
    # "+ trial/**/action/0", "+ trial/**/action/1", "+ trial/**/state/joint_pos/0",
    # "+ trial/**/state/joint_pos/1" and a "right" view with dims 2,3; a camera view
    # for "top"; llm and llm/latest views; a reward view.


def test_blueprint_state_key_length_mismatch_gets_own_view(...) -> None:
    # StateSpec fields: joint_pos shape (4,) aligned; eef_pose shape (7,) NOT aligned
    # with the 4-dim action box -> eef_pose appears as its own TimeSeriesView with
    # contents "+ trial/**/state/eef_pose/**" and NOT inside left/right views.


def test_blueprint_skipped_without_bind_spaces_or_blueprint_api(...) -> None:
    # (a) on_eval_start without any bind_spaces call -> send_blueprint never called;
    # (b) bind_spaces called but fake rr lacks the blueprint attr -> no crash, no send;
    # (c) fake rr lacks send_blueprint -> no crash, no send.


def test_blueprint_build_failure_warns_once_and_run_continues(...) -> None:
    # Fake blueprint class raises in __init__ -> exactly one RuntimeWarning mentioning
    # the automatic layout fallback; init/connect calls still recorded; log_step still
    # enqueues afterwards (sink not disabled).
```

- [ ] **Step 2: run, confirm FAIL.**
- [ ] **Step 3: implement.** Distillation and grouping:

```python
def _joint_groups(labels: tuple[str, ...] | None, dim: int) -> list[tuple[str, list[int]]]:
    """Group action dims by their label's first-underscore prefix.

    YAM-style labels (``left_j0`` .. ``right_gripper``) yield one group per
    side, in first-appearance order. Missing labels, a label count that does
    not match the dim count, or fewer than two distinct prefixes all collapse
    to a single "joints" group so unlabeled embodiments keep one plot.
    """
    if labels is None or len(labels) != dim:
        return [("joints", list(range(dim)))]
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        prefix = label.split("_", 1)[0] if "_" in label else label
        groups.setdefault(prefix, []).append(index)
    if len(groups) < 2:
        return [("joints", list(range(dim)))]
    return list(groups.items())
```

`bind_spaces` stores plain derived state (set in `__init__` defaults: `self._dim_labels: tuple[str, ...] | None = None`, `self._action_dim: int | None = None`, `self._camera_names: tuple[str, ...] = ()`, `self._state_lengths: dict[str, int] = {}`):

```python
    def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
        """Distill the resolved spaces into the fields the blueprint needs.

        Called by ``eval()`` before ``on_eval_start``; storing plain tuples
        (never the space objects) keeps the worker thread free of shared
        mutable state.
        """
        self._action_dim = action_space.size
        semantics = action_space.semantics
        self._dim_labels = None if semantics is None else semantics.dim_labels
        self._camera_names = tuple(c.name for c in observation_space.cameras)
        state = observation_space.state
        self._state_lengths = (
            {f.key: f.shape[0] for f in state.fields if len(f.shape) == 1}
            if state is not None
            else {}
        )
```

`_send_blueprint(rr)` — called as the last statement inside `on_eval_start`'s existing `try` block? NO: a blueprint failure must not disable the sink like a connect failure does. Call it AFTER the `try/except`, guarded by its own `try` and by `if self._rr is None or self._disabled: return`:

```python
    def _send_blueprint(self, rr: Any) -> None:
        """Send an explicit layout once, if spaces were bound and the SDK can.

        Wildcard ``trial/**`` contents cover every trial namespace, so one
        send at eval start is enough and later trials never stomp operator
        layout tweaks. Any failure degrades to the viewer's automatic layout
        with a single warning; the data stream is unaffected.
        """
        if self._action_dim is None:
            return
        rrb = getattr(rr, "blueprint", None)
        send = getattr(rr, "send_blueprint", None)
        if rrb is None or send is None:
            return
        try:
            plots = []
            for name, indices in _joint_groups(self._dim_labels, self._action_dim):
                contents = [f"+ trial/**/action/{i}" for i in indices]
                for key, length in self._state_lengths.items():
                    if length == self._action_dim:
                        contents += [f"+ trial/**/state/{key}/{i}" for i in indices]
                plots.append(rrb.TimeSeriesView(name=name, origin="/", contents=contents))
            for key, length in self._state_lengths.items():
                if length != self._action_dim:
                    plots.append(
                        rrb.TimeSeriesView(
                            name=key, origin="/", contents=[f"+ trial/**/state/{key}/**"]
                        )
                    )
            plots.append(
                rrb.TimeSeriesView(name="reward", origin="/", contents=["+ trial/**/reward"])
            )
            cameras = [
                rrb.Spatial2DView(name=cam, origin="/", contents=[f"+ trial/**/camera/{cam}"])
                for cam in self._camera_names
            ]
            text = rrb.Vertical(
                rrb.TextDocumentView(name="latest", contents=["+ trial/**/llm/latest"]),
                rrb.TextLogView(name="llm", contents=["+ trial/**/llm"]),
            )
            columns = [rrb.Vertical(*plots), text]
            if cameras:
                columns.insert(0, rrb.Grid(*cameras))
            send(rrb.Blueprint(rrb.Horizontal(*columns)))
        except Exception as exc:
            warnings.warn(
                f"RerunSink could not send the blueprint layout ({exc}); "
                "the viewer will use its automatic layout",
                RuntimeWarning,
                stacklevel=2,
            )
```

In `on_eval_start`, after the existing `except Exception` block (sink still enabled path only): `if not self._disabled: self._send_blueprint(rr)`. Guard ordering: the existing failure path sets `self._rr = None` and `self._disabled = True`, so the call is skipped after startup failure.

Coverage note: every branch above is reachable from the tests in Step 1 (labels/no-labels/mismatch groups; aligned + unaligned state; zero cameras via `columns.insert` skip; missing rrb/send; exception path; `_action_dim is None`). If a branch cannot be reached, delete the branch rather than excluding it.

- [ ] **Step 4: run to green, full gates; Step 5: commit** `rerun: send per-arm blueprint from bound spaces (#265)`

---

### Task 3: docs

**Files:**
- Modify: the rerun docs section (`grep -rn "rerun" docs/ README.md --include="*.md" -l` and pick the page describing the viewer; likely `docs/guide/cli.md` or a visualization guide), `CHANGELOG.md` (`## [Unreleased]` → `### Added`), `src/inspect_robots/CLAUDE.md` module-map row for `logging/`.

- [ ] **Step 1:** One short paragraph in the viewer docs: the sink sends a layout grouping joint series per labeled arm (commanded `action/*` and measured `state/*` together per side), with cameras, the LLM transcript pane, and reward laid out alongside; embodiments without `dim_labels` get one combined joints plot; the layout is sent once at eval start and never re-sent, so operator tweaks survive trial boundaries. No em dashes in prose.
- [ ] **Step 2:** CHANGELOG under `### Added`, referencing #265 and plan 0041, mirroring existing entry format.
- [ ] **Step 3:** Extend the module-map row for the rerun sink with the blueprint mention (plan 0041), matching phrasing density.
- [ ] **Step 4:** `uv run pytest -q` green; commit `docs: describe the per-arm rerun blueprint layout (#265)`.

---

## Out of scope

- Per-trial blueprint re-sends or `make_active` juggling: wildcard contents make one send sufficient; re-sending would stomp operator layout tweaks between trials.
- A declaration protocol for plot grouping (PLOT_GROUPS-style): `dim_labels` already carries the grouping and is conformance-checked; a dedicated protocol is YAGNI until an embodiment needs groups that labels cannot express.
- Splitting commanded vs measured into separate views, styling (colors, line styles), and gripper-vs-joint separation: the per-side overlay is the requested "one plot per hand"; refinements are viewer-side tweaks operators can make live, and the sent layout never overwrites them.
- The `run --policy agent` CLI path needs no change: it reaches `eval()` and the bus like every other run.
