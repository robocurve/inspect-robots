# Rerun per-arm blueprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Rerun viewer shows one time-series plot per labeled action-dim group (YAM: "left" with dims 0-6, "right" with dims 7-13, commanded and measured series together) instead of the heuristic subset it picks today, by sending an explicit blueprint at eval start (#265).

**Architecture:** A new duck-typed sink hook (`bind_spaces`, mirroring the existing `bind_task`/`log_policy_messages` getattr precedents) hands sinks the resolved `Box` action space and `ObservationSpace` before `on_eval_start`. `RerunSink` distills them into plain fields (dim labels, action dim count, camera names, state keys and 1-D state field lengths), derives groups by first-underscore label prefix, and sends one blueprint per trial namespace FROM THE WORKER THREAD: `_emit` notices the first payload of each new prefix and sends a blueprint whose views use concrete exact paths under that prefix (rerun's entity-query grammar supports `/**` ONLY as a suffix — mid-path wildcards like `trial/**/action/0` are accepted silently but match nothing, so per-prefix sends with exact paths are the only correct shape). The existing threading contract ("all SDK calls after startup happen on the worker") is preserved untouched. Every failure mode (hook never called, unlabeled dims, SDK without blueprint support, build/send exception) degrades to today's heuristic layout with at most one warning.

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
- Produces on `RerunSink`: `bind_spaces(action_space: Box, observation_space: ObservationSpace) -> None` (stores distilled fields; needs `Box`/`ObservationSpace` added to the `TYPE_CHECKING` imports of rerun_sink.py), module-level `_joint_groups(labels: tuple[str, ...] | None, dim: int) -> list[tuple[str, list[int]]]`, and private `_send_blueprint(rr: Any, prefix: str) -> None` invoked from `_emit` on the worker whenever a payload's prefix differs from `self._blueprint_prefix` (new `__init__` field, `str | None = None`). The module docstring gains one sentence describing the per-trial blueprint send in `_emit`; the threading-contract paragraph needs NO exception added.
- Grouping semantics (`_joint_groups`): collapse to a single `("joints", all)` group unless labels are present, their count equals `dim`, EVERY label contains an underscore, and there are at least two distinct first-underscore prefixes. This keeps CubePick's `("dx","dy")` and eef-style `("x","y","z","grip")` labels in one plot instead of one clutter-view per dim.

- [ ] **Step 1: failing tests** (fake `rr` namespace per existing conventions, extended with a fake `blueprint` submodule whose view/container classes record their kwargs, plus a recording `send_blueprint`):

```python
def test_joint_groups_split_on_label_prefix() -> None:
    labels = tuple(f"{s}_{p}" for s in ("left", "right") for p in ("j0", "j1", "gripper"))
    assert _joint_groups(labels, 6) == [("left", [0, 1, 2]), ("right", [3, 4, 5])]
    assert _joint_groups(None, 3) == [("joints", [0, 1, 2])]
    assert _joint_groups(("a_x", "a_y"), 2) == [("joints", [0, 1])]  # single prefix
    assert _joint_groups(("ax", "by"), 2) == [("joints", [0, 1])]  # no underscore anywhere
    assert _joint_groups(("l_a", "up", "r_b"), 3) == [("joints", [0, 1, 2])]  # mixed labels
    assert _joint_groups(("l_a", "r_b"), 3) == [("joints", [0, 1, 2])]  # label/dim mismatch


def test_blueprint_sent_per_trial_prefix_with_per_group_views(...) -> None:
    # bind_spaces with a 4-dim Box labeled left_j0,left_gripper,right_j0,right_gripper,
    # an ObservationSpace with one camera "top" and StateSpec field joint_pos shape (4,).
    # Drive: on_eval_start(fake) -> on_trial_start("s0", 0) -> log_step -> flush.
    # send_blueprint called exactly once; recorded TimeSeriesView kwargs include a
    # "left" view whose contents contain "+ trial/s0/e0/action/0", ".../action/1",
    # "+ trial/s0/e0/state/joint_pos/0", ".../joint_pos/1" and a "right" view with
    # dims 2,3; a camera view for "top" ("+ trial/s0/e0/camera/top"); a TextLogView
    # whose contents include BOTH "+ trial/s0/e0/llm" and "+ trial/s0/e0/event/**"
    # (the terminated marker must stay visible); an llm/latest TextDocumentView; a
    # reward view. Then on_trial_start("s1", 0) -> log_step -> flush: a SECOND send
    # with trial/s1/e0 paths. A further log_step in the same trial: still two sends.
    # Also assert the FIRST payload being a transcript (log_policy_messages before
    # any log_step) triggers the send too, in a small separate test or parametrize.


def test_blueprint_state_keys_without_statespec_get_own_views(...) -> None:
    # ObservationSpace with state_keys={"eef_pos","cube_pos"} and state=None (the
    # CubePick shape): per-key TimeSeriesViews with contents
    # "+ trial/s0/e0/state/eef_pos/**" (suffix wildcard is legal) and no aligned
    # overlay inside the joints view. Camera-less space here -> no camera grid
    # (covers the zero-cameras arm). StateSpec fields: joint_pos (4,) aligned;
    # eef_pose (7,) NOT aligned -> own view; a (2,2)-shaped field is ignored for
    # alignment but still gets a per-key view (cover in this or a sibling test).


def test_blueprint_skipped_without_bind_spaces_or_blueprint_api(...) -> None:
    # (a) full drive without any bind_spaces call -> send_blueprint never called
    #     (existing startup fakes prove no attribute is even probed);
    # (b) bind_spaces called but fake rr lacks the blueprint attr -> no crash, no send;
    # (c) fake rr has blueprint but lacks send_blueprint -> no crash, no send.


def test_blueprint_build_failure_warns_once_and_run_continues(...) -> None:
    # Fake blueprint view class raises in __init__ -> exactly one RuntimeWarning
    # mentioning the automatic layout fallback ACROSS TWO TRIALS (the second prefix
    # must not re-warn), and scalar log calls still happen afterwards.


@pytest.mark.skipif(not _RERUN_INSTALLED, reason="rerun-sdk not installed")
def test_real_rerun_accepts_the_blueprint(...) -> None:
    # Mirror test_real_rerun_accepts_the_transcript_document_call: real SDK, save to
    # tmp .rrd; bind_spaces with the labeled Box + camera + StateSpec space, then call
    # sink._send_blueprint(rr, "trial/s0/e0") directly under
    # warnings.simplefilter("error") -> no fallback warning means every constructor
    # kwarg and send_blueprint call is real. (Viewer-side content matching cannot be
    # asserted in-process; the exact-path design is what makes it correct by
    # construction.)
```

- [ ] **Step 2: run, confirm FAIL.**
- [ ] **Step 3: implement.** Distillation and grouping:

```python
def _joint_groups(labels: tuple[str, ...] | None, dim: int) -> list[tuple[str, list[int]]]:
    """Group action dims by their label's first-underscore prefix.

    YAM-style labels (``left_j0`` .. ``right_gripper``) yield one group per
    side, in first-appearance order. Missing labels, a label count that does
    not match the dim count, any label without an underscore (CubePick's
    ``dx``/``dy``, eef-style ``x``/``y``/``grip``), or fewer than two distinct
    prefixes all collapse to a single "joints" group: one plot beats one
    clutter-view per dim.
    """
    if labels is None or len(labels) != dim or not all("_" in label for label in labels):
        return [("joints", list(range(dim)))]
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label.split("_", 1)[0], []).append(index)
    if len(groups) < 2:
        return [("joints", list(range(dim)))]
    return list(groups.items())
```

`bind_spaces` stores plain derived state (set in `__init__` defaults: `self._dim_labels: tuple[str, ...] | None = None`, `self._action_dim: int | None = None`, `self._camera_names: tuple[str, ...] = ()`, `self._state_keys: tuple[str, ...] = ()`, `self._state_lengths: dict[str, int] = {}`, `self._blueprint_prefix: str | None = None`, `self._blueprint_warned = False`):

```python
    def bind_spaces(self, action_space: Box, observation_space: ObservationSpace) -> None:
        """Distill the resolved spaces into the fields the blueprint needs.

        Called by ``eval()`` before ``on_eval_start`` (and therefore before
        the worker thread exists); storing plain tuples rather than the space
        objects keeps the worker free of shared mutable state.
        """
        self._action_dim = action_space.dim
        semantics = action_space.semantics
        self._dim_labels = None if semantics is None else semantics.dim_labels
        self._camera_names = tuple(c.name for c in observation_space.cameras)
        self._state_keys = tuple(sorted(observation_space.state_keys))
        state = observation_space.state
        self._state_lengths = (
            {f.key: f.shape[0] for f in state.fields if len(f.shape) == 1}
            if state is not None
            else {}
        )
```

(Verify `Box.dim` is the dim-count property in spaces.py:127-133 — `Box.size` does NOT exist.)

`_send_blueprint(rr, prefix)` runs on the worker. Trigger at the top of `_emit` (before the transcript `isinstance` branch, so a transcript-first trial still gets its layout):

```python
        if payload.prefix != self._blueprint_prefix:
            self._blueprint_prefix = payload.prefix
            self._send_blueprint(rr, payload.prefix)
```

```python
    def _send_blueprint(self, rr: Any, prefix: str) -> None:
        """Send an explicit layout for one trial namespace, if the SDK can.

        Rerun's entity queries support ``/**`` only as a suffix, so the views
        name each trial's entities with concrete paths and the layout is
        re-sent when a new trial namespace begins (the viewer follows the
        live trial; a single-trial ``run`` sends exactly once). Skipped when
        spaces were never bound or the SDK predates blueprints; any build or
        send failure degrades to the automatic layout with a single warning.
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
                contents = [f"+ {prefix}/action/{i}" for i in indices]
                for key, length in self._state_lengths.items():
                    if length == self._action_dim:
                        contents += [f"+ {prefix}/state/{key}/{i}" for i in indices]
                plots.append(rrb.TimeSeriesView(name=name, origin="/", contents=contents))
            aligned = {
                key for key, length in self._state_lengths.items()
                if length == self._action_dim
            }
            for key in self._state_keys:
                if key not in aligned:
                    plots.append(
                        rrb.TimeSeriesView(
                            name=key, origin="/", contents=[f"+ {prefix}/state/{key}/**"]
                        )
                    )
            plots.append(
                rrb.TimeSeriesView(name="reward", origin="/", contents=[f"+ {prefix}/reward"])
            )
            cameras = [
                rrb.Spatial2DView(name=cam, origin="/", contents=[f"+ {prefix}/camera/{cam}"])
                for cam in self._camera_names
            ]
            text = rrb.Vertical(
                rrb.TextDocumentView(name="latest", contents=[f"+ {prefix}/llm/latest"]),
                rrb.TextLogView(
                    name="llm", contents=[f"+ {prefix}/llm", f"+ {prefix}/event/**"]
                ),
            )
            columns = [rrb.Vertical(*plots), text]
            if cameras:
                columns.insert(0, rrb.Grid(*cameras))
            send(rrb.Blueprint(rrb.Horizontal(*columns)))
        except Exception as exc:
            if not self._blueprint_warned:
                self._blueprint_warned = True
                warnings.warn(
                    f"RerunSink could not send the blueprint layout ({exc}); "
                    "the viewer will use its automatic layout",
                    RuntimeWarning,
                    stacklevel=2,
                )
```

Notes: keys in `_state_keys` without a 1-D declared length (StateSpec absent, or a non-1-D field) get suffix-wildcard views so nothing the run logs is hidden — an explicit blueprint hides unmatched entities, which is also why `event/**` rides in the TextLogView (the `event/terminated` marker must survive). `on_eval_start` is NOT touched beyond the docstring sentence; all sends happen on the worker, so the module threading contract holds verbatim.

Coverage note: every branch above is reachable from the tests in Step 1 (labels/no-labels/no-underscore/mismatch groups; aligned + unaligned + undeclared state; zero cameras; missing rrb vs missing send; `_action_dim is None` via the no-bind drive; warn-once across two prefixes; prefix-change true/false arcs via the second log_step). If a branch cannot be reached, delete the branch rather than excluding it.

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

- Preserving operator layout tweaks across trial boundaries: rerun has no mid-path wildcards, so the layout must be re-sent per trial namespace and each re-send resets viewer tweaks. Accepted: the layout follows the live trial, and the motivating single-instruction `run` flow sends exactly once. `make_active`/`make_default` juggling stays at SDK defaults until someone needs it.
- A declaration protocol for plot grouping (PLOT_GROUPS-style): `dim_labels` already carries the grouping and is conformance-checked; a dedicated protocol is YAGNI until an embodiment needs groups that labels cannot express.
- Splitting commanded vs measured into separate views, styling (colors, line styles), and gripper-vs-joint separation: the per-side overlay is the requested "one plot per hand"; refinements are viewer-side tweaks operators can make live, and the sent layout never overwrites them.
- The `run --policy agent` CLI path needs no change: it reaches `eval()` and the bus like every other run.
