# 0062 — Session-owned end-gesture hint on footer status lines

- **Status:** draft
- **Issue:** #345
- **Critique rounds:** R1: 4 substantive (skew guard preserved stale "Enter
  ends the episode" prose instead of replacing it; yam banner rewrite
  reintroduced cross-boundary drift for the message affordance; footer-active
  is not coextensive with a live Esc gesture on the console-degradation path;
  storage-time suffixing forfeits the width decision and its plain-path safety
  is incidental, not structural) — all four redesigned below.

## Problem

The episode-end gesture (Esc, `/stop`) is owned by the framework console
(`console.py` grammar + `session.py` footer editor, plan 0056). But the status
text that *describes* the gesture is composed by embodiment plugins: yam's
per-second ticker (`t = 4s / 1200s | Esc ends the episode`) hardcodes gesture
prose. When 0.47 moved the gesture from Enter to Esc, core's own strings
updated atomically with the behavior while yam's description went stale on the
rigs (yam#120/#121). Prose duplicated across an API boundary drifts; the fix
is to move the prose to the side that owns the behavior.

## Design

`OperatorSession` composes the end-gesture hint into footer-mode status
rendering itself. Plugins report rig state only (`t = 4s / 1200s`); they never
mention the gesture. If a plugin does mention it (old versions), the session
*replaces* that clause with the canonical one, so stale gesture prose can
never reach the terminal.

### Composition rule (render-time, not storage-time)

`self._status_line` always stores exactly what the plugin passed. The hint is
composed where footer status text is rendered — `_clipped_status_text()`, the
single helper both the `status()` redraw and the `write_line` scrollback
redraw already share — so the plain path can never see a suffixed line by
construction (plain redraw uses `_status_line` directly and never calls the
composer).

```python
_END_HINT = "Esc ends the episode"
_END_HINT_PHRASE = "ends the episode"

def _composed_status_text(self) -> str:
    line = self._status_line  # raw plugin text
    head, sep, tail = line.rpartition(" | ")
    if sep and _END_HINT_PHRASE in tail:
        line = head  # replace a stale/duplicate trailing gesture clause
    if not line:
        return _END_HINT
    return f"{line} | {_END_HINT}"
```

- **Replace, not suppress.** Pre-#121 yam (≤0.27.x) tickers send
  `t = 4s | Enter ends the episode`; the trailing clause is stripped and the
  canonical hint appended, so the wrong gesture is *corrected*, not preserved.
  yam 0.28.0 sends the already-correct suffix; stripped and re-appended,
  rendering identically (dedup falls out of the same rule). A whole-line
  status that merely mentions the phrase without a `" | "` separator (e.g.
  "budget exhaustion ends the episode") is left intact and gets the hint
  appended after it — both clauses true, nothing lost. The only lossy case is
  a plugin using `" | "` before a non-gesture clause that happens to contain
  the phrase; accepted and documented in the composer's docstring.
- **Empty line:** `status("")` renders the bare hint, no dangling separator.

### Width priority: rig state first

The composed line adds 23 characters (`" | Esc ends the episode"`). On narrow
terminals the dynamic rig state must win over a static hint the operator has
already learned:

```python
def _clipped_status_text(self) -> str:
    width = self._width_fn() - 1
    composed = self._composed_status_text()
    if len(composed) <= width:
        return composed
    return _clip_tail(self._status_line, width)  # drop the hint, keep rig state
```

If even the raw line overflows, existing tail-clip semantics apply unchanged.
All branches are drivable through the injectable `width_fn` seam, so the 100%
branch-coverage gate is satisfiable.

### Footer window integrity (rollout change)

`begin_trial()` opens the footer *after* `embodiment.reset()` returns
(rollout.py:269→278) and `end_trial()` closes it in the per-trial `finally`,
so in the normal path footer-active is the window where Esc ends the episode.
One path breaks that today: when `operator_input.begin_trial()` or `.poll()`
raises, rollout disables the console for the rest of the trial
(`console_ok = False`) but leaves the footer up, so the hint (yam's today,
ours after this plan) advertises a dead gesture for the remainder of the
trial. Fix in the same PR: at the moment rollout sets `console_ok = False`, it
also calls the duck-typed best-effort `end_trial()` it already uses in the
`finally` (idempotent, guarded by `_footer_active`), restoring the terminal
and dropping subsequent statuses to the hint-less plain path.

### Non-goals / out of scope

- Plain-path statuses (pre-reset homing, post-trial parking, non-TTY
  fallback, degraded-console remainder) render exactly what the plugin set:
  outside the footer window the hint would be false or (non-TTY) describe the
  wrong gesture, and the console usage line covers those modes.
- Sessions built with a caller-injected `console=...` never enter footer mode
  (`enable_footer` is a documented no-op), so they never render the hint —
  correct, since the decision-7 dispatch seams aren't wired there.
- The console usage reminder (`USAGE` / `USAGE_END_ONLY`) is unchanged.
- Gate prompts ("press Enter…") are readiness gates, still Enter-driven,
  untouched.
- No public API change: constants and the composer stay module-private,
  `__all__` untouched, `tests/test_api_snapshot.py` unchanged.

## yam follow-up (separate PR in robocurve/inspect-robots-yam)

After this ships in a core release (call it 0.49):

- Deferred-mode ticker becomes `self._status(f"t = {span}")`.
- Deferred-mode banner becomes `f"Running.{limit}"` ("Running. Max 120s.") —
  rig-owned facts only. The previous draft kept "type a message + Enter to
  send feedback", but message delivery is a *console* affordance that core
  decides per policy (`accepts_operator_messages` → `USAGE` vs
  `USAGE_END_ONLY`); yam cannot know the mode, so any message clause is wrong
  for end-only policies — the same defect class this plan removes. Core's
  mode-correct usage line owns that prose. (The banner is set at the tail of
  `reset()`, before `begin_trial()`, so it renders on the plain path and
  never receives the footer hint; it simply stops making console claims.)
- Never-connected legacy path keeps its own "any key" text: that path reads
  raw stdin itself and never routes through an `OperatorSession`.
- `inspect-robots` floor rises to the release carrying this change, so a rig
  never runs hint-less (new yam + old core would otherwise show no end hint).

## Tests (core)

Extend `tests/test_session.py` footer coverage (assert on written bytes,
matching existing footer render tests):

1. Footer `status("t = 1s")` renders `t = 1s | Esc ends the episode`.
2. Replacement: `status("t = 1s | Enter ends the episode")` renders
   `t = 1s | Esc ends the episode` (stale gesture corrected — the motivating
   skew case); `status("t = 1s | Esc ends the episode")` renders identically
   (dedup).
3. Whole-line phrase without separator: `status("budget exhaustion ends the
   episode")` keeps the line and appends the hint.
4. `status("")` renders the bare hint.
5. Width: at a width fitting the raw line but not the composed one, the hint
   is dropped and the rig state shown un-clipped; at a width smaller than the
   raw line, existing tail-clip applies.
6. Plain mode: `status("parking")` renders without the hint.
7. Footer→plain transition: after `end_trial()`, a plain `status()` +
   `write_line` redraw never shows the hint (pins the structural guarantee).
8. Degraded console: a rollout whose operator input raises on `poll()` calls
   `end_trial()` at disable time — footer closed, later statuses hint-less
   (extends the existing rollout console-degradation test).
9. Existing `status(None)` close and repeated-status tests keep passing
   unchanged (render-time composition leaves storage semantics untouched).

Repo gates: ruff, ruff format, mypy strict (src+tests), pytest --cov at 100%
with branch coverage.

## Docs

- `docs/guide/cli.md`: note the footer status line always carries the
  framework-appended end-gesture hint, so embodiment status text never goes
  stale (short prose near the existing Esc gesture section).
- `src/inspect_robots/CLAUDE.md`: extend the `session.py` row (hint
  composition + replacement rule) and the `rollout.py` row (footer closed at
  console-disable time).
- `CHANGELOG.md`: Changed entry under Unreleased referencing this plan, #345,
  and the yam drift incident that motivated it.

## Release sequencing

1. Merge core PR, cut core minor (0.49.0): the appended hint is a visible
   behavior change for every footer-mode plugin status.
2. yam PR: drop gesture clauses, floor `>=0.49`, lock refresh, cut yam minor.
   Skew matrix: new core + yam 0.28.0 → replace-rule dedups (identical
   render); new core + yam ≤0.27.x → stale "Enter" clause *corrected* in
   place; new yam + old core → prevented by the floor bump. All combinations
   render a correct hint or are uninstallable.
