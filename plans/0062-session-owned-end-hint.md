# 0062 — Session-owned end-gesture hint on footer status lines

- **Status:** draft
- **Issue:** #345
- **Critique rounds:** R1: 4 substantive (skew guard preserved stale "Enter
  ends the episode" prose instead of replacing it; yam banner rewrite
  reintroduced cross-boundary drift for the message affordance; footer-active
  is not coextensive with a live Esc gesture on the console-degradation path;
  storage-time suffixing forfeits the width decision and its plain-path safety
  is incidental, not structural) — all four redesigned below. R2: 4
  substantive (width fallback clipped the RAW line, re-exposing stale prose
  and inverting rig-state-first; skew matrix overclaimed correction of the
  old yam banner, which renders on the plain path; end_trial-at-disable left
  a sticky plain status line colliding with the verdict/gate prompts; test
  list missed the multi-pipe non-gesture tail and the begin_trial-raise
  disable site) — all four resolved below.

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
never reach the terminal through a footer-rendered status. (Plain-path
statuses — e.g. an old yam banner set during `reset()` — are out of the
composer's reach by design; see the skew matrix.)

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

def _status_texts(self) -> tuple[str, str]:
    """Return (stripped, composed): plugin text minus any trailing gesture
    clause, and that text with the canonical hint appended."""
    assert self._status_line is not None  # call sites render only a set line
    line = self._status_line
    head, sep, tail = line.rpartition(" | ")
    if sep and _END_HINT_PHRASE in tail:
        line = head  # replace a stale/duplicate trailing gesture clause
    if not line:
        return "", _END_HINT
    return line, f"{line} | {_END_HINT}"
```

Both halves matter: `composed` is the normal render, and `stripped` (never the
raw line) is the only thing width-clipping may fall back to, so stale gesture
prose cannot re-enter through the clip path either.

- **Replace, not suppress.** Pre-#121 yam (≤0.27.x) tickers send
  `t = 4s | Enter ends the episode`; the trailing clause is stripped and the
  canonical hint appended, so the wrong gesture is *corrected*, not preserved.
  yam 0.28.0 sends the already-correct suffix; stripped and re-appended,
  rendering identically (dedup falls out of the same rule). A whole-line
  status that merely mentions the phrase without a `" | "` separator (e.g.
  "budget exhaustion ends the episode") is left intact and gets the hint
  appended after it — both clauses true, nothing lost. A multi-pipe line whose
  last segment is not gesture prose (`t = 4s | left arm ok`) is untouched and
  suffixed. Documented accepted costs (composer docstring): a `" | "` segment
  that contains the phrase without being gesture prose is dropped; a gesture
  mention placed mid-line or after a non-pipe separator renders a duplicated
  hint (no first-party plugin does either — a grep of `plugins/*` finds no
  `session.status` callers; yam is the only known one).
- **Empty line:** `status("")` renders the bare hint, no dangling separator.
  A line ending in exactly `" | "` composes with a doubled separator —
  harmless cosmetic, accepted.

### Width priority: rig state first

The composed line adds 23 characters (`" | Esc ends the episode"`). On narrow
terminals the dynamic rig state must win over a static hint the operator has
already learned:

```python
def _clipped_status_text(self) -> str:
    width = self._width_fn() - 1
    stripped, composed = self._status_texts()
    if len(composed) <= width:
        return composed
    return _clip_tail(stripped, width)  # drop the hint, keep rig state
```

The fallback clips the *stripped* line — never the raw one — so a narrow
terminal drops the static hint and keeps the dynamic rig state, and a stale
plugin gesture clause cannot ride back in through the clip path. If even the
stripped line overflows, existing tail-clip semantics apply. All branches are
drivable through the injectable `width_fn` seam, so the 100% branch-coverage
gate is satisfiable.

### Footer window integrity (rollout change)

`begin_trial()` opens the footer *after* `embodiment.reset()` returns
(rollout.py:269→278) and `end_trial()` closes it in the per-trial `finally`,
so in the normal path footer-active is the window where Esc ends the episode.
One path breaks that today: when `operator_input.begin_trial()` or `.poll()`
raises, rollout disables the console for the rest of the trial
(`console_ok = False`) but leaves the footer up, so the hint (yam's today,
ours after this plan) advertises a dead gesture for the remainder of the
trial. Fix in the same PR: at both disable sites (the `begin_trial()` raise
and the `poll()` raise), rollout also calls the duck-typed best-effort
`end_trial()` it already uses in the `finally` (idempotent, guarded by
`_footer_active`), restoring the terminal and dropping subsequent statuses to
the hint-less plain path. Two consequences handled with it:

- **Prompt hygiene.** After an early `end_trial()`, subsequent plain-path
  ticks leave `_status_open` sticky, and today nothing closes it before the
  verdict prompt or the next trial's readiness gate — the prompt would print
  appended to the leftover ticker text. `prompt_verdict()` and `gate()` gain
  an idempotent `self.status(None)` first (plain-close is already idempotent),
  pinned by tests. This also hardens the pre-existing plain-fallback mode.
- **Contract note.** The duck-typed `end_trial` contract widens from "called
  once in the per-trial finally" to "may be called mid-trial and again in the
  finally"; the `end_trial()` docstring states the idempotency requirement.

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
  Accepted, deliberately: for roughly the first second of each trial (until
  the first `_emit_status` tick) the screen shows no gesture hint beyond the
  once-per-run usage line — the hint arrives with the first footer tick.
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
4. Multi-pipe, non-gesture tail: `status("t = 4s | left arm ok")` renders
   `t = 4s | left arm ok | Esc ends the episode` (covers the sep-present /
   phrase-absent branch).
5. `status("")` renders the bare hint.
6. Width: at a width fitting the stripped line but not the composed one, the
   hint is dropped and the *stripped* line shown un-clipped — including for a
   stale input like `t = 4s | Enter ends the episode`, which must clip to
   `t = 4s`, never re-expose "Enter"; at a width smaller than the stripped
   line, existing tail-clip applies to the stripped line.
7. Plain mode: `status("parking")` renders without the hint.
8. Footer→plain transition: after `end_trial()`, a plain `status()` +
   `write_line` redraw never shows the hint (pins the structural guarantee).
9. Degraded console, both sites: a rollout whose operator input raises on
   `poll()`, and one whose input raises on `begin_trial()`, each call
   `end_trial()` at disable time — footer closed, later statuses hint-less
   (extends the existing rollout console-degradation tests).
10. Prompt hygiene: with a sticky plain status line open, `prompt_verdict()`
    and `gate()` close it before prompting (assert the written bytes end the
    status line before the prompt text).
11. Existing `status(None)` close and repeated-status tests keep passing
    unchanged (render-time composition leaves storage semantics untouched).

Repo gates: ruff, ruff format, mypy strict (src+tests), pytest --cov at 100%
with branch coverage.

## Docs

- `docs/guide/cli.md`: note the footer status line always carries the
  framework-appended end-gesture hint, so embodiment status text never goes
  stale (short prose near the existing Esc gesture section).
- `src/inspect_robots/CLAUDE.md`: extend the `session.py` row (hint
  composition + replacement rule, prompt-hygiene close) and the `rollout.py`
  row (footer closed at console-disable time; `end_trial` may fire twice).
- `CHANGELOG.md`: Changed entry under Unreleased referencing this plan, #345,
  and the yam drift incident that motivated it.

## Release sequencing

1. Merge core PR, cut core minor (0.49.0): the appended hint is a visible
   behavior change for every footer-mode plugin status.
2. yam PR: drop gesture clauses, floor `>=0.49`, lock refresh, cut yam minor.
   Skew matrix: new core + yam 0.28.0 → replace-rule dedups the ticker
   (identical render) and the banner already says Esc; new core + yam ≤0.27.x
   → the stale "Enter" *ticker* clause is corrected from the first footer
   tick, but the once-per-trial *banner* still shows "Enter ends the episode"
   on the plain path, out of the composer's reach — a residual contradiction
   resolved only by upgrading yam (accepted: core cannot rewrite plain-path
   text without breaking the footer-window truth condition); new yam + old
   core → prevented by the floor bump.
