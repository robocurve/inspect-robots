# 0062 — Session-owned end-gesture hint on footer status lines

- **Status:** draft
- **Issue:** #345
- **Critique rounds:** (recorded here per repo convention)

## Problem

The episode-end gesture (Esc, `/stop`) is owned by the framework console
(`console.py` grammar + `session.py` footer editor, plan 0056). But the status
text that *describes* the gesture is composed by embodiment plugins: yam's
per-second ticker (`t = 4s / 1200s | Esc ends the episode`) and running banner
hardcode gesture prose. When 0.47 moved the gesture from Enter to Esc, core's
own strings updated atomically with the behavior while yam's description went
stale on the rigs (yam#120/#121). Prose duplicated across an API boundary
drifts; the fix is to move the prose to the side that owns the behavior.

## Design

`OperatorSession` appends the end-gesture hint to footer-mode status lines
itself. Plugins report rig state only (`t = 4s / 1200s`); they never mention
the gesture.

### Rule

In `_footer_status(line)`, when `line is not None`:

```python
if _END_HINT_PHRASE not in line:
    line = f"{line} | {_END_HINT}"
self._status_line = line
```

with module-private constants:

```python
_END_HINT = "Esc ends the episode"
_END_HINT_PHRASE = "ends the episode"
```

- **Footer mode only.** `begin_trial()` opens the footer *after*
  `embodiment.reset()` returns (rollout.py:269→278) and `end_trial()` closes it
  in the per-trial `finally`, so footer-active is exactly the window where a
  bare Esc keypress ends the episode. Plain-path statuses (pre-reset homing,
  post-trial parking, non-TTY fallback) get no hint: outside the window the
  hint would be false, and in the non-TTY fallback the gesture differs (Esc
  then Enter) and the console usage line already covers it.
- **Skew guard.** The `_END_HINT_PHRASE` substring check keeps the footer sane
  when an old plugin (yam ≤0.28.0) still appends its own gesture clause: both
  its ticker suffix ("Esc ends the episode") and its banner clause ("Esc (or
  /stop) ends the episode") contain the phrase, so no double hint is rendered.
  New plugins never trip it. The guard is deliberately a substring test, not
  parsing: worst case a false positive suppresses the hint on a status that
  happens to say "ends the episode", which is the text saying what the hint
  would say.
- **Storage, not render-time.** The suffixed line is stored in
  `self._status_line`, so the `write_line` scrollback redraw and the
  `_clipped_status_text()` path stay single-source. `_clip_tail` keeps the
  trailing tail on narrow terminals, so the hint survives clipping (same
  clipping semantics as today).
- The hint is mode-independent: Esc ends the episode whether or not the policy
  accepts operator messages, so the session does not need a structured
  messages flag and `console_usage` stays as is.

### Out of scope

- The plain-path single-line ticker keeps rendering exactly what the plugin
  set. No behavior change outside footer mode.
- The console usage reminder (`USAGE` / `USAGE_END_ONLY`) is unchanged.
- Gate prompts ("press Enter…") are readiness gates, still correctly
  Enter-driven, untouched.
- No public API change: constants stay module-private, `__all__` untouched,
  `tests/test_api_snapshot.py` unchanged.

## yam follow-up (separate PR in robocurve/inspect-robots-yam)

After this ships in a core release (call it 0.49):

- Deferred-mode ticker becomes `self._status(f"t = {span}")`.
- Deferred-mode banner drops its gesture clause:
  `f"Running: type a message + Enter to send feedback.{limit}"` (the message
  affordance is stable across core versions; only the end gesture drifted).
- Never-connected legacy path keeps its own "any key" text: that path reads
  raw stdin itself and never routes through an `OperatorSession`, so the
  boundary argument does not apply.
- `inspect-robots` floor rises to the release carrying this change, so a rig
  never runs hint-less (new yam + old core would otherwise show no end hint
  at all).

## Tests (core)

Extend `tests/test_session.py` footer coverage:

1. Footer-mode `status("t = 1s")` renders `t = 1s | Esc ends the episode` in
   the status row (assert on written bytes, matching existing footer render
   tests).
2. Skew guard: `status("t = 1s | Esc ends the episode")` and
   `status("Running: Esc (or /stop) ends the episode.")` render unchanged (no
   double hint).
3. Plain mode: `status("parking")` renders without the hint.
4. `write_line` during an open suffixed status redraws the suffixed line
   (single-source storage).
5. `status(None)` close behavior unchanged (existing tests keep passing).

Repo gates: ruff, ruff format, mypy strict (src+tests), pytest --cov at 100%
with branch coverage.

## Docs

- `docs/guide/cli.md`: note that the footer status line always carries the
  end-gesture hint appended by the framework, so embodiment status text never
  goes stale (short prose near the existing Esc gesture section).
- `src/inspect_robots/CLAUDE.md`: extend the `session.py` row.
- `CHANGELOG.md`: Changed entry under Unreleased referencing this plan, #345,
  and the yam drift incident that motivated it.

## Release sequencing

1. Merge core PR, cut core minor (0.49.0): the appended hint is a visible
   behavior change for every footer-mode plugin status.
2. yam PR: drop gesture clauses, floor `>=0.49`, lock refresh, cut yam minor.
   During the skew window (new core + yam 0.28.0) the guard suppresses
   doubling, so ordering is safe in both directions; the floor bump only
   protects the reverse skew (new yam + old core).
