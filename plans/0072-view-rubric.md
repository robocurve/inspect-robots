# view: render the generated rubric on scene cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the grading rubric in `inspect-robots view` reports (issue
#387). Plan 0070 persists a generated rubric at the per-scene sample's
`scene_metadata["rubric"]` and plan 0069's `vlm` grader judges against it,
but the HTML report never shows it: a reader of an auto-task report sees
the instruction and the verdict with no way to tell what the judge was
asked to check.

**Architecture:** one rendering addition in `_html.py`'s `_scene_section`:
when `scene.scene_metadata.get("rubric")` is a non-blank string, a labeled
rubric block renders directly after the instruction paragraph, escaped once
at interpolation exactly like the instruction, with pre-wrap styling.

**Tech stack:** the existing dependency-free HTML renderer; no new deps;
mypy strict; pytest at 100% coverage.

**Spec:** issue #387 + this plan (the plan is the spec, per repo convention).

**Critique:** round 1 (fresh-context subagent) found 1 major + 2 minor,
all fixed in this revision: live/started snapshots never carry
`scene_metadata` (LiveLogSink drops it, like the instruction), so the
live-page claim was corrected and the sink extension moved out of scope;
no summary/details CSS (the global dropdown styles already apply);
docs target corrected to cli.md's view section. The critic verified all
other claims against the code and blessed the fixes as specified.

## Global constraints

- Gates: `ruff check .`, `ruff format --check .`, `mypy` (strict,
  src + tests), `pytest --cov` at 100%.
- Escaping rule of `_html.py`: everything escaped once at interpolation.
- User-facing text (the rendered label) follows the repo writing-style
  rule: no em dashes in prose.
- No public API change (`render_html` signature untouched); no log-schema
  change (the field already exists).

## Binding decisions

1. **Eligibility mirrors the operator grader's display rule.** The block
   renders iff `scene.scene_metadata.get("rubric")` is an instance of
   `str` and `.strip()` is truthy; anything else (absent key, empty log
   field default, non-string, whitespace) renders nothing. Older logs
   without `scene_metadata` deserialize to `{}` (plan 0070), so pre-0070
   logs render unchanged byte-for-byte.
2. **Markup and style.** After the `instruction` paragraph:

   ```html
   <details class="rubric"><summary>Rubric</summary>
   <p class="rubric-text">{escaped rubric}</p></details>
   ```

   Collapsed by default: scene cards are scanned in lists and the rubric
   is judge-facing reference text, not headline identity — the same reason
   raw transcripts are dropdowns. **No summary or details CSS is added**:
   the stylesheet's existing global `summary` rule (muted, bold,
   `cursor: pointer`) and `details` spacing already style every dropdown
   (transcript, wire, raw transcript), and the rubric must look like its
   siblings. The only CSS change is appending `.rubric-text` to the
   existing `.instruction, .error` pre-wrap selector so multi-line
   rubrics keep their line structure. The rubric text is escaped with the
   module's `_escape` at interpolation, never earlier.
3. **The renderer adds no state branch; live pages show it only once the
   canonical log lands.** `_scene_section` renders the block for any
   sample whose `scene_metadata` qualifies, in every page state. But
   `LiveLogSink` snapshots never carry `scene_metadata` (its
   `on_trial_start` receives only `(scene_id, epoch)` and its `_samples()`
   builds every snapshot sample without it — the same reason live pages
   already show no instruction), so on a real `--serve` running page the
   rubric appears when the completed log replaces the snapshot, matching
   the instruction's existing behavior. Extending `LiveLogSink` to carry
   `instruction`/`scene_metadata` is a real gap but a separate sink-schema
   change, out of scope here.

## Tasks

- [ ] **1. Render + style.** Decision 2: the conditional block in
  `_scene_section` plus the two CSS additions in the stylesheet string.
- [ ] **2. Tests.** In the existing `_html` test module: a log whose
  sample carries a multi-line rubric renders the `<details class="rubric">`
  block with the escaped text and preserved line content (assert an
  HTML-special character in the rubric arrives escaped); a sample without
  the key, with a blank string, and with a non-string value each render no
  `class="rubric"` occurrence; a renderer-robustness test (explicitly
  labeled as such — production started-pages never carry the field) checks
  a synthetic `status="started"` log with `scene_metadata` set still
  renders the block. Extend the existing escaped-exactly-once test's
  fixture with a rubric containing an HTML-special payload and bump its
  occurrence count rather than duplicating fixtures.
- [ ] **3. Docs + module map.** `docs/guide/cli.md`: one sentence in the
  `inspect-robots view` section (report contents enumeration) noting the
  rubric dropdown on scene cards of auto-task logs. Not `live-view.md` —
  live pages never carry the field (decision 3). Update the `_html.py`
  row in `src/inspect_robots/CLAUDE.md`.
- [ ] **4. Gates + PR.** `ruff check .`, `ruff format --check .`,
  `uv run mypy`, `uv run pytest --cov` (100%). Push, CI green, fresh-eye
  review loop, merge (closes #387).

## Out of scope

- Extending `LiveLogSink` snapshots to carry `instruction`/`scene_metadata`
  (a real gap — live pages show neither — but a separate sink-schema
  change).
- Rendering other `scene_metadata` keys (provenance under `taskgen` stays
  log-only).
- Any change to `inspect` terminal output or the markdown summarizer.
- Log-schema or renderer-API changes.
