# 0034 — Embodiment-contributed guardrails

Issue: #232. Sibling: inspect-robots-yam#93 / its plan 0017.

## Problem

The default CLI safety chain (`_build_guardrails`, plan 0008 §3e) is fixed:
bounds clamp + per-step delta limit. Both are generic — they know the action
space, not the robot. Embodiment plugins can ship materially stronger,
embodiment-specific approvers: inspect-robots-yam's MuJoCo collision guardrail
(yam #85/#86) predicts bimanual self-, cross-arm, and table collisions and
holds at the last safe pose — complete, tested, and released. But no run path
can activate it:

- `_build_guardrails` builds only the two generic approvers; nothing consults
  the embodiment.
- The agent plugin's `pre_check` hook (#210) is programmatic-only by design
  ("-P CLI flags cannot carry callables") — and it protects only the agent
  policy, not scripted or learned policies, and sits *above* the approver
  layer rather than below it.

So a bimanual rig runs `clamp + delta-limit` while a purpose-built collision
guardrail sits unreachable in the installed wheel. The gap is a missing seam,
not a missing feature.

## Design

One optional, duck-typed embodiment method, consulted by `_build_guardrails`,
appending embodiment-contributed approvers to the default chain. Contributions
are **additive only**: the generic clamp + delta-limit stay exactly as they
are, in front, so the CLI is never less protective with a contribution than
without one. `--disable-guardrails` continues to disable the entire chain,
contributions included.

### 1. `GuardrailContribution` (approver.py)

```python
@dataclass(frozen=True)
class GuardrailContribution:
    """Approvers an embodiment adds to the CLI's default guardrail chain.

    ``approvers`` pairs a short display name (shown in the ``guardrails:``
    banner, e.g. ``"yam-collision"``) with the approver to append.
    ``warnings`` name contributions the embodiment declined to make and why
    (mode unsupported, optional dependency missing) — printed like the
    builder's own skip warnings, so a declined contribution is always visible.
    """

    approvers: tuple[tuple[str, Approver], ...] = ()
    warnings: tuple[str, ...] = ()
```

Validation in `__post_init__`: display names must be non-empty and contain no
newline (they land in a one-line banner); approvers must satisfy the
`Approver` protocol shape (`callable review` attribute), checked with the same
lenience the chain itself applies. Exported from `inspect_robots.approver`.

### 2. The embodiment method (documented protocol, duck-typed)

```python
def contribute_guardrails(self, action_space: Box) -> GuardrailContribution: ...
```

- Optional: absence means no contribution (every existing embodiment keeps
  working, zero migration).
- Instance method, not ClassVar: whether and how to contribute depends on the
  embodiment's runtime config (control interface, user opt-out flag).
- Documented on the `Embodiment` protocol in `types.py` alongside `info`, with
  the conformance suite gaining a shape check: *if* the attribute exists it
  must be callable, and its result must be a `GuardrailContribution`.
- The contract mirrors `_build_guardrails`' own degrade philosophy: an
  embodiment that cannot contribute in the current mode returns a warning
  entry, never raises for "not applicable". Raising is reserved for actual
  bugs and propagates — a safety component that fails unexpectedly must be
  loud, not silently skipped.

### 3. `_build_guardrails` (cli.py)

Signature gains the embodiment:

```python
def _build_guardrails(
    space: Box, max_action_delta: float | None, embodiment: Embodiment | None = None
) -> tuple[Approver, list[str], list[str]]:
```

After the existing clamp/delta assembly:

- `hook = getattr(embodiment, "contribute_guardrails", None)`; if absent →
  done (exact current behavior).
- If present but not callable → warning
  `"embodiment guardrails skipped: contribute_guardrails is not callable"`.
- If calling it returns a non-`GuardrailContribution` → warning naming the
  actual type (defensive: a plugin built against a newer/older core must
  degrade visibly, not crash the banner).
- Otherwise append each contributed approver to `parts`, its display name to
  `active`, and extend `warnings` with the contribution's warnings.

`_build_and_announce_guardrails` passes `resolved.embodiment` through from
both call sites (single-task and eval-set paths); the banner then reads e.g.
`guardrails: clamp + delta-limit + yam-collision`. The `AutoApprover`
fallback branch ("no guardrails are active…") is unreachable when a
contribution exists, and the fallback message stays accurate: contributions
append to `parts`, so `parts` non-empty ⇒ chain built.

`--disable-guardrails` already returns before `_build_guardrails` is invoked;
contributions are therefore disabled with everything else, and the
`--max-action-delta`/`--disable-guardrails` conflict check is untouched.

### 4. Ordering

Chain order is `clamp → delta-limit → contributions…` (declaration order
within a contribution). Contributed approvers see targets already clamped and
rate-limited — the cheap generic gates run first, the expensive
embodiment-specific ones last, and a contribution can rely on its input being
in-bounds.

## Not in scope

- Contributions replacing or reordering the generic chain (additive only).
- A CLI flag to disable a single contribution (opt-out belongs to the
  embodiment's own config, where the wizard can interview it — see yam plan
  0017).
- Plumbing contributions into programmatic `eval()` callers: `eval()` already
  accepts any `approver`; callers compose their own chain.

## Tests

`tests/test_cli_guardrails.py` (extending the existing `_build_guardrails`
coverage):

- No method → identical chain and banner to today (regression).
- Contribution with one approver → appended after delta-limit, name in
  `active`, banner string exact.
- Contribution with warnings only → warnings printed, chain unchanged.
- Non-callable attribute / wrong return type → visible warning, generic chain
  intact.
- Contribution while clamp+delta are both skipped (unbounded space) → chain is
  contribution-only, `AutoApprover` fallback not taken.
- `--disable-guardrails` with a contributing embodiment → no approver, no
  contribution call (method not invoked; assert via spy).
- Conformance: embodiment with `contribute_guardrails` returning a wrong type
  fails the shape check; absent attribute passes.

## Release

Minor bump (new public API `GuardrailContribution`): core `0.31.0`. The yam
plugin's floor bumps to `>=0.31` in its own PR (yam #93), which imports the
dataclass.
