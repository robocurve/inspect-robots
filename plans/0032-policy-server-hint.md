# 0032 — Actionable hint when a policy's action server is unreachable

Closes robocurve/inspect-robots#219.

## Problem

Server-backed policies (e.g. `molmoact2` from inspect-robots-yam) are thin
HTTP clients; the model server is a separate process the user starts
themselves. When that server is down, a run fails with raw urllib3 spew:

```
  [error] scene-0: PolicyError: HTTPConnectionPool(host='127.0.0.1', port=8202):
  Max retries exceeded with url: /act (Caused by NewConnectionError(...
  [Errno 111] Connection refused))
```

Nothing tells the user a separate server exists, which URL was expected, or
how to start it. Observed live on the yam rig 2026-07-29: the arms homed,
ran an episode, and parked before the user learned the server was never up
(the pre-homing preflight is a separate follow-up; this plan is the
error-time hint only).

## Design

Enrich the `PolicyError` message at wrap time in the rollout, where both the
original exception and the policy instance are in hand. The hint becomes part
of the recorded error string, so it persists into the log JSON and every
surface that renders it (`run` summary, `inspect`, HTML viewer) with zero
extra plumbing.

### Detection: `_connection_failure(exc)` (`src/inspect_robots/rollout.py`)

Walk the exception's `__cause__`/`__context__` chain (cycle-safe via an
`id()` seen-set). Return `True` when any node is an instance of builtin
`ConnectionError` (covers `ConnectionRefusedError`, `ConnectionResetError`)
**or** its type name is one of `{"ConnectionError", "NewConnectionError",
"MaxRetryError"}` — matching `requests`/`urllib3` types by name keeps core
dependency-free. Timeouts are deliberately excluded: a slow server is a
different failure from an absent one, and a timeout hint saying "is the
server running?" would mislead.

### Wrapping: `_policy_error(policy, exc)` (`src/inspect_robots/rollout.py`)

Replace the two generic-exception wrap sites — `rollout.py:207` (policy
`reset`) and `rollout.py:227` (`controller.next_action`) — with a helper:

```python
def _policy_error(policy: Policy, exc: Exception) -> PolicyError:
    """Wrap a generic policy exception, appending a hint on connection failures."""
    message = str(exc)
    if _connection_failure(exc):
        url = getattr(policy, "server_url", None)
        where = f" at {url}" if url else ""
        message += (
            f"\nhint: policy {policy.info.name!r} could not connect to its "
            f"action server{where} — is the server running? Start it, then rerun."
        )
        remedy = getattr(policy, "remedy", None)
        if remedy:
            message += f"\nhint: {remedy}"
    return PolicyError(message)
```

The dimension-mismatch `PolicyError` at `rollout.py:254` is untouched (not a
connection failure by construction).

### Policy-declared remedy (duck-typed protocol)

Policies MAY expose two optional string attributes, read via `getattr`:

- `server_url` — the endpoint the policy talks to (inspect-robots-yam's
  `molmoact2` config already has this field).
- `remedy` — a one-line, policy-specific pointer, e.g.
  `"start the MolmoAct2 YAM server (serves :8202), then rerun"`.

No change to the `Policy` protocol or `PolicyConfig` — absence of the
attributes degrades to the generic hint. Populating them for `molmoact2` is
an inspect-robots-yam follow-up PR.

## Tests (`tests/test_rollout.py`)

Fakes only; no `requests` import.

1. Policy `reset` raises builtin `ConnectionRefusedError` → recorded
   `PolicyError` message contains the generic hint (no URL segment).
2. Policy exposing `server_url` and `remedy`, raising a fake
   `requests`-style chain (locally defined classes *named*
   `ConnectionError`/`NewConnectionError`, not builtins, linked with
   `raise ... from ...`) at step time → message contains the URL and the
   remedy line.
3. Policy raising `ValueError` → message contains no `hint:`.
4. `_connection_failure` on a self-referential `__context__` cycle
   terminates and returns `False`.
5. Timeout-shaped exception (type name `ReadTimeout`/`Timeout`) → no hint.

## Docs / changelog

- `CHANGELOG.md` → `[Unreleased] / Added`: one entry for the hint.
- `PolicyError` docstring (`src/inspect_robots/errors.py:54`): note that
  connection-level failures carry a remediation hint in the message.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
`uv run pytest --cov` (must stay at 100%). Baseline before changes: 876
passed, 3 skipped.

## Out of scope

- inspect-robots-yam `molmoact2` remedy population (follow-up plugin PR).
- Probing `server_url` in the `setup` runtime-requirements checklist
  (`_setup.py:171`).
- Pre-homing liveness preflight (separate safety-focused issue, per #219).
