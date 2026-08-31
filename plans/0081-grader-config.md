# 0081: Record the effective grader configuration in `EvalSpec`

Follow-up to #413, which #422 closed by adding `SceneResult.judgement_sources`.
That field says *which path* produced each verdict. This one says *what
configuration that path ran under*. Invited on #413: "recording grader
configuration (model, rubric, effort) in `EvalSpec` is a separate additive
change and is not in that PR; happy to take a PR for it".

## Problem

`EvalSpec` records `policy` + `policy_config` and `embodiment` +
`embodiment_info`, but nothing at all about the grader. Two gaps follow.

1. A log cannot say **whether a grader ran, or which one**. `judgement_sources`
   now names the path per trial, but a run where every trial errored, or where
   the VLM degraded to ungraded on every trial, records `None` throughout and
   is indistinguishable from `--grader none`.
2. For the `vlm` grader the log cannot say **what governed the judgement**. The
   model id, the run-level rubric, the camera cap and the effort level are all
   resolved inside `vlm_grader()` and then held privately on `_VLMGrader`. Two
   runs of the same task graded by different models, or against different
   rubrics, produce byte-identical `EvalSpec`s. The prompt cannot be rebuilt
   from the log, so a metric that moves between runs cannot be attributed.

The values are *resolved* rather than passed through, which is why the caller's
own arguments are not a substitute: `rubric_file` has already been read to text,
an omitted `rubric` has become `_DEFAULT_RUBRIC`, and `effort` has been through
the `_Unset` normalization from plan 0075.

## Design

Two additive fields on `EvalSpec` (`src/inspect_robots/log.py`), the exact
parallel of the `policy` / `policy_config` pair beside them:

```python
grader: str | None = None
grader_config: dict[str, Any] = field(default_factory=dict)
```

- `grader` is the grader's registry name, the value the `Grader` protocol
  already documents as "Registry name identifying this grader in logs and CLI
  output". `None` means the run graded nothing.
- `grader_config` is what the grader reports about itself, `{}` when it reports
  nothing.

### Reading the configuration

`Grader` is `@runtime_checkable` and `_grading_hook` gates on
`isinstance(grader, Grader)`, so adding a `config` member to the Protocol would
make every existing out-of-tree grader fail that check. The hook is therefore
**duck-typed and optional**, matching `Embodiment.bind_task` and the grader's
own `connect_session`:

```python
hook = getattr(grader, "config", None)
if not callable(hook):
    return grader.name, {}
return grader.name, cast("dict[str, Any]", dict(hook()))
```

A grader written before this plan keeps working and is still named in the log.

`_VLMGrader.config()` returns the five resolved values it actually applies:
`model`, `base_url`, `rubric`, `max_cameras`, `effort`. The API key is
deliberately absent; a log is not a place for a credential, and a test asserts
the key text appears nowhere in the written file.

### Where it is captured

`_grading_hook` previously returned only `grader.grade`, discarding the object
before `EvalSpec` is built. It now returns `(hook, grader)`.

`eval_set()` needs the same treatment for a non-obvious reason: it resolves the
grader once and passed only the bound `before_scoring` method down to `eval()`,
so every task's spec would have recorded no grader at all. It now hands the
grader itself down instead. Re-resolution per task is idempotent (an object
passes the isinstance check and yields the same bound method).

Live and final logs need no shared helper here, unlike plan 0080:
`bus.on_eval_start(spec)` and `EvalLog(eval=spec)` distribute one `EvalSpec`
instance, so `LiveLogSink` snapshots cannot disagree with the final log. A test
pins that.

## Decisions

1. **The rubric is stored as text, not a hash or a path.** The audit question
   from #413 is whether the prompt can be rebuilt, and a digest cannot rebuild
   it. A rubric may also arrive inline via `-G rubric=`, where there is no path
   to record. It is one string per run, not per trial.
2. **The stored rubric is the run-level one, and only that.** `_rubric_for()`
   lets a non-blank `scene.metadata["rubric"]` win per scene, and that value is
   already persisted in `SceneResult.scene_metadata`, rendered by `_html.py`.
   Copying it into `EvalSpec` would create a second source of truth for a
   per-scene fact on a run-level record. The field comment says so explicitly.
3. **`effort` is stored as the resolved value, keeping `None` distinct from
   `"none"`.** After plan 0075 these are different facts: `None` omits
   `reasoning_effort` so the provider default applies, `"none"` asks for the
   minimum. The type is `str | float | None`, and a float is recorded as sent
   rather than snapped to a level, matching `AgentPolicyConfig.effort`.
4. **`base_url` and `max_cameras` are included** even though the invitation
   named three fields. `AgentPolicyConfig`, the repo's other LLM-backed
   component config, records `base_url` for the same reason: a model id alone
   does not say which provider answered. `max_cameras` caps how many frames
   reach the prompt, so it changes what the judge saw.
5. **`api_key_env` is not included.** `AgentPolicyConfig` records it, but
   `_VLMGrader` does not retain it, and threading a new constructor parameter
   through to record the *name* of a variable (which can hold different keys on
   different days) buys little for the widening it costs.

## Not in scope

Surfacing grader configuration in `inspect-robots view` or the `summarize`
digest, the same line #422 drew for `judgement_sources`. Recording per-trial
grader state, which is a different record (`SceneResult`) from this one.

## Compatibility

Additive with defaults, no schema bump, consistent with every prior additive
field. `EvalSpec(**data["eval"])` fills both defaults for a log written before
they existed, and `test_v1_log_without_additive_fields_reads_back` covers it.
As with `judgement_sources`, the repo's guarantee is that newer Inspect Robots
reads older logs, not the reverse.
