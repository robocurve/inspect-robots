# Paired robot pilot for adaptive Fable 5.2

Run five matched pairs to compare always-high reasoning and full camera views
against per-turn effort and event-triggered wrist detail. Both conditions use
the same model at fast speed. Use the pilot to check for scoring, control,
and thermal regressions before a larger trial.

## Conditions

| Condition | Reasoning effort | Cameras |
| --- | --- | --- |
| Baseline | `effort=high`, `effort_policy=fixed` | `camera_history_policy=full` |
| Adaptive | `effort=high`, `effort_policy=event` | `camera_history_policy=transition_witness` |

The adaptive policy uses medium effort for routine decisions and high effort
on the first observation, on a gripper command or readback change of at least
0.12 of a normalized stroke, on recovery, and when gripper telemetry is
unavailable. It also uses high effort every four observations, counted from
the initial observation. Tool-only rounds do not advance this count.
The top view keeps its original size. The `left_cam` and `right_cam` color
views have a 112 px maximum edge on routine observations and keep their
original size at events. Rendered depth keeps its original size. If the
previous observation used smaller wrist images, a
gripper event can attach the preceding full wrist views alongside the new
ones. A recovery after a missed or rejected tool call can append the current
full wrist views and raise effort before the model retries. Review the
pre/post frames to verify a grasp.

Both conditions set `image_horizon=none` to keep earlier messages intact.
The effort policy appends an effort marker when the level changes and keeps
the top-level effort at high. Use the first-party Claude Messages endpoint
with access to fast mode and the [per-message effort beta](https://platform.claude.com/docs/en/build-with-claude/effort-per-message).

## Setup and pairing

1. Pin one Inspect Robots commit, the rig and task plugin revisions, and the
   rubric and grader. Use the same configuration, camera calibration,
   collision limits, call and step limits, timeouts, and control interface
   for both arms. Follow the [rig health check](rig-reference.md#yam-health-checks)
   and the rig's normal supervised operating procedure.
2. Preselect five registered single-scene task configurations. For
   [RoboDojo-RC](https://github.com/robocurve/robodojo-rc#running-on-robots),
   use the installed task IDs `robodojo-rc/cap_pen`, `robodojo-rc/stand_up_bottles`,
   `robodojo-rc/pack_and_pour_fruit`, `robodojo-rc/stack_bowls`, and
   `robodojo-rc/store_in_safe`. These cover fine alignment, repeated grasps,
   and occlusion. Verify the registrations with `inspect-robots list tasks`.
3. Record the five layout and seed combinations before running. Randomly
   choose three pairs to run adaptive first and two to run baseline first,
   or swap those counts by a coin flip. Within each pair use the same rig,
   seed, scene, initial object poses, and starting temperature band. Mark
   object positions or take a reset photo so the operator can reproduce the
   arrangement. Reestablish that starting temperature band before each run.
4. Keep a manifest with pair ID, order, task and seed, rig ID, starting and
   peak temperatures, and the two returned log paths. Before the first run,
   agree on the score criterion, acceptable control delay, and the rig's
   existing hardware safety limits.

## Commands

Run from the rig's environment with the benchmark and YAM plugins installed.
Set `RIG_CONFIG` to the rig's absolute config path, `LOG_ROOT` to the
pilot log directory, and `GRADER` to the pinned rubric grader. For RoboDojo-RC,
use `robodojo_astra` for the published-style judge or `robodojo_operator` for
manual rubric scoring. For each pair set `TASK`, `PAIR`, and `SEED` from the
manifest, then run the two arms in the assigned order. Supply any task
parameters required to select that single scene in both calls. The example
uses the reference 40-call cap, rendered depth, and Cartesian YAM control.
Use the task's configured step cap and the operator-approved motion limits.

```bash
run_arm() {
  arm="$1"
  shift
  inspect-robots run --config "$RIG_CONFIG" --task "$TASK" \
    --policy agent --embodiment yam_arms --grader "$GRADER" \
    -E control_interface=eef_pos -E auto_start=false \
    --seed "$SEED" --epochs 1 --store-frames --log-dir "$LOG_ROOT/$PAIR/$arm" \
    -P model=anthropic/claude-fable-5-2 -P wire=messages -P effort=high \
    -P speed=fast -P max_llm_calls=40 -P max_output_tokens=128000 \
    -P images=always -P image_horizon=none -P depth=render -P wire_capture=true "$@"
}
```

For the baseline run use

```bash
run_arm baseline -P effort_policy=fixed -P camera_history_policy=full
```

For the adaptive run use

```bash
run_arm adaptive -P effort_policy=event -P camera_history_policy=transition_witness
```

Reset the scene and check the starting temperature band between the two calls.

## Record and compare

| Measure | Where to get it |
| --- | --- |
| Rubric score, scored completion, and termination | Eval log. Apply the same rubric to the same stored camera views and blind the grader to condition where feasible. Keep stopped, errored, and unscored trials visible. |
| Duration and policy latency | Eval log `stats.duration_s` for total run time, including setup and grading. Use the record's `chunks.jsonl` field `wall_s` for median and p95 time in policy `act()`. Use rig telemetry for rollout duration or time from camera capture to motor execution. |
| API latency, usage, retries, and cost | Read `duration_s`, `attempt`, `status`, and raw response `usage` from the wire capture. Count each attempt where usage is reported. Keep base input, cache reads, cache writes, and output separate. Price at the applicable fast rates and mark estimated totals. |
| Effective effort and camera detail | Read each wire request's latest effort marker, or top-level effort before the first marker. Match available `trial_metadata.observation_events` entries by zero-based `call` for `step`, `reason`, and `effort`. Check camera labels and blob dimensions for the views sent. |
| Protective stops, thermal stops, and temperatures | Use the termination reason and the rig's telemetry. Record start and peak temperature and any intervention in the manifest. |

Use [wire capture](logging-and-rerun.md#wire-capture) and [stored trial records](logging-and-rerun.md#trial-records)
to trace suspect decisions back to the request, the original frames, and the
executed actions. Use the [Fable 5.2 prices](https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2#pricing)
and the [fast-mode caching multipliers](https://platform.claude.com/docs/en/build-with-claude/fast-mode#pricing)
when provider billing is unavailable.

Report the adaptive-minus-baseline difference within each pair for score,
cost, duration, and stop outcome. Review any treatment-only protective
or thermal stop, material score loss, or control-delay breach before
expanding the pilot. Report API spend and total time for both successful
and unsuccessful trials. Include rig and operator time at the team's
agreed rate when comparing cost per successful trial.

Five pairs can expose practical regressions. Use a larger matched study
with a prespecified margin for a claim about score preservation. To measure
the effect of each option, compare all four combinations of
`fixed`/`event` effort and `full`/`transition_witness` camera policy at the
same speed and settings.
