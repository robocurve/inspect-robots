"""Run a deterministic fixed-trials Actuate conference campaign.

Run with:  python examples/actuate/run_trials.py [-- extra inspect-robots run args]
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run
from _thermal import config_channels

# Scored evals each test-taker must complete; booth-editable.
TRIALS_PER_MODEL = 10
# Consecutive logged evals without a score before halting; booth-editable.
UNSCORED_STREAK_STOP = 10
# Wall-clock kill switch per eval; booth-editable. OpenRouter keeps feeding
# keepalive bytes while a wedged upstream provider grinds, so the agent
# plugin's httpx read timeout never fires and a single eval can hang a rig's
# whole overnight half. A killed eval produces no final log, so the existing
# failure-streak machinery counts and reports it.
EVAL_TIMEOUT_S = 1800
# Grace between the interrupt and the hard kill. SIGINT gives the eval its
# normal teardown (park the arms, release cameras); a bare SIGKILL left a
# rig-2 arm slumped past its joint limit and wedged the cameras mid-campaign
# on 2026-08-19, failing every later eval at bring-up.
EVAL_KILL_GRACE_S = 120


def _run_eval_with_watchdog(cmd: list[str]) -> bool:
    """Run one eval; True if it finished, False if the watchdog killed it.

    The eval runs in its own process group. On timeout the whole group gets
    SIGINT first so the embodiment tears down cleanly, then SIGKILL after
    EVAL_KILL_GRACE_S for a process too wedged to honor the interrupt.
    """
    process = subprocess.Popen(cmd, start_new_session=True)
    try:
        process.wait(timeout=EVAL_TIMEOUT_S)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=EVAL_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass
    return False


def _trial_scores(test_taker_names: list[str]) -> dict[str, list[float]]:
    """Return finite scored evals for each requested test-taker."""
    observations: dict[str, list[float]] = {name: [] for name in test_taker_names}
    try:
        lines = run.RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    for line in lines:
        try:
            result = json.loads(line)
            name = result["roles"]["test_taker"]
            score = result.get("score")
            numeric_score = float(score) if score is not None else None
            if name in observations and numeric_score is not None and math.isfinite(numeric_score):
                observations[name].append(numeric_score)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return observations


def run_campaign(
    rig_config: Path,
    state_dir: Path,
    logs_dir: Path,
    trials_per_model: int,
    argv: list[str],
) -> None:
    """Run a fixed-trials campaign with the requested rig and output directories."""
    run.RIG_CONFIG = rig_config
    run.STATE_DIR = state_dir
    run.LOGS_DIR = logs_dir
    run.STATUS_PATH = state_dir / "status.json"
    run.RESULTS_PATH = state_dir / "results.jsonl"
    run.MEDIA_DIR = state_dir / "media"

    if not run.RIG_CONFIG.is_file():
        raise SystemExit(
            f"rig config not found: {run.RIG_CONFIG}\n"
            "fix: edit RIG_CONFIG in run.py or in the launching rig script"
        )
    # One campaign per rig config, enforced up front: a second campaign on the
    # same config would probe the live CAN bus mid-eval and burn a failure
    # streak before the downstream flock claim guard surfaced the conflict.
    # The handle stays open for the campaign's lifetime; the OS releases the
    # lock on exit however the process dies.
    config_key = hashlib.sha1(str(run.RIG_CONFIG.resolve()).encode()).hexdigest()[:12]
    lock_path = state_dir.parent / f".campaign-{config_key}.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"another campaign is already running against {run.RIG_CONFIG}; "
            "stop it first (each rig config supports one campaign at a time)"
        ) from None
    # Resolved once from RIG_CONFIG only: a --config override after "--"
    # re-routes the eval but not the gate's probe channels, so on a rig with
    # different channel names the gate degrades to per-round warnings.
    channels = config_channels(run.RIG_CONFIG)
    run.STATE_DIR.mkdir(parents=True, exist_ok=True)
    run.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    extra_args = run._extra_args(argv)
    taskers = run._eligible("tasker")
    test_takers = run._eligible("test_taker")
    graders = run._eligible("grader")
    if len(taskers) != 1 or len(graders) != 1:
        raise SystemExit(
            "the fixed-trials runner requires pinned tasker and grader roles; "
            "fix the roster in _roster.py"
        )
    if not test_takers:
        raise SystemExit("no eligible test-takers; fix the roster in _roster.py")
    eval_index = run._next_eval_index()
    failure_streak = 0
    unscored_streak = 0
    gate_state: dict[str, Any] = {"channels": channels, "disabled": False}
    tasker_name, tasker = taskers[0]
    grader_name, grader = graders[0]
    test_taker_names = [name for name, _model in test_takers]

    try:
        while True:
            scores_by_model = _trial_scores(test_taker_names)
            remaining = [
                entry for entry in test_takers if len(scores_by_model[entry[0]]) < trials_per_model
            ]
            if not remaining:
                for name, _model in test_takers:
                    scores = scores_by_model[name]
                    mean_score = sum(scores) / len(scores)
                    print(f"{name}: n={len(scores)}, mean={mean_score:.2f}", flush=True)
                return

            # Randomized block design: fewest-completed keeps every model to
            # one trial per round of the roster, and drawing randomly among
            # the tied models randomizes each round's order, so a
            # position-in-round effect (coolest motors right after a cooling
            # pause, freshest scene reset) cannot align with any one model.
            fewest = min(len(scores_by_model[entry[0]]) for entry in remaining)
            block = [entry for entry in remaining if len(scores_by_model[entry[0]]) == fewest]
            test_taker_name, test_taker = random.choice(block)
            completed_trials = len(scores_by_model[test_taker_name])

            run._thermal_gate(eval_index, gate_state)
            roles = {
                "tasker": tasker_name,
                "test_taker": test_taker_name,
                "grader": grader_name,
            }
            models = {
                "tasker": tasker["model"],
                "test_taker": test_taker["model"],
                "grader": grader["model"],
            }
            run._write_status(
                {
                    "eval_index": eval_index,
                    "started_at": run._utc_now(),
                    "roles": roles,
                    "models": models,
                }
            )
            print(
                f"Eval {eval_index}: tasker={tasker_name}, "
                f"test-taker={test_taker_name}, grader={grader_name}; "
                f"trial {completed_trials + 1}/{trials_per_model} for {test_taker_name}",
                flush=True,
            )

            existing_logs = run._final_log_paths()
            finished = _run_eval_with_watchdog(run._command(tasker, test_taker, grader, extra_args))
            if not finished:
                print(
                    f"eval exceeded {EVAL_TIMEOUT_S}s; interrupted for a clean "
                    "park-and-release teardown (hard kill only if that hangs "
                    f"{EVAL_KILL_GRACE_S}s)",
                    flush=True,
                )

            log_path = run._newest_final_log(existing_logs)
            score, task = run._log_outcome(log_path)
            clips = run._render_clips(log_path, eval_index)
            run._append_result(
                {
                    "ts": run._utc_now(),
                    "eval_index": eval_index,
                    "roles": roles,
                    "models": models,
                    "task": task,
                    "score": score,
                    "log": log_path.name if log_path else None,
                    "clips": clips,
                }
            )
            score_text = f"{score:.2f}" if score is not None else "unscored"
            log_text = log_path.name if log_path else "no log"
            print(f"Eval {eval_index} complete: score={score_text}, log={log_text}")

            # Only a scored (finite) eval resets the unscored streak: selection
            # retries the same lagging model, so a rig or model alternating
            # between no-log and logged-but-unscored failures must not keep
            # both counters at zero forever. NaN scores count as unscored for
            # the same reason (they never count as trials).
            scored = score is not None and math.isfinite(score)
            if log_path is None:
                failure_streak += 1
            elif not scored:
                failure_streak = 0
                unscored_streak += 1
            else:
                failure_streak = 0
                unscored_streak = 0
            if failure_streak >= run.FAILURE_STREAK_STOP:
                raise SystemExit(
                    f"{failure_streak} consecutive evals produced no log; the rig "
                    "is likely faulted (check arms, CAN, cameras). Fix it and "
                    "relaunch."
                )
            if unscored_streak >= UNSCORED_STREAK_STOP:
                raise SystemExit(
                    f"{unscored_streak} consecutive evals produced logs but no score; "
                    f"the grader ({grader_name}) or the repeatedly drawn test-taker "
                    f"({test_taker_name}) is likely broken. Fix it and relaunch."
                )
            eval_index += 1
            counts = _trial_scores(test_taker_names)
            if all(len(counts[name]) >= trials_per_model for name in test_taker_names):
                # Campaign complete: skip the pause and let the top-of-loop
                # check print the summary (a Ctrl-C during a pointless final
                # sleep would eat it). Clear the roles from the status file so
                # the monitor stops showing the last eval as current.
                run._write_status({"eval_index": eval_index - 1})
                continue
            pause = run.PAUSE_S if log_path is not None else run.FAILURE_PAUSE_S
            print(f"next eval in {pause}s (Ctrl-C to stop)", flush=True)
            time.sleep(pause)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


def main() -> None:
    """Run the fixed-trials campaign with the single-rig defaults."""
    run_campaign(run.RIG_CONFIG, run.STATE_DIR, run.LOGS_DIR, TRIALS_PER_MODEL, sys.argv[1:])


if __name__ == "__main__":
    main()
