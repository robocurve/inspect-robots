"""Run a no-key Relay demonstration with a scripted provider and the CubePick world.

Install with ``uv sync --all-packages --extra dev --extra relay --python 3.11``.
Run with ``uv run --no-sync python examples/relay_agent.py --repeats 10``.
This measures instrumentation behavior, not model quality or hardware safety.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
from inspect_robots_agent import LLMAgentPolicy

from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task


def run(path: Path, enabled: bool) -> tuple[Any, int, float]:
    """Reject one unknown tool, then reach the cube from the observation's state."""
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        messages = json.loads(request.content)["messages"]
        text = "\n".join(
            part["text"]
            for message in messages
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        )
        positions = {
            key: np.asarray(json.loads(re.findall(rf"state\[{key}\]: (\[[^\]]+\])", text)[-1]))
            for key in ("eef_pos", "cube_pos")
        }
        delta = positions["cube_pos"] - positions["eef_pos"]
        name = "unknown" if calls == 1 else "move_by"
        arguments = (
            {}
            if calls == 1
            else {
                "deltas": {"dx": float(delta[0]), "dy": float(delta[1])},
                "note": "reach the seeded cube",
            }
        )
        return httpx.Response(
            200,
            json={
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": f"call-{calls}", "name": name, "input": arguments}
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    policy = LLMAgentPolicy(
        model="scripted-demo",
        wire="messages",
        base_url="http://demo.invalid/v1",
        env={},
        transport=httpx.MockTransport(respond),
        relay=enabled,
    )
    task = Task(
        name="relay-recovery",
        scenes=[
            Scene(
                id="seed-0",
                instruction="reach the cube",
                init_seed=0,
            )
        ],
        scorer=success_at_end(),
        max_steps=20,
    )
    start = time.perf_counter()
    (log,) = eval(task, policy, CubePickEmbodiment(), seed=0, log_dir=str(path))
    return log, calls, time.perf_counter() - start


def main() -> None:
    """Compare paired trials and print the last trace's tool outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/relay-demo"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    base = args.log_dir / uuid4().hex
    timings: dict[bool, list[float]] = {False: [], True: []}
    # Warm both paths, then alternate order to reduce order-dependent bias.
    for enabled in (False, True):
        run(base / f"warmup-{enabled}", enabled)
    for i in range(args.repeats):
        trials = {}
        for enabled in (False, True) if i % 2 == 0 else (True, False):
            path = base / f"{i}-{enabled}"
            log, calls, elapsed = run(path, enabled)
            timings[enabled].append(elapsed)
            trials[enabled] = (log, calls)
        baseline, baseline_calls = trials[False]
        traced, traced_calls = trials[True]
        assert baseline_calls == traced_calls == 2
        assert baseline.results.metrics == traced.results.metrics
        assert baseline.samples[0].policy_transcripts == traced.samples[0].policy_transcripts
        assert traced.samples[0].termination_reasons == ("success",)
        receipt = traced.samples[0].trial_metadata[0]["relay"]
        assert receipt["status"] == "captured"
    trace_path = base / f"{args.repeats - 1}-True" / receipt["path"]
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    print(f"Matched {args.repeats} pairs: calls, transcripts, scores; cube reached.")
    print(f"Scores: {traced.results.metrics}")
    print(f"Trace: {trace_path}")
    for row in rows:
        if row["category"] == "tool" and row["scope_category"] == "end":
            print(f"  {row['name']}: {row['metadata']['status']}")
    for enabled, values in timings.items():
        print(
            f"Relay {'on' if enabled else 'off'} median: {statistics.median(values) * 1000:.2f} ms"
        )
    print("Local scripted-provider timings include eval/log I/O and trace finalization.")


if __name__ == "__main__":
    main()
