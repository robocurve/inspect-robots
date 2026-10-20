"""Export one scripted CubePick trial through Relay's ATIF consumer.

Install with ``uv sync --all-packages --extra dev --extra relay --python 3.11``.
Run with ``uv run --no-sync python examples/relay_agent.py``.
No model API key or robot hardware is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from inspect_robots_agent import LLMAgentPolicy

from inspect_robots import EvalLog, eval
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task


def run(path: Path) -> EvalLog:
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
        relay=True,
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
    (log,) = eval(task, policy, CubePickEmbodiment(), seed=0, log_dir=str(path))
    return log


async def export_atif(path: Path) -> None:
    """Consume a real Inspect event stream with Relay's local ATIF exporter."""
    from nemo_relay import observability, plugin

    config = plugin.PluginConfig(
        components=[
            observability.ComponentSpec(
                observability.ObservabilityConfig(
                    atif=observability.AtifConfig(
                        enabled=True,
                        agent_name="inspect-robots-agent",
                        agent_version=version("inspect-robots-agent"),
                        model_name="scripted-demo",
                        output_directory=str(path / "atif"),
                    ),
                ),
            ),
        ],
    )
    async with plugin.activate(config):
        # Keep the plugin's event loop available while the synchronous eval
        # waits for its local capture subscriber to drain.
        log = await asyncio.to_thread(run, path / "inspect")
        assert log.samples[0].trial_metadata[0]["relay"]["status"] == "captured"
    # Closing the plugin flushes the consumer before we inspect its output.
    (trajectory_path,) = (path / "atif").glob("*.json")
    trajectory = json.loads(trajectory_path.read_text())
    outcomes = [
        (result["extra"]["event_name"], result["extra"]["metadata"]["status"])
        for step in trajectory["steps"]
        for result in step.get("observation", {}).get("results", [])
    ]
    assert outcomes == [("unknown", "rejected"), ("move_by", "accepted")]
    assert log.results.metrics == {"success_at_end": 1.0}
    print(f"Relay ATIF consumer: {trajectory_path}")
    print(f"  dispatch outcomes: {outcomes}")
    print(f"  token metrics: {trajectory['final_metrics']}")
    print(f"  independent Inspect score: {log.results.metrics}")


def main() -> None:
    """Write native Inspect evidence and an ATIF export in a new run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/relay-demo"))
    args = parser.parse_args()
    asyncio.run(export_atif(args.log_dir / uuid4().hex))


if __name__ == "__main__":
    main()
