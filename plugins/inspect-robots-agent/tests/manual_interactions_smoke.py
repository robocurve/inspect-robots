"""Probe the real Gemini Interactions endpoint before announcing wire support."""

from __future__ import annotations

import os

from inspect_robots_agent._interactions import InteractionsClient
from inspect_robots_agent._llm import Provider
from inspect_robots_agent.policy import _INTERACTIONS_BASE


def main() -> int:
    """Exercise basic, chained, and function-result calls when a key is available."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is unset; skipping Interactions smoke probe")
        return 0

    model = os.environ.get("INSPECT_ROBOTS_MODEL", "google/gemini-3.7-flash")
    client = InteractionsClient(Provider(base_url=_INTERACTIONS_BASE, api_key=key, model=model))
    messages = [
        {"role": "system", "content": "Answer briefly and follow function requests."},
        {"role": "user", "content": "Reply with the word ready."},
    ]
    tool = {
        "type": "function",
        "function": {
            "name": "report_smoke",
            "description": "Report that the smoke probe reached its function step.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    try:
        first = client.complete(messages, [])
        messages.extend([first.raw(), {"role": "user", "content": "Now reply chained."}])
        second = client.complete(messages, [])
        messages.extend(
            [
                second.raw(),
                {"role": "user", "content": "Call report_smoke with no arguments."},
            ]
        )
        function_turn = client.complete(messages, [tool])
        if not function_turn.tool_calls:
            raise RuntimeError("Interactions smoke probe did not receive a function call")
        call = function_turn.tool_calls[0]
        messages.extend(
            [
                function_turn.raw(),
                {"role": "tool", "tool_call_id": call.id, "content": "smoke passed"},
            ]
        )
        final = client.complete(messages, [tool])
        print(f"Interactions smoke probe passed: {final.content or 'function round-trip complete'}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
