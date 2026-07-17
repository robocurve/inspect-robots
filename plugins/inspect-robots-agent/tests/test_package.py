from __future__ import annotations


def test_package_imports_and_exports() -> None:
    import inspect_robots_agent

    # Pinned so a version bump that misses either side fails loudly (the
    # 0.1.0 hardcode shipped stale through two releases unnoticed).
    assert inspect_robots_agent.__version__ == "0.12.0"
    assert callable(inspect_robots_agent.agent_policy)
    assert inspect_robots_agent.__all__ == [
        "ENV_MODEL",
        "AgentPolicyConfig",
        "AssistantMessage",
        "ChatClient",
        "LLMAgentPolicy",
        "Provider",
        "ResponsesClient",
        "agent_policy",
        "encode_png",
        "png_data_url",
        "resolve_provider",
    ]
