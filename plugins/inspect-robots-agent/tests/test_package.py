from __future__ import annotations


def test_package_imports_and_exports() -> None:
    import inspect_robots_agent

    assert inspect_robots_agent.__version__
    assert callable(inspect_robots_agent.agent_policy)
