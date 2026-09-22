from importlib.metadata import entry_points

import inspect_robots_jev


def test_version_is_string() -> None:
    assert isinstance(inspect_robots_jev.__version__, str)


def test_entry_points_declared() -> None:
    policies = {ep.name: ep.value for ep in entry_points(group="inspect_robots.policies")}
    scorers = {ep.name: ep.value for ep in entry_points(group="inspect_robots.scorers")}
    assert policies["jev"] == "inspect_robots_jev.policy:jev_policy"
    assert scorers["jev_cube_in_bowl"] == "inspect_robots_jev.scorer:cube_in_bowl"
