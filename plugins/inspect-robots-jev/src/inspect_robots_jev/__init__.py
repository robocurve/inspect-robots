"""inspect-robots-jev: TypeSafe's Jev decision model as an Inspect Robots policy.

Jev reads text and picks one option from a list; it never sees images. This
plugin reads AprilTag poses from the fixed top camera, writes a curated text
state in directional words, asks Jev one Choice question per step, and turns
the pick into a bounded Cartesian move. Registered as the policy ``jev`` and
the scorer ``jev_cube_in_bowl``.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("inspect-robots-jev")
