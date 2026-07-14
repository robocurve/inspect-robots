"""Boot-only proof: launch Isaac Sim via the Inspect Robots adapter on the real install.

Run with the venv that has isaacsim/isaaclab, and Inspect Robots + this package on the
PYTHONPATH. Confirms the adapter's lazy bring-up (`_ensure_app`) reaches a live
`SimulationApp` on the GPU, then closes cleanly. Does NOT need `isaaclab_tasks`.
"""

from __future__ import annotations

import time

from inspect_robots import Embodiment
from inspect_robots_isaacsim import IsaacSimEmbodiment


def main() -> int:
    """Launch, exercise, and close the real Isaac Sim application as a boot proof."""
    emb = IsaacSimEmbodiment(headless=True)
    # Sanity: the adapter is a valid Inspect Robots Embodiment and its info is readable
    # without any Isaac import.
    assert isinstance(emb, Embodiment)
    print(
        f"[adapter] name={emb.info.name} action_dim={emb.info.action_space.dim} "
        f"sim={emb.info.is_simulated} caps={sorted(emb.info.capabilities)}"
    )

    t0 = time.perf_counter()
    print("[boot] launching Isaac Sim SimulationApp (headless)...", flush=True)
    app = emb._ensure_app()
    dt = time.perf_counter() - t0
    running = getattr(app, "is_running", lambda: None)()
    print(f"[boot] SimulationApp live in {dt:.1f}s  is_running={running}  app={type(app).__name__}")

    # Step the kit app a few times to prove the loop is alive.
    for _ in range(3):
        app.update()
    print("[boot] app.update() x3 OK")

    # Print the verdict BEFORE close(): Isaac's close() does a hard os._exit that
    # swallows buffered stdout, so the proof must be emitted first.
    print(f"BOOT PROOF: SUCCESS  (SimulationApp started + stepped on GPU in {dt:.1f}s)", flush=True)
    emb.close()
    print("[boot] closed cleanly")  # may be swallowed by Isaac's hard exit; that's fine
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
