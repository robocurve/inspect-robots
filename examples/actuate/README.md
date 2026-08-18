# Actuate conference demo

This conference demo is not shipped as part of Inspect Robots. It runs attended evaluations and
serves a local combined monitor screen.

**Credentials:** Export `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`.

**Terminal 1:** Run `python examples/actuate/run.py [-- extra inspect-robots run args]`.

**Terminal 2:** Run `python examples/actuate/serve.py`.

**Display:** Open http://localhost:8377/. Tile the monitor page beside the Rerun viewer window
spawned by the run. The `/leaderboard` route is an alias for the same combined screen.

**Booth setup:** The roster constants in `_roster.py` are meant to be edited on site. Confirm all
model IDs with their providers before the event.
