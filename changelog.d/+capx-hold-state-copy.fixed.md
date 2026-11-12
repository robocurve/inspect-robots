**CaP-X plugin:** the terminal hold sent on `FINISH`, `GIVE_UP` or an
exhausted LLM call budget is built from a copy of the state observed at the
start of the turn. Model code that edited `obs["state"][...]` in place could
previously turn that hold into an arbitrary one-step target that skipped the
plugin's speed limit and bounds clip.
