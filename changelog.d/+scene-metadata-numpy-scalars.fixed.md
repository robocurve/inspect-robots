**Core:** NumPy integer, Boolean and floating scalars in `Scene.metadata` are
kept in the saved log as plain numbers instead of being silently dropped;
temporal and complex scalars are still omitted.
