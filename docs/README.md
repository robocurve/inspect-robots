# Inspect Robots documentation

Guides for evaluating robot policies with Inspect Robots, in suggested reading
order. API details live in the source docstrings; guide pages link each symbol
to the module that defines it.

## Guide

1. [Quickstart](guide/quickstart.md): install and run your first evaluation.
2. [Example commands](guide/examples.md): copy-paste recipes for common run
   shapes.
3. [Rig reference](guide/rig-reference.md): set up and check a physical rig,
   then run policies on it.
4. [Concepts](guide/concepts.md): how an evaluation is factored into small,
   orthogonal pieces.
5. [Writing a benchmark](guide/writing-a-benchmark.md): tasks, scenes and
   scorers.
6. [Policies and embodiments](guide/policies-and-embodiments.md): the two
   runtime protocols and their base classes.
7. [Scoring](guide/scoring.md): built-in and custom scorers,
   epochs, and operator/VLM grading.
8. [Watching a run live](guide/live-view.md): follow an evaluation in the HTML
   viewer.
9. [Logging & Rerun](guide/logging-and-rerun.md): the eval log, sinks and
   Rerun visualization.
10. [Plugins & the registry](guide/plugins.md): register components and
    resolve them by name.
11. [Voice operator input](guide/voice-mode.md): speak operator feedback during
    attended runs.
12. [Authoring an embodiment adapter](guide/adapters.md): declare spaces and
    pass conformance.
13. [Command-line interface](guide/cli.md): every `inspect-robots` command.

## Cookbooks

- [Evaluate a GR00T fine-tune on real YAM arms](cookbooks/gr00t-on-yam.md)
