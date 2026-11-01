"""Private package holding the ``inspect-robots`` subcommand handlers.

``inspect_robots.cli`` builds the parser, dispatches, and re-exports every name
defined here. Handlers import their cross-command helpers from sibling modules,
and reach the collaborators that tests patch (``claim_devices``,
``_resolve_components``, ``render_html``, the ``view --serve`` cluster,
``_print_degraded``) through the ``inspect_robots.cli`` namespace at call time,
so monkeypatching ``inspect_robots.cli.<name>`` keeps intercepting them.
"""
