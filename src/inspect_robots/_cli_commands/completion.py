"""Shell completion script generator for inspect-robots CLI."""

from __future__ import annotations


def generate_completion_script(shell: str, subcommands: tuple[str, ...]) -> str:
    """Generate ready-to-source completion script for bash or zsh."""
    if shell == "bash":
        subcmds = " ".join(subcommands)
        return f"""# inspect-robots bash completion script
_inspect_robots_completions() {{
    local cur prev subcommands
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    subcommands="{subcmds}"

    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "$subcommands" -- "$cur") )
        return 0
    fi
}}
complete -F _inspect_robots_completions inspect-robots
"""
    elif shell == "zsh":
        sub_list = "\n        ".join(
            f"'{cmd}:inspect-robots {cmd} subcommand'" for cmd in subcommands
        )
        return f"""#compdef inspect-robots

_inspect_robots() {{
    local -a subcommands
    subcommands=(
        {sub_list}
    )
    _describe -t commands 'inspect-robots subcommand' subcommands
}}

_inspect_robots "$@"
"""
    raise ValueError(f"unsupported shell: {shell}")
