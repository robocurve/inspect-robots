"""Generate the single-page API reference from source docstrings with Griffe."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from griffe import Attribute, Class, Function, Module, Object, load

_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _ROOT / "docs" / "api" / "index.md"
_SEARCH_PATHS = [str(_ROOT / "src")]
_AUTOREF = re.compile(r"\[`([^`]+)`\]\[([^\]]+)\]")
_SECTIONS = (
    ("Core types & spaces", ("inspect_robots.types", "inspect_robots.spaces")),
    (
        "Policy & embodiment",
        ("inspect_robots.policy", "inspect_robots.embodiment"),
    ),
    ("Tasks & scenes", ("inspect_robots.scene", "inspect_robots.task")),
    ("Scoring", ("inspect_robots.scorer",)),
    (
        "Rollout, controllers & safety",
        (
            "inspect_robots.rollout",
            "inspect_robots.controller",
            "inspect_robots.approver",
            "inspect_robots.frames",
            "inspect_robots.transcript",
        ),
    ),
    (
        "Compatibility & errors",
        ("inspect_robots.compat", "inspect_robots.errors"),
    ),
    ("Evaluation & logs", ("inspect_robots.eval", "inspect_robots.log")),
    (
        "Logging sinks",
        (
            "inspect_robots.logging.sink",
            "inspect_robots.logging.json_log",
            "inspect_robots.logging.rerun_sink",
        ),
    ),
    ("Registry & CLI", ("inspect_robots.registry", "inspect_robots.cli")),
    (
        "Mock world",
        ("inspect_robots.mock.cubepick", "inspect_robots.mock.policies"),
    ),
)
_PREAMBLE = """Generated automatically from the source docstrings. The public,
stability-guaranteed surface is everything exported by `inspect_robots.__all__`
(`eval`, `eval_set`, `read_eval_log`, `EvalLog` and the other log dataclasses);
the sections below document the full framework."""


def _load_modules() -> dict[str, Module]:
    modules: dict[str, Module] = {}
    for _section, module_names in _SECTIONS:
        for module_name in module_names:
            loaded = load(module_name, search_paths=_SEARCH_PATHS)
            if not isinstance(loaded, Module):
                raise TypeError(f"{module_name} did not resolve to a module")
            modules[module_name] = loaded
    return modules


def _public_members(module: Module) -> list[Object]:
    members = [
        member
        for name, member in module.members.items()
        if not name.startswith("_") and isinstance(member, Object) and member.parent is module
    ]
    if not members:
        raise RuntimeError(f"module {module.path} yielded zero public members")
    return members


def _signature(member: Object) -> str:
    if isinstance(member, Function):
        return f"def {member.signature()}:"
    if isinstance(member, Class):
        initializer = member.members.get("__init__")
        if isinstance(initializer, Function):
            call = str(initializer.signature()).replace("__init__", member.name, 1)
            call = call.split(" -> ", maxsplit=1)[0]
            return f"class {call}:"
        bases = ", ".join(str(base) for base in member.bases)
        return f"class {member.name}({bases}):" if bases else f"class {member.name}:"
    if isinstance(member, Attribute):
        annotation = str(member.annotation) if member.annotation is not None else ""
        value = str(member.value) if member.value is not None else ""
        declaration = member.name
        if annotation:
            declaration += f": {annotation}"
        if value:
            declaration += f" = {value}"
        return declaration
    raise TypeError(f"unsupported public member type for {member.path}: {member.kind}")


def _rewrite_autorefs(text: str, documented_targets: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target in documented_targets:
            return f"[`{label}`](#{target})"
        return f"`{label}`"

    return _AUTOREF.sub(replace, text)


def _docstring(object_: Object, documented_targets: set[str]) -> str:
    if object_.docstring is None:
        return ""
    return _rewrite_autorefs(object_.docstring.value.strip(), documented_targets)


def _render(modules: dict[str, Module]) -> str:
    members_by_module = {
        module_name: _public_members(module) for module_name, module in modules.items()
    }
    documented_targets = {
        str(member.path) for members in members_by_module.values() for member in members
    }
    lines = ["# API reference", "", _PREAMBLE, ""]
    for section_name, module_names in _SECTIONS:
        lines.extend((f"## {section_name}", ""))
        for module_name in module_names:
            module = modules[module_name]
            lines.extend((f"### {module_name}", ""))
            module_docstring = _docstring(module, documented_targets)
            if module_docstring:
                lines.extend((module_docstring, ""))
            for member in members_by_module[module_name]:
                lines.extend(
                    (
                        f"#### {member.name} {{#{member.path}}}",
                        "",
                        "```python",
                        _signature(member),
                        "```",
                        "",
                    )
                )
                member_docstring = _docstring(member, documented_targets)
                if member_docstring:
                    lines.extend((member_docstring, ""))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Write a complete API page atomically after every listed module validates."""
    try:
        output = _render(_load_modules())
    except Exception as error:
        print(f"API documentation generation failed: {error}", file=sys.stderr)
        return 1
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(output, encoding="utf-8")
    print(f"Generated {_OUTPUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
