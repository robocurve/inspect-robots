"""Generate a scene whose instruction goes to the policy and rubric goes to the grader.

The returned [`Scene`][inspect_robots.scene.Scene] is a two-consumer contract:
``scene.instruction`` tells the policy what to attempt, while
``scene.metadata["rubric"]`` tells a grader how to judge the final state.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inspect_robots._pngenc import png_data_url
from inspect_robots.embodiment import Embodiment
from inspect_robots.errors import ConfigError
from inspect_robots.rollout import derive_seed
from inspect_robots.scene import Scene

if TYPE_CHECKING:
    from inspect_robots._chatwire import HttpPost

_REPLY_EXCERPT_LIMIT = 500
_TASK_RE = re.compile(r"^\s*TASK:\s*(\S.*)$")
_RUBRIC_RE = re.compile(r"^\s*RUBRIC:(.*)$")

_DEFAULT_INSTRUCTIONS = """You are designing one evaluation task for a general-purpose robot
manipulation system. Observe the tabletop and the items on it in the
attached camera frames. Write one concrete task a single robot arm could
plausibly attempt in this scene, such as picking, placing, stacking,
sorting, or grouping the visible items. Use only items you can actually
see. Then write a grading rubric for that task: the observable conditions
a judge should check on the final camera frames to call the trial a
success, strict enough to be meaningful and fair enough to be achievable."""

_REPLY_CONTRACT = """Reply with your reasoning first if you wish, then end your reply with
exactly these two sections:
TASK: <the task instruction, on one line>
RUBRIC:
<the grading rubric, one or more lines>"""


def _reply_excerpt(reply: str) -> str:
    text = reply.strip()
    return (text or "(empty reply)")[:_REPLY_EXCERPT_LIMIT]


def _reply_error(diagnosis: str, reply: str) -> ConfigError:
    return ConfigError(
        f"task generation reply {diagnosis}: {_reply_excerpt(reply)}\n"
        "fix: use a model that follows the required TASK:/RUBRIC: reply contract"
    )


def _parse_reply(reply: str) -> tuple[str, str]:
    lines = reply.splitlines()
    rubric_index: int | None = None
    rubric_match: re.Match[str] | None = None
    for index, line in enumerate(lines):
        match = _RUBRIC_RE.match(line)
        if match is not None:
            rubric_index = index
            rubric_match = match
    if rubric_index is None or rubric_match is None:
        raise _reply_error("is missing a RUBRIC: marker", reply)

    task: str | None = None
    for line in lines[:rubric_index]:
        match = _TASK_RE.match(line)
        if match is not None:
            task = match.group(1).strip()
    if task is None:
        raise _reply_error("is missing a TASK: line before its final RUBRIC: marker", reply)

    rubric_lines: list[str] = []
    inline = rubric_match.group(1)
    if inline.strip():
        rubric_lines.append(inline)
    rubric_lines.extend(lines[rubric_index + 1 :])
    rubric = "\n".join(rubric_lines).strip()
    if not rubric:
        raise _reply_error("has an empty rubric", reply)
    return task, rubric


def _wire_fix(api_key_env: str) -> str:
    return f"fix: check model=, base_url=, and the ${api_key_env} key (-A k=v from the CLI)"


def _reword_wire_error(error: ConfigError, api_key_env: str) -> ConfigError:
    lines = str(error).splitlines()
    diagnosis = [re.sub(r"^summary\b", "task generation", line) for line in lines]
    diagnosis = [line for line in diagnosis if not line.startswith("fix:")]
    return ConfigError("\n".join([*diagnosis, _wire_fix(api_key_env)]))


def generate_scene(
    embodiment: Embodiment,
    *,
    model: str | None = None,
    instructions: str | None = None,
    instructions_file: str | None = None,
    base_url: str = "https://api.anthropic.com/v1",
    api_key_env: str = "ANTHROPIC_API_KEY",
    max_cameras: int = 4,
    scene_id: str = "auto-0",
    seed: int | None = 0,
    http_post: HttpPost | None = None,
) -> Scene:
    """Peek at one seeded world and return a policy instruction plus grader rubric.

    ``seed`` is the eval seed the caller will pass to
    [`eval`][inspect_robots.eval.eval]. The peek uses its first-trial derived
    seed so a seedable embodiment presents the same initial world to generation
    and evaluation. Configuration, transport, and reply-contract failures raise
    [`ConfigError`][inspect_robots.errors.ConfigError] before rollout starts.
    """
    if model is None or not model.strip():
        raise ConfigError(
            "task generation requires a non-empty model.\n"
            "fix: pass model=... (-A model=... from the CLI)"
        )
    model = model.strip()
    if instructions is not None and instructions_file is not None:
        raise ConfigError(
            "task generation received both instructions and instructions_file.\n"
            "fix: pass only one prompt override"
        )
    if max_cameras < 1:
        raise ConfigError(
            f"task generation max_cameras must be >= 1, got {max_cameras}.\n"
            "fix: pass max_cameras=1 or greater"
        )
    if not scene_id:
        raise ConfigError(
            "task generation scene_id must be non-empty.\n"
            "fix: pass scene_id=... with a stable scene identifier"
        )
    if seed is None:
        raise ConfigError(
            "task generation cannot match eval's randomly drawn seed.\n"
            "fix: pass the same integer seed you will pass to eval()"
        )

    if instructions_file is not None:
        try:
            meta_instructions = Path(instructions_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(
                f"task generation could not read instructions_file {instructions_file!r}: {exc}.\n"
                "fix: pass a readable UTF-8 text file"
            ) from exc
    else:
        meta_instructions = _DEFAULT_INSTRUCTIONS if instructions is None else instructions

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ConfigError(
            f"task generation API key environment variable ${api_key_env} is unset or empty.\n"
            f"fix: set ${api_key_env} or pass api_key_env=... (-A api_key_env=... from the CLI)"
        )

    peek_scene = Scene(
        id=f"{scene_id}-peek",
        instruction="Hold still and observe the scene.",
    )
    observation = embodiment.reset(peek_scene, seed=derive_seed(seed, None, 0))
    if not observation.images:
        raise ConfigError(
            "task generation peek observation has no camera images.\n"
            "fix: configure the embodiment to return at least one image in Observation.images"
        )

    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"{meta_instructions}\n\n{_REPLY_CONTRACT}"}
    ]
    for camera, frame in sorted(observation.images.items())[:max_cameras]:
        try:
            frame_url = png_data_url(frame)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"task generation could not encode the frame from camera {camera!r}: {exc}\n"
                "fix: configure the embodiment to return uint8 HxWx3 images"
            ) from exc
        parts.append({"type": "text", "text": f"camera {camera!r}"})
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_url},
            }
        )
    messages: list[dict[str, Any]] = [{"role": "user", "content": parts}]

    from inspect_robots._chatwire import chat_completion

    try:
        reply = chat_completion(
            base_url,
            api_key,
            model,
            messages,
            http_post=http_post,
        )
    except ConfigError as exc:
        raise _reword_wire_error(exc, api_key_env) from exc
    except OSError as exc:
        raise ConfigError(
            f"task generation request failed: {exc}\n{_wire_fix(api_key_env)}"
        ) from exc

    task_instruction, rubric = _parse_reply(reply)
    return Scene(
        id=scene_id,
        instruction=task_instruction,
        metadata={
            "rubric": rubric,
            "taskgen": {"model": model, "base_url": base_url},
        },
    )
