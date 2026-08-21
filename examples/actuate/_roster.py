"""Shared, booth-editable model roster for the Actuate conference demo."""

from __future__ import annotations

ALL_ROLES = ("tasker", "test_taker", "grader")

# Booth-editable. Confirm model IDs against each provider before the event.
# model/base_url/api_key_env drive the task maker (-A) and grader (-G) over
# each provider's OpenAI-compatible chat endpoint. "policy" is the complete
# -P argument set used when the model is drawn as test-taker (effort is
# appended separately): GPT must speak the Responses API (OpenAI rejects
# function tools with reasoning_effort on /chat/completions) and Opus runs
# the native messages wire at speed=fast, matching the rig's usual command.
# Roles are pinned: GPT-5.6 Sol always authors tasks and Gemini 3.7 Flash
# always grades (each role's pool has one member), while every entry
# competes as test-taker.
ROSTER: dict[str, dict[str, object]] = {
    "GPT-5.6 Sol": {
        "roles": ("tasker", "test_taker"),
        "model": "gpt-5.6-sol",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "policy": {
            "model": "gpt-5.6-sol",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "wire": "responses",
        },
    },
    "Opus 5": {
        "roles": ("test_taker",),
        "model": "claude-opus-5",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "policy": {
            "model": "anthropic/claude-opus-5",
            "wire": "messages",
            "speed": "fast",
        },
    },
    "Gemini 3.7 Flash": {
        "roles": ("test_taker", "grader"),
        "model": "gemini-3.7-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "policy": {
            "model": "google/gemini-3.7-flash",
            "wire": "interactions",
        },
    },
    "Kimi K3": {
        # test-taker only: drives the robot but never authors tasks or grades.
        "roles": ("test_taker",),
        # OpenRouter because the rig already holds OPENROUTER_API_KEY; for
        # Moonshot direct use model=kimi-k3, base_url=https://api.moonshot.ai/v1,
        # api_key_env=MOONSHOT_API_KEY.
        "model": "moonshotai/kimi-k3",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "policy": {
            "model": "moonshotai/kimi-k3",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    },
    # There is no plain gemini-3-pro text model on the Gemini API; the
    # pro-line text model is the 3.1 preview.
    "Gemini 3.1 Pro": {
        "roles": ("test_taker",),
        "model": "gemini-3.1-pro-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "policy": {
            "model": "google/gemini-3.1-pro-preview",
            "wire": "interactions",
        },
    },
    "GPT-5": {
        "roles": ("test_taker",),
        "model": "gpt-5",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "policy": {
            "model": "gpt-5",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "wire": "responses",
        },
    },
    # The base thinking model, not gpt-5.4-pro.
    "GPT-5.4": {
        "roles": ("test_taker",),
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "policy": {
            "model": "gpt-5.4",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "wire": "responses",
        },
    },
}

ACCENTS = {
    "GPT-5.6 Sol": "#10a37f",
    "Opus 5": "#d97757",
    "Gemini 3.7 Flash": "#4796e3",
    "Kimi K3": "#9067e8",
    "Gemini 3.1 Pro": "#3fb6c9",
    "GPT-5": "#d4608d",
    "GPT-5.4": "#c9a54a",
}

EFFORT = "high"
