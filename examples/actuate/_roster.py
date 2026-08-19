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
ROSTER: dict[str, dict[str, object]] = {
    "GPT-5.6 Sol": {
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
}

ACCENTS = {
    "GPT-5.6 Sol": "#10a37f",
    "Opus 5": "#d97757",
    "Gemini 3.7 Flash": "#4796e3",
    "Kimi K3": "#9067e8",
}

EFFORT = "high"
