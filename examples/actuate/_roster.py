"""Shared, booth-editable model roster for the Actuate conference demo."""

from __future__ import annotations

# Booth-editable. Confirm model IDs against each provider before the event.
ROSTER: dict[str, dict[str, str]] = {
    "GPT-5.6 Sol": {
        "model": "gpt-5.6-sol",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "Opus 5": {
        "model": "claude-opus-5",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "Gemini 3.7 Flash": {
        "model": "gemini-3.7-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
}

ACCENTS = {
    "GPT-5.6 Sol": "#10a37f",
    "Opus 5": "#d97757",
    "Gemini 3.7 Flash": "#4796e3",
}

EFFORT = "high"
