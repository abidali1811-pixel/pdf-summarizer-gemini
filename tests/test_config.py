"""Tests for config resolution."""

from __future__ import annotations

import config


def test_clamp_chunk_tokens_bounds() -> None:
    assert config.clamp_chunk_tokens(10) == config.MIN_CHUNK_TOKENS
    assert config.clamp_chunk_tokens(10**9) == config.MAX_CHUNK_TOKENS
    assert config.clamp_chunk_tokens(4000) == 4000


def test_get_settings_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "env-model")

    settings = config.get_settings()

    assert settings.api_key == "env-key"
    assert settings.model == "env-model"
    assert settings.has_api_key


def test_get_settings_uses_streamlit_secrets_when_environment_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    settings = config.get_settings(
        secrets={"GEMINI_API_KEY": "cloud-key", "GEMINI_MODEL": "cloud-model"}
    )

    assert settings.api_key == "cloud-key"
    assert settings.model == "cloud-model"


def test_get_settings_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    settings = config.get_settings(model="chosen-model", chunk_tokens=2000)
    assert settings.model == "chosen-model"
    assert settings.chunk_tokens == 2000


def test_model_falls_back_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert config.get_settings().model == config.DEFAULT_MODEL


def test_placeholder_key_is_not_usable(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    assert not config.get_settings().has_api_key


def test_thinking_budget_default_is_disabled() -> None:
    # Discovered in Phase 0: thinking eats the output budget on Gemini 3.x.
    assert config.THINKING_BUDGET == 0
    assert config.OUTPUT_TOKEN_BUDGET >= 512
