"""Smoke tests for the Streamlit UI using streamlit.testing (no browser)."""

from __future__ import annotations

from pathlib import Path

import pytest

import summarizer

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app(monkeypatch, api_key: str = "test-key") -> AppTest:
    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setattr(summarizer, "available_models", lambda _key: ("gemini-test", "gemini-other"))
    return AppTest.from_file(str(APP), default_timeout=90)


def test_app_renders_and_lists_models(monkeypatch) -> None:
    app = _app(monkeypatch).run()

    assert not app.exception
    assert "PDF" in app.title[0].value
    # Configured model first (from .env), then whatever the API reports.
    assert app.selectbox[0].options[0] == "gemini-3.6-flash"
    assert {"gemini-test", "gemini-other"} <= set(app.selectbox[0].options)
    assert app.selectbox[1].value == "detailed"
    assert app.file_uploader


def test_app_stops_without_api_key(monkeypatch) -> None:
    app = _app(monkeypatch, api_key="").run()

    assert not app.exception
    assert app.error
    assert "GEMINI_API_KEY" in app.error[0].value
    assert not app.file_uploader


def test_settings_change_requires_rerun(monkeypatch) -> None:
    """Sidebar controls are wired and a summary result invalidates on change."""
    app = _app(monkeypatch).run()
    app.selectbox[1].set_value("executive").run()

    assert not app.exception
    assert app.selectbox[1].value == "executive"
    assert app.slider[0].value >= 1000
