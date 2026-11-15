"""Central configuration for the PDF summarizer.

Everything tunable lives here so the UI, the processing pipeline and the tests
share one source of truth. Values come from the environment where it makes sense
(`GEMINI_API_KEY`, `GEMINI_MODEL`) and fall back to safe defaults otherwise.

Import-safe: it never fails if `python-dotenv` is missing, so `scripts/check_setup.py`
and unit tests keep working in a bare environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

# --- Gemini ---------------------------------------------------------------
# Used when GEMINI_MODEL is absent. The UI validates against the live model list
# rather than trusting this constant.
DEFAULT_MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash")

# Gemini 3.x spends output tokens on internal reasoning before writing anything.
# With a small budget the response comes back empty with finishReason=MAX_TOKENS,
# so every call sets an explicit budget:
#   THINKING_BUDGET = 0     -> fastest and cheapest; summarising is not a
#                              multi-step reasoning task
#   THINKING_BUDGET = None  -> provider default (model decides), needs a larger
#                              OUTPUT_TOKEN_BUDGET
THINKING_BUDGET: int | None = 0
OUTPUT_TOKEN_BUDGET = 8192

TEMPERATURE: float | None = None
# Some Gemini 3.x models use fixed sampling defaults and reject/ignore
# `temperature` (langchain warns loudly). Left unset we omit it entirely; set a
# float here only for models that accept it.
REQUEST_TIMEOUT_SECONDS = 120
MAX_CONCURRENCY = 4
MAX_RETRIES = 3

# --- Chunking -------------------------------------------------------------
DEFAULT_CHUNK_TOKENS = 8000
MIN_CHUNK_TOKENS = 1000
MAX_CHUNK_TOKENS = 16000

# --- Uploads --------------------------------------------------------------
MAX_UPLOAD_MB = 25
ALLOWED_EXTENSIONS = ("pdf",)

# --- Display --------------------------------------------------------------
APP_TITLE = "PDF Reader + AI Summarizer"
APP_ICON = "📄"


def _read_dotenv(path: Path) -> None:
    """Load .env into os.environ without overriding real environment variables."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 - optional dependency

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # Fallback for a bare environment (no python-dotenv yet).
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_read_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    api_key: str
    model: str
    temperature: float | None = TEMPERATURE
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    output_tokens: int = OUTPUT_TOKEN_BUDGET
    thinking_budget: int | None = THINKING_BUDGET
    max_concurrency: int = MAX_CONCURRENCY

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("your")


def get_settings(
    model: str | None = None,
    chunk_tokens: int | None = None,
    secrets: Mapping[str, object] | None = None,
) -> Settings:
    """Build a Settings object from the environment plus optional overrides."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key and secrets is not None:
        api_key = str(secrets.get("GEMINI_API_KEY", "")).strip()

    env_model = os.getenv("GEMINI_MODEL", "").strip()
    secret_model = ""
    if not env_model and secrets is not None:
        secret_model = str(secrets.get("GEMINI_MODEL", "")).strip()

    return Settings(
        api_key=api_key,
        model=model or env_model or secret_model or DEFAULT_MODEL,
        chunk_tokens=chunk_tokens or DEFAULT_CHUNK_TOKENS,
    )


def clamp_chunk_tokens(value: int) -> int:
    return max(MIN_CHUNK_TOKENS, min(MAX_CHUNK_TOKENS, int(value)))
