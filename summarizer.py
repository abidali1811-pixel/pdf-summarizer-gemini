"""Gemini chains: per-chunk summaries, hierarchical combine, analysis, quotes.

Design notes
------------
* Gemini 3.x returns "thinking" tokens before any visible text. A small
  `max_output_tokens` therefore produces an *empty* response with
  `finishReason=MAX_TOKENS` (this bit us during Phase 0), so every call sets both
  an output budget and an explicit thinking budget, and an empty answer is
  treated as an error instead of a summary.
* `AIMessage.content` can be a string or a list of content blocks, with thought
  blocks mixed in. `message_text()` normalises both shapes.
* Chunk summaries are produced with `batch(..., return_exceptions=True)` so one
  bad chunk cannot destroy a long document: failures become rows in the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from config import DEFAULT_MODEL, MAX_CONCURRENCY, Settings, get_settings
from pdf_processor import Chunk, Document
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

# Long documents are reduced in rounds so the combine prompt never overflows.
MAX_REDUCE_GROUP = 8
# Analysis and quotes look at the start of the document, capped for safety.
MAX_CONTEXT_CHARS = 32_000


class SummarizerError(RuntimeError):
    """Base class for summarizer failures."""


class EmptyResponseError(SummarizerError):
    """The model returned no text (usually output budget eaten by thinking)."""


@dataclass
class ChunkSummary:
    chunk_number: int
    summary: str
    ok: bool
    original_chars: int
    summary_chars: int
    error: str = ""

    @property
    def compression(self) -> float:
        if not self.original_chars:
            return 0.0
        return round(self.summary_chars / self.original_chars, 2)


@dataclass
class SummaryResult:
    style: str
    model: str
    combined_summary: str
    chunks: list[ChunkSummary] = field(default_factory=list)
    analysis: str | None = None
    quotes: list[str] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def successful_chunks(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.ok)

    @property
    def failed_chunks(self) -> list[ChunkSummary]:
        return [chunk for chunk in self.chunks if not chunk.ok]


def message_text(message: Any) -> str:
    """Extract visible text from a chat message, dropping thought blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                kind = block.get("type", "text")
                if kind in {"text", "text_delta"} and not block.get("thought"):
                    parts.append(str(block.get("text", "")))
                continue
            # Provider block objects (e.g. TextAccessor).
            if getattr(block, "thought", False):
                continue
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _parse(message: Any) -> str:
    text = message_text(message)
    if not text:
        raise EmptyResponseError(
            "The model returned no text. Raise the output token budget or set "
            "thinking_budget=0 (Gemini 3 spends the budget on thinking first)."
        )
    return text


def clean_quotes(raw: str) -> list[str]:
    """Normalise the quotes response into plain strings."""
    quotes: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        line = line.strip('"“”').strip()
        if len(line) < 15:
            continue
        if line.lower().startswith(("here are", "the following", "quotes:")):
            continue
        if line not in quotes:
            quotes.append(line)
    return quotes


def describe_error(exc: BaseException) -> str:
    """Short, actionable message for a failed chunk."""
    text = str(exc).strip().replace("\n", " ")
    lowered = text.lower()
    if "429" in text or "resource_exhausted" in lowered or "quota" in lowered:
        return "Rate limit or quota reached — wait a minute, reduce chunk size, or switch model"
    if "404" in text or "not_found" in lowered:
        return "Model not available to this API key — pick another model"
    if "403" in text or "permission_denied" in lowered:
        return "API key rejected or restricted — check the key and enabled APIs"
    if "safety" in lowered or "blocked" in lowered:
        return "Response blocked by safety filters"
    if isinstance(exc, EmptyResponseError):
        return str(exc)
    return text[:300] if text else type(exc).__name__


@lru_cache(maxsize=8)
def available_models(api_key: str) -> tuple[str, ...]:
    """Model ids that support generateContent, newest first (per the API)."""
    from google import genai  # imported lazily so tests never need network

    client = genai.Client(api_key=api_key)
    names: list[str] = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        name = (model.name or "").removeprefix("models/")
        if name:
            names.append(name)
    return tuple(names)


def pick_models(api_key: str, preferred: str) -> list[str]:
    """Preferred model first, then the rest of the account's usable models."""
    try:
        remote = list(available_models(api_key))
    except Exception:  # noqa: BLE001 - offline or invalid key: fall back to config
        remote = []
    ordered = [preferred] if preferred else []
    ordered += [name for name in remote if name != preferred]
    if not ordered:
        ordered = [DEFAULT_MODEL]
    return ordered


class PDFSummarizer:
    """Map-reduce summarizer around a Gemini chat model."""

    def __init__(
        self,
        model: str | None = None,
        *,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings or get_settings(model)
        if llm is None and not self.settings.has_api_key:
            raise SummarizerError("GEMINI_API_KEY is not set — copy .env.example to .env")

        llm_kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_output_tokens": self.settings.output_tokens,
            "thinking_budget": self.settings.thinking_budget,
            "max_retries": 3,
            "timeout": 120,
            "google_api_key": self.settings.api_key,
        }
        # Only send sampling parameters when configured: several Gemini 3.x models
        # use fixed sampling defaults and warn that the value is ignored.
        if self.settings.temperature is not None:
            llm_kwargs["temperature"] = self.settings.temperature

        self.llm: BaseChatModel = llm or ChatGoogleGenerativeAI(**llm_kwargs)
        parser: Runnable = RunnableLambda(_parse)
        self.chunk_chain = CHUNK_SUMMARY | self.llm | parser
        self.combine_chain = COMBINE | self.llm | parser
        self.analysis_chain = ANALYSIS | self.llm | parser
        self.quotes_chain = QUOTES | self.llm | parser

    # -- single chunk ------------------------------------------------------
    def summarize_text(self, text: str, style: str = "detailed") -> str:
        return self.chunk_chain.invoke({"style": STYLES.get(style, STYLES["detailed"]), "text": text})

    # -- map ---------------------------------------------------------------
    def summarize_chunks(
        self,
        chunks: Iterable[Chunk | str],
        style: str = "detailed",
        *,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> list[ChunkSummary]:
        texts = [chunk.text if isinstance(chunk, Chunk) else str(chunk) for chunk in chunks]
        if not texts:
            raise SummarizerError("Nothing to summarize: the document has no chunks")

        style_text = STYLES.get(style, STYLES["detailed"])
        payload = [{"style": style_text, "text": text} for text in texts]
        results = self.chunk_chain.batch(
            payload,
            config={"max_concurrency": max_concurrency},
            return_exceptions=True,
        )

        rows: list[ChunkSummary] = []
        for number, (text, result) in enumerate(zip(texts, results), start=1):
            if isinstance(result, BaseException):
                rows.append(
                    ChunkSummary(
                        chunk_number=number,
                        summary="",
                        ok=False,
                        original_chars=len(text),
                        summary_chars=0,
                        error=describe_error(result),
                    )
                )
            else:
                rows.append(
                    ChunkSummary(
                        chunk_number=number,
                        summary=str(result),
                        ok=True,
                        original_chars=len(text),
                        summary_chars=len(str(result)),
                    )
                )
        return rows

    # -- reduce ------------------------------------------------------------
    def combine(self, rows: list[ChunkSummary], style: str = "detailed") -> str:
        """Merge chunk summaries, reducing in rounds for very long documents."""
        style_text = STYLES.get(style, STYLES["detailed"])
        current = [row.summary for row in rows if row.ok and row.summary.strip()]
        if not current:
            failed = [row for row in rows if not row.ok]
            details = "; ".join(
                f"chunk {row.chunk_number}: {row.error or 'unknown error'}"
                for row in failed[:3]
            )
            remaining = len(failed) - 3
            if remaining > 0:
                details += f"; and {remaining} more"
            message = "Every chunk failed — nothing to combine"
            if details:
                message += f" ({details})"
            raise SummarizerError(message)
        if len(current) == 1:
            return current[0]

        rounds = 0
        while len(current) > 1:
            rounds += 1
            if rounds > 6:
                break
            next_round: list[str] = []
            for start in range(0, len(current), MAX_REDUCE_GROUP):
                group = current[start : start + MAX_REDUCE_GROUP]
                if len(group) == 1:
                    next_round.append(group[0])
                    continue
                numbered = "\n\n".join(
                    f"[Section {start + offset + 1}]\n{text}"
                    for offset, text in enumerate(group)
                )
                next_round.append(
                    self.combine_chain.invoke({"style": style_text, "summaries": numbered})
                )
            current = next_round

        return current[0] if len(current) == 1 else "\n\n".join(current)

    # -- optional extras ---------------------------------------------------
    def analyze(self, text: str) -> str:
        excerpt = text[:MAX_CONTEXT_CHARS]
        return self.analysis_chain.invoke({"text": excerpt})

    def extract_quotes(self, text: str) -> list[str]:
        excerpt = text[:MAX_CONTEXT_CHARS]
        return clean_quotes(self.quotes_chain.invoke({"text": excerpt}))

    # -- orchestration -----------------------------------------------------
    def run(
        self,
        document: Document,
        style: str = "detailed",
        *,
        include_analysis: bool = False,
        include_quotes: bool = False,
        progress: Any = None,
    ) -> SummaryResult:
        """Summarize a processed document end to end.

        `progress` may be a callable receiving (fraction: float, message: str).
        """
        def report(fraction: float, message: str) -> None:
            if callable(progress):
                progress(fraction, message)

        report(0.05, f"Summarizing {len(document.chunks)} chunks with {self.settings.model}...")
        rows = self.summarize_chunks(document.chunks, style)
        failed = [row for row in rows if not row.ok]
        report(
            0.65,
            f"Combining {len(rows) - len(failed)} summaries"
            + (f" ({len(failed)} failed)" if failed else "")
            + "...",
        )
        combined = self.combine(rows, style)

        analysis = None
        if include_analysis:
            report(0.8, "Analysing document structure...")
            analysis = self.analyze(document.text)

        quotes: list[str] = []
        if include_quotes:
            report(0.9, "Extracting key quotes...")
            quotes = self.extract_quotes(document.text)

        report(1.0, "Done")
        return SummaryResult(
            style=style,
            model=self.settings.model,
            combined_summary=combined,
            chunks=rows,
            analysis=analysis,
            quotes=quotes,
        )
