"""Tests for the summary chains, using fakes instead of the Gemini API."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import summarizer
from config import Settings
from pdf_processor import Chunk, Document
from summarizer import (
    EmptyResponseError,
    PDFSummarizer,
    SummaryResult,
    clean_quotes,
    describe_error,
    message_text,
    pick_models,
    _parse,
)


def make_settings(model: str = "gemini-test") -> Settings:
    return Settings(api_key="test-key", model=model)


@pytest.fixture
def document() -> Document:
    return Document(
        name="doc.pdf",
        pages=["Alpha beta. Gamma delta."],
        text="Alpha beta. Gamma delta.",
        chunks=[
            Chunk(index=1, text="Alpha beta.", tokens=2),
            Chunk(index=2, text="Gamma delta.", tokens=2),
        ],
    )


# -- message parsing -------------------------------------------------------
def test_message_text_handles_plain_string() -> None:
    assert message_text(AIMessage(content="  hello  ")) == "hello"


def test_message_text_drops_thought_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "answer"},
            {"type": "text", "text": " more", "thought": True},
        ]
    )
    assert message_text(message) == "answer"


def test_parse_rejects_empty_response() -> None:
    with pytest.raises(EmptyResponseError, match="no text"):
        _parse(AIMessage(content=""))


# -- quotes ----------------------------------------------------------------
def test_clean_quotes_strips_numbering_and_dedupes() -> None:
    raw = '1. "The first important quote here."\n- The second important quote here.\n\n'
    raw += "The first important quote here.\nHere are the quotes:\nshort\n"
    assert clean_quotes(raw) == [
        "The first important quote here.",
        "The second important quote here.",
    ]


# -- errors ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 RESOURCE_EXHAUSTED", "Rate limit"),
        ("404 NOT_FOUND", "Model not available"),
        ("403 PERMISSION_DENIED", "API key rejected"),
        ("blocked by safety filters", "safety"),
        ("", "RuntimeError"),
    ],
)
def test_describe_error_maps_common_failures(message: str, expected: str) -> None:
    exc = EmptyResponseError(message) if message else RuntimeError()
    assert expected in describe_error(exc)


# -- summarize -------------------------------------------------------------
def test_summarize_chunks_collects_one_row_per_chunk(document: Document) -> None:
    llm = FakeListChatModel(responses=["summary one", "summary two"])
    summarizer_ = PDFSummarizer(settings=make_settings(), llm=llm)

    rows = summarizer_.summarize_chunks(document.chunks, "brief")

    assert [row.chunk_number for row in rows] == [1, 2]
    assert all(row.ok for row in rows)
    assert [row.summary for row in rows] == ["summary one", "summary two"]


def test_failed_chunk_becomes_error_row(document: Document) -> None:
    def flaky(prompt_value):
        text = prompt_value.to_string()
        if "Gamma" in text:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return AIMessage(content="fine")

    summarizer_ = PDFSummarizer(settings=make_settings(), llm=RunnableLambda(flaky))
    rows = summarizer_.summarize_chunks(document.chunks, "brief")

    assert rows[0].ok and rows[0].summary == "fine"
    assert not rows[1].ok
    assert "Rate limit" in rows[1].error
    assert rows[1].summary_chars == 0


def test_empty_model_response_is_an_error_row(document: Document) -> None:
    summarizer_ = PDFSummarizer(
        settings=make_settings(), llm=FakeListChatModel(responses=["", ""])
    )
    rows = summarizer_.summarize_chunks(document.chunks, "brief")
    assert not any(row.ok for row in rows)
    assert "no text" in rows[0].error


def test_summarize_chunks_requires_chunks() -> None:
    summarizer_ = PDFSummarizer(
        settings=make_settings(), llm=FakeListChatModel(responses=["x"])
    )
    with pytest.raises(summarizer.SummarizerError):
        summarizer_.summarize_chunks([], "brief")


# -- combine ---------------------------------------------------------------
def test_combine_returns_single_summary_without_calling_model(document: Document) -> None:
    llm = FakeListChatModel(responses=["one", "two"])
    summarizer_ = PDFSummarizer(settings=make_settings(), llm=llm)
    rows = summarizer_.summarize_chunks(document.chunks, "brief")

    # One successful row needs no reduce call: the summary is passed through.
    assert summarizer_.combine(rows[:1], "brief") == "one"


def test_combine_reduces_multiple_summaries(document: Document) -> None:
    llm = FakeListChatModel(responses=["one", "two", "merged"])
    summarizer_ = PDFSummarizer(settings=make_settings(), llm=llm)
    rows = summarizer_.summarize_chunks(document.chunks, "brief")

    assert summarizer_.combine(rows, "brief") == "merged"


def test_combine_raises_when_every_chunk_failed(document: Document) -> None:
    summarizer_ = PDFSummarizer(
        settings=make_settings(), llm=FakeListChatModel(responses=["", ""])
    )
    rows = summarizer_.summarize_chunks(document.chunks, "brief")
    with pytest.raises(summarizer.SummarizerError, match="Every chunk failed"):
        summarizer_.combine(rows, "brief")


# -- orchestration ---------------------------------------------------------
def test_run_returns_result_and_reports_progress(document: Document) -> None:
    llm = FakeListChatModel(
        responses=["alpha", "gamma", "merged", "analysis", "This quote is long enough to keep."]
    )
    summarizer_ = PDFSummarizer(settings=make_settings(), llm=llm)
    seen: list[float] = []

    result = summarizer_.run(
        document,
        style="brief",
        include_analysis=True,
        include_quotes=True,
        progress=lambda fraction, _message: seen.append(fraction),
    )

    assert isinstance(result, SummaryResult)
    assert result.combined_summary == "merged"
    assert result.analysis == "analysis"
    assert result.quotes == ["This quote is long enough to keep."]
    assert result.successful_chunks == 2
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_missing_api_key_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "get_settings", lambda *a, **k: Settings(api_key="", model="x"))
    with pytest.raises(summarizer.SummarizerError, match="GEMINI_API_KEY"):
        PDFSummarizer()
    assert not Settings(api_key="your_key", model="x").has_api_key


# -- model discovery -------------------------------------------------------
def test_pick_models_prefers_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "available_models", lambda _key: ("gemini-a", "gemini-b"))
    assert pick_models("key", "gemini-b") == ["gemini-b", "gemini-a"]


def test_pick_models_survives_api_failure(monkeypatch) -> None:
    def explode(_key: str):
        raise RuntimeError("offline")

    monkeypatch.setattr(summarizer, "available_models", explode)
    assert pick_models("key", "gemini-x") == ["gemini-x"]
