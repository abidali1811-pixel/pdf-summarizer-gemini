"""Tests for extraction, cleaning and chunking."""

from __future__ import annotations

import io

import pytest

import pdf_processor
from pdf_processor import PDFError, PDFProcessor


@pytest.fixture(scope="module")
def processor() -> PDFProcessor:
    return PDFProcessor()


class _FakePage:
    def __init__(self, text: str | Exception) -> None:
        self._text = text

    def extract_text(self) -> str:
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class _FakeReader:
    def __init__(self, pages, encrypted: bool = False) -> None:
        self.pages = pages
        self.is_encrypted = encrypted

    def decrypt(self, _password: str) -> None:
        return None


def test_clean_text_collapses_whitespace_but_keeps_paragraphs(processor: PDFProcessor) -> None:
    raw = "Title\r\n\r\n\r\n\nBody   with\tspaces\x00 and control\x1f chars."
    assert processor.clean_text(raw) == "Title\n\nBody with spaces and control chars."


def test_clean_text_joins_hyphenated_line_breaks(processor: PDFProcessor) -> None:
    assert processor.clean_text("sum-\nmarise this") == "summarise this"


def test_count_tokens_matches_tiktoken(processor: PDFProcessor) -> None:
    assert processor.count_tokens("hello world") == 2
    assert processor.count_tokens("") == 0


def test_chunks_respect_budget_and_are_numbered(processor: PDFProcessor) -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    chunks = processor.chunk_text(text, max_tokens=120)

    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.tokens <= 120 for chunk in chunks)
    joined = " ".join(chunk.text for chunk in chunks)
    assert joined.count("Sentence number") == 400


def test_single_sentence_longer_than_budget_is_hard_split(processor: PDFProcessor) -> None:
    text = "word " * 2000  # one giant "sentence", no terminal punctuation
    chunks = processor.chunk_text(text, max_tokens=100)

    assert len(chunks) >= 20
    assert all(chunk.tokens <= 100 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_empty_text_raises(processor: PDFProcessor) -> None:
    with pytest.raises(PDFError, match="scanned images"):
        processor.chunk_text("   \n  ", max_tokens=100)


def test_invalid_budget_raises(processor: PDFProcessor) -> None:
    with pytest.raises(ValueError):
        processor.chunk_text("Some text.", max_tokens=0)


def test_extract_pages_skips_unreadable_pages(monkeypatch, processor: PDFProcessor) -> None:
    reader = _FakeReader([_FakePage("first"), _FakePage(RuntimeError("boom")), _FakePage("third")])
    monkeypatch.setattr(pdf_processor, "PdfReader", lambda _source: reader)

    pages = processor.extract_pages(io.BytesIO(b"%PDF-fake"))

    assert pages == ["first", "", "third"]


def test_extract_pages_raises_on_unreadable_file(monkeypatch, processor: PDFProcessor) -> None:
    def explode(_source):
        raise ValueError("not a pdf")

    monkeypatch.setattr(pdf_processor, "PdfReader", explode)
    with pytest.raises(PDFError, match="Cannot open PDF"):
        processor.extract_pages(io.BytesIO(b"garbage"))


def test_extract_pages_rejects_password_protected(monkeypatch, processor: PDFProcessor) -> None:
    class _Locked(_FakeReader):
        def decrypt(self, _password: str) -> None:
            raise ValueError("password required")

    monkeypatch.setattr(pdf_processor, "PdfReader", lambda _s: _Locked([], encrypted=True))
    with pytest.raises(PDFError, match="password protected"):
        processor.extract_pages(io.BytesIO(b"%PDF-fake"))


def test_missing_file_raises(processor: PDFProcessor, tmp_path) -> None:
    with pytest.raises(PDFError, match="File not found"):
        processor.extract_pages(tmp_path / "nope.pdf")


def test_process_builds_document(monkeypatch, processor: PDFProcessor) -> None:
    reader = _FakeReader([_FakePage("Alpha beta. " * 50), _FakePage("Gamma delta.")])
    monkeypatch.setattr(pdf_processor, "PdfReader", lambda _s: reader)

    document = processor.process(io.BytesIO(b"%PDF-fake"), name="sample.pdf", max_tokens=60)

    assert document.name == "sample.pdf"
    assert document.page_count == 2
    assert document.chunks
    assert document.tokens == sum(chunk.tokens for chunk in document.chunks)
    assert "Alpha beta." in document.text
