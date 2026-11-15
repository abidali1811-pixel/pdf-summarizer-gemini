"""PDF text extraction, cleaning and token-aware chunking.

The pipeline is deliberately pure: no Streamlit, no network. Everything here is
unit-testable with plain objects, which is why it lives apart from the UI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"[ \t]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class PDFError(RuntimeError):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass
class Chunk:
    """One slice of the document, sized in tokens."""

    index: int
    text: str
    tokens: int

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Document:
    """A processed PDF: raw pages plus the chunks derived from them."""

    name: str
    pages: list[str] = field(default_factory=list)
    text: str = ""
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def tokens(self) -> int:
        return sum(chunk.tokens for chunk in self.chunks)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def empty_pages(self) -> list[int]:
        """1-based numbers of pages that produced no text."""
        return [i for i, page in enumerate(self.pages, start=1) if not page.strip()]


class PDFProcessor:
    """Extract, clean and chunk the text of a PDF."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    # -- extraction --------------------------------------------------------
    def extract_pages(self, pdf_file: str | Path | BinaryIO) -> list[str]:
        """Return the text of every page. Unreadable pages yield an empty string."""
        source: str | Path | BinaryIO = pdf_file
        if isinstance(source, (str, Path)):
            if not Path(source).exists():
                raise PDFError(f"File not found: {source}")

        if hasattr(source, "seek"):
            source.seek(0)

        try:
            reader = PdfReader(source)
        except Exception as exc:  # noqa: BLE001 - pypdf raises many types
            raise PDFError(f"Cannot open PDF: {exc}") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise PDFError("PDF is password protected") from exc

        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - skip bad pages, keep going
                logger.warning("Skipping unreadable page %d: %s", number, exc)
                pages.append("")
        return pages

    def extract_text(self, pdf_file: str | Path | BinaryIO) -> str:
        return "\n\n".join(self.extract_pages(pdf_file))

    # -- cleaning ----------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Drop control characters, collapse runs of spaces, keep paragraphs."""
        text = _CONTROL_CHARS.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _WHITESPACE.sub(" ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Words split across a line break by hyphenation: "sum-\nmarise" -> "summarise"
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        return text.strip()

    # -- sizing ------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    # -- chunking ----------------------------------------------------------
    def chunk_text(self, text: str, max_tokens: int = 8000) -> list[Chunk]:
        """Split text into chunks of at most `max_tokens`, preferring sentence edges.

        Sentences longer than a whole chunk are hard-split on token boundaries so
        no chunk can exceed the budget.
        """
        if not text.strip():
            raise PDFError("No extractable text — the PDF may be scanned images")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        pieces: list[tuple[str, int]] = []
        for sentence in _SENTENCE_END.split(text.strip()):
            if not sentence:
                continue
            tokens = self._encoding.encode(sentence, disallowed_special=())
            if len(tokens) <= max_tokens:
                pieces.append((sentence, len(tokens)))
                continue
            for start in range(0, len(tokens), max_tokens):
                window = tokens[start : start + max_tokens]
                pieces.append((self._encoding.decode(window), len(window)))

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffered_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffered_tokens
            if buffer:
                chunks.append(
                    Chunk(index=len(chunks) + 1, text=" ".join(buffer).strip(), tokens=buffered_tokens)
                )
                buffer, buffered_tokens = [], 0

        for piece, tokens in pieces:
            # +1 for the joining space
            if buffer and buffered_tokens + tokens + 1 > max_tokens:
                flush()
            buffer.append(piece)
            buffered_tokens += tokens + (1 if len(buffer) > 1 else 0)
        flush()

        return [chunk for chunk in chunks if chunk.text]

    def process(
        self,
        pdf_file: str | Path | BinaryIO,
        name: str = "document.pdf",
        max_tokens: int = 8000,
    ) -> Document:
        """Full pipeline: extract -> clean -> chunk."""
        pages = self.extract_pages(pdf_file)
        text = self.clean_text("\n\n".join(pages))
        if not text:
            raise PDFError(
                "No extractable text — the PDF may be scanned images. Run OCR first."
            )
        chunks = self.chunk_text(text, max_tokens)
        return Document(name=name, pages=pages, text=text, chunks=chunks)
