"""Pure helpers: timing estimates, text export, statistics tables.

Deliberately free of Streamlit imports so they can be unit-tested and reused
from a CLI or notebook. Rendering lives in `app.py`.
"""

from __future__ import annotations

import re
import time
from typing import Iterable

import pandas as pd

from summarizer import SummaryResult

STYLE_LABELS = {
    "brief": "Brief",
    "detailed": "Detailed",
    "bullet_points": "Bullet points",
    "executive": "Executive",
}


def humanize_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return "under a second"
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def estimate_processing_time(
    num_chunks: int, seconds_per_chunk: float = 4.0, max_concurrency: int = 4
) -> str:
    """Rough wall-clock estimate: chunks run in parallel batches."""
    if num_chunks <= 0:
        return "nothing to process"
    batches = -(-num_chunks // max(1, max_concurrency))
    return f"about {humanize_seconds(batches * seconds_per_chunk)}"


def summary_to_text(result: SummaryResult, document_name: str = "") -> str:
    """Plain-text report: header, final summary, optional extras, chunk table."""
    labels = STYLE_LABELS.get(result.style, result.style.replace("_", " ").title())
    lines = [
        "PDF SUMMARY REPORT",
        "==================",
        "",
    ]
    if document_name:
        lines.append(f"Document      : {document_name}")
    lines += [
        f"Summary style : {labels}",
        f"Model         : {result.model}",
        f"Chunks        : {result.total_chunks} ({result.successful_chunks} succeeded)",
        f"Generated     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "-------------",
        "",
        result.combined_summary.strip(),
        "",
    ]

    if result.analysis:
        lines += ["DOCUMENT ANALYSIS", "-----------------", "", result.analysis.strip(), ""]

    if result.quotes:
        lines += ["KEY QUOTES", "----------", ""]
        lines += [f"- {quote}" for quote in result.quotes]
        lines.append("")

    lines += ["CHUNK SUMMARY", "-------------", ""]
    for chunk in result.chunks:
        status = "ok" if chunk.ok else f"FAILED: {chunk.error}"
        lines += [
            f"Chunk {chunk.chunk_number} [{status}] "
            f"{chunk.original_chars} -> {chunk.summary_chars} chars",
        ]
        if chunk.ok:
            lines += [chunk.summary.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def chunks_to_dataframe(result: SummaryResult) -> pd.DataFrame:
    """One row per chunk with lengths, compression ratio and status."""
    return pd.DataFrame(
        [
            {
                "Chunk": chunk.chunk_number,
                "Status": "Success" if chunk.ok else "Error",
                "Original chars": chunk.original_chars,
                "Summary chars": chunk.summary_chars,
                "Compression": chunk.compression,
                "Error": chunk.error,
            }
            for chunk in result.chunks
        ]
    )


def safe_filename(name: str, suffix: str) -> str:
    """Trim a document name down to something filesystem-safe."""
    stem = (name or "document").rsplit(".", 1)[0]
    cleaned = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in stem)
    cleaned = "_".join(cleaned.split())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_") or "document"
    return f"{cleaned[:60]}{suffix}"


def coerce_quotes(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        return [line.strip() for line in raw.splitlines() if line.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]
