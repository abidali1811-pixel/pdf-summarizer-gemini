"""Tests for export, statistics and filename helpers."""

from __future__ import annotations

import pytest

from summarizer import ChunkSummary, SummaryResult
from utils import (
    chunks_to_dataframe,
    estimate_processing_time,
    humanize_seconds,
    safe_filename,
    summary_to_text,
)


@pytest.fixture
def result() -> SummaryResult:
    return SummaryResult(
        style="executive",
        model="gemini-test",
        combined_summary="The document recommends three actions.",
        chunks=[
            ChunkSummary(1, "First part summary.", True, 400, 100),
            ChunkSummary(2, "", False, 300, 0, error="Rate limit"),
        ],
        analysis="A short report.",
        quotes=["Quote one is here.", "Quote two is here."],
    )


def test_humanize_seconds_ranges() -> None:
    assert humanize_seconds(0.4) == "under a second"
    assert humanize_seconds(12) == "12 seconds"
    assert humanize_seconds(125) == "2 min 05 s"
    assert humanize_seconds(7300) == "2 h 01 min"


def test_estimate_processing_time_scales_with_chunks() -> None:
    assert estimate_processing_time(0) == "nothing to process"
    assert "seconds" in estimate_processing_time(4)
    assert "min" in estimate_processing_time(400)


def test_report_contains_all_sections(result: SummaryResult) -> None:
    report = summary_to_text(result, "report.pdf")
    for expected in (
        "PDF SUMMARY REPORT",
        "Document      : report.pdf",
        "Summary style : Executive",
        "FINAL SUMMARY",
        "DOCUMENT ANALYSIS",
        "KEY QUOTES",
        "CHUNK SUMMARY",
        "Chunk 2 [FAILED: Rate limit]",
    ):
        assert expected in report
    assert report.endswith("\n")


def test_report_omits_optional_sections_when_absent(result: SummaryResult) -> None:
    result.analysis = None
    result.quotes = []
    report = summary_to_text(result)
    assert "DOCUMENT ANALYSIS" not in report
    assert "KEY QUOTES" not in report


def test_dataframe_one_row_per_chunk(result: SummaryResult) -> None:
    frame = chunks_to_dataframe(result)
    assert list(frame.columns) == [
        "Chunk",
        "Status",
        "Original chars",
        "Summary chars",
        "Compression",
        "Error",
    ]
    assert frame["Status"].tolist() == ["Success", "Error"]
    assert frame["Compression"].tolist() == [0.25, 0.0]


def test_safe_filename_strips_unsafe_characters() -> None:
    assert safe_filename("Q3 report: final?.pdf", "_summary.txt") == "Q3_report_final_summary.txt"
    assert safe_filename("", ".txt") == "document.txt"


def test_safe_filename_truncates_long_names() -> None:
    name = "x" * 200 + ".pdf"
    assert len(safe_filename(name, ".txt")) <= 65
