"""Summarize a PDF from the command line (no Streamlit).

Handy for end-to-end checks and for scripting:

    python scripts/summarize_cli.py sample_pdfs/sample.pdf --style brief
    python scripts/summarize_cli.py report.pdf --style executive --analysis --quotes --out summary.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, clamp_chunk_tokens, get_settings  # noqa: E402
from pdf_processor import PDFError, PDFProcessor  # noqa: E402
from prompts import STYLES  # noqa: E402
from summarizer import PDFSummarizer, SummarizerError  # noqa: E402
from utils import summary_to_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a PDF with Gemini via LangChain")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--style", choices=sorted(STYLES), default="detailed")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=4000,
        help=f"tokens per chunk ({MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS})",
    )
    parser.add_argument("--analysis", action="store_true", help="also analyse structure")
    parser.add_argument("--quotes", action="store_true", help="also extract key quotes")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    args = parser.parse_args()

    settings = get_settings(args.model, clamp_chunk_tokens(args.chunk_tokens))
    if not settings.has_api_key:
        print("GEMINI_API_KEY is not set. Copy .env.example to .env first.", file=sys.stderr)
        return 2
    if not args.pdf.exists():
        print(f"No such file: {args.pdf}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        document = PDFProcessor().process(
            args.pdf, name=args.pdf.name, max_tokens=settings.chunk_tokens
        )
    except PDFError as exc:
        print(f"PDF error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{document.name}: {document.page_count} pages, {document.chars:,} chars, "
        f"{len(document.chunks)} chunks, model {settings.model}"
    )

    def on_progress(fraction: float, message: str) -> None:
        print(f"  [{fraction:>4.0%}] {message}")

    try:
        summarizer = PDFSummarizer(settings=settings)
        result = summarizer.run(
            document,
            style=args.style,
            include_analysis=args.analysis,
            include_quotes=args.quotes,
            progress=on_progress,
        )
    except SummarizerError as exc:
        print(f"Summarizer error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    report = summary_to_text(result, document.name)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"\nreport written to {args.out} ({len(report):,} chars) in {elapsed:.1f}s")
    else:
        print()
        print(report)

    if result.failed_chunks:
        print(
            f"\n{len(result.failed_chunks)} of {result.total_chunks} chunks failed:",
            file=sys.stderr,
        )
        for chunk in result.failed_chunks:
            print(f"  chunk {chunk.chunk_number}: {chunk.error}", file=sys.stderr)
        return 1
    print(f"\nfinished in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
