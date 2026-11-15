"""Generate a small, valid, text-based PDF for demos and end-to-end runs.

No third-party libraries: the PDF is assembled by hand so the sample can be
regenerated anywhere.

Usage:
    python scripts/make_sample_pdf.py [output.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "sample_pdfs" / "sample.pdf"

PAGES: list[list[str]] = [
    [
        "Quarterly Operations Review",
        "",
        "Prepared by the Operations team for the leadership meeting.",
        "",
        "1. Summary of the quarter",
        "",
        "Throughput rose by eleven percent compared with the previous quarter while",
        "the defect rate stayed flat at two percent. The two new regional depots",
        "absorbed most of the increase, and overtime hours fell by a third after",
        "the scheduling change went live in the second month.",
        "",
        "2. Cost position",
        "",
        "Total operating cost increased by four percent, driven mainly by fuel and",
        "by the depot leases signed in March. Cost per delivered unit fell slightly,",
        "so the increase reflects higher volume rather than weaker efficiency.",
        "",
        "3. Key risks",
        "",
        "Two suppliers account for sixty percent of packaging spend. A disruption at",
        "either supplier would delay roughly one week of shipments. The team also",
        "flagged that the current forecasting model under-predicts demand spikes",
        "during public holidays.",
    ],
    [
        "4. Recommendations",
        "",
        "First, qualify a third packaging supplier before the next peak season to",
        "reduce single-source exposure. Second, replace the holiday forecasting",
        "heuristics with the model piloted in the eastern region, which reduced",
        "stock-outs by eighteen percent. Third, hold depot lease renewals until the",
        "utilisation study completes in six weeks.",
        "",
        "5. Decisions requested",
        "",
        "Approve the supplier qualification budget of forty thousand. Endorse the",
        "forecasting pilot as the standard from the next quarter. Note the lease",
        "moratorium and the review date.",
        "",
        "Prepared for discussion. Figures are provisional and subject to finance",
        "sign-off at month end.",
    ],
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 12 Tf", "14 TL", "54 738 Td"]
    for line in lines:
        parts.append(f"({_escape(line)}) Tj T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: dict[int, bytes] = {}
    kids: list[int] = []

    for index, lines in enumerate(pages):
        page_number = 4 + index * 2
        content_number = page_number + 1
        kids.append(page_number)
        objects[page_number] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_number} 0 R >>"
        ).encode("latin-1")
        stream = _content_stream(lines)
        objects[content_number] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{kid} 0 R' for kid in kids)}] "
        f"/Count {len(kids)} >>"
    ).encode("latin-1")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode("latin-1")
        output += objects[number]
        output += b"\nendobj\n"

    xref_offset = len(output)
    highest = max(objects)
    output += f"xref\n0 {highest + 1}\n".encode("latin-1")
    output += b"0000000000 65535 f \n"
    for number in range(1, highest + 1):
        output += f"{offsets[number]:010d} 00000 n \n".encode("latin-1")
    output += (
        f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(output)


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build_pdf(PAGES)
    target.write_bytes(data)
    print(f"wrote {target} ({len(data):,} bytes, {len(PAGES)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
