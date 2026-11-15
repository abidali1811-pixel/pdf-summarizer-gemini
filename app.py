"""Streamlit UI for the PDF summarizer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from config import (
    APP_ICON,
    APP_TITLE,
    MAX_CONCURRENCY,
    MAX_UPLOAD_MB,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    clamp_chunk_tokens,
    get_settings,
)
from pdf_processor import Document, PDFError, PDFProcessor
from prompts import STYLE_LABELS
from summarizer import PDFSummarizer, SummarizerError, pick_models
from utils import (
    STYLE_LABELS as STYLE_TITLES,
    chunks_to_dataframe,
    estimate_processing_time,
    safe_filename,
    summary_to_text,
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


@st.cache_resource(show_spinner=False)
def _processor() -> PDFProcessor:
    return PDFProcessor()


@st.cache_data(ttl=600, show_spinner=False)
def _model_options(api_key: str, preferred: str) -> list[str]:
    return pick_models(api_key, preferred)


def _document_for(data: bytes, name: str, chunk_tokens: int) -> Document:
    """Parse a PDF, reusing the result while the file and chunk size are unchanged."""
    signature = hashlib.sha256(data).hexdigest()[:16] + f":{chunk_tokens}"
    cached = st.session_state.get("document")
    if cached and st.session_state.get("document_signature") == signature:
        return cached

    document = _processor().process(io.BytesIO(data), name=name, max_tokens=chunk_tokens)
    st.session_state.document = document
    st.session_state.document_signature = signature
    st.session_state.pop("result", None)
    st.session_state.pop("result_signature", None)
    return document


def _render_summary(result, document: Document) -> None:
    left, middle, right, far = st.columns(4)
    left.metric("Pages", document.page_count)
    middle.metric("Chunks", result.total_chunks)
    right.metric(
        "Chunks summarised",
        f"{result.successful_chunks}/{result.total_chunks}",
        delta=None if not result.failed_chunks else f"{len(result.failed_chunks)} failed",
        delta_color="off" if not result.failed_chunks else "inverse",
    )
    far.metric("Compression", f"{document.chars and round(document.chars / max(1, len(result.combined_summary)), 1)}x")

    st.subheader("Final summary")
    st.markdown(result.combined_summary)

    if result.failed_chunks:
        with st.expander(f"⚠ {len(result.failed_chunks)} chunk(s) failed", expanded=False):
            for chunk in result.failed_chunks:
                st.write(f"**Chunk {chunk.chunk_number}** — {chunk.error}")
            st.caption(
                "The final summary was built from the chunks that succeeded. "
                "Retry, reduce chunk size, or pick a different model."
            )

    tabs = [f"Chunk details ({result.total_chunks})"]
    if result.analysis:
        tabs.insert(0, "Document analysis")
    if result.quotes:
        tabs.insert(1, f"Key quotes ({len(result.quotes)})")

    panels = st.tabs(tabs)
    panel_map = dict(zip(tabs, panels))

    if result.analysis:
        with panel_map["Document analysis"]:
            st.markdown(result.analysis)
    if result.quotes:
        with panel_map[f"Key quotes ({len(result.quotes)})"]:
            for quote in result.quotes:
                st.markdown(f"> {quote}")

    with panel_map[f"Chunk details ({result.total_chunks})"]:
        for chunk in result.chunks:
            label = f"Chunk {chunk.chunk_number} · {chunk.original_chars:,} → {chunk.summary_chars:,} chars"
            with st.expander(label, expanded=False):
                if chunk.ok:
                    st.markdown(chunk.summary)
                else:
                    st.error(chunk.error)
        st.dataframe(
            chunks_to_dataframe(result),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Download")
    download_left, download_right = st.columns(2)
    report = summary_to_text(result, document.name)
    download_left.download_button(
        "⬇ Summary report (.txt)",
        data=report,
        file_name=safe_filename(document.name, "_summary.txt"),
        mime="text/plain",
        width="stretch",
    )
    download_right.download_button(
        "⬇ Chunk statistics (.csv)",
        data=chunks_to_dataframe(result).to_csv(index=False).encode("utf-8"),
        file_name=safe_filename(document.name, "_chunks.csv"),
        mime="text/csv",
        width="stretch",
    )


def main() -> None:
    try:
        settings = get_settings(secrets=st.secrets)
    except StreamlitSecretNotFoundError:
        settings = get_settings()
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Upload a PDF and get a structured summary, powered by Google Gemini via LangChain.")

    if not settings.has_api_key:
        st.error("GEMINI_API_KEY is not configured.")
        st.info(
            "For Streamlit Community Cloud, add GEMINI_API_KEY under "
            "App settings > Secrets. For local runs, set it in .env."
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Model",
            _model_options(settings.api_key, settings.model),
            index=0,
            help="Models are listed live from your API key",
        )
        style = st.selectbox(
            "Summary style",
            list(STYLE_LABELS),
            index=1,
            format_func=lambda key: STYLE_TITLES.get(key, key),
        )
        chunk_tokens = clamp_chunk_tokens(
            st.slider(
                "Max tokens per chunk",
                MIN_CHUNK_TOKENS,
                MAX_CHUNK_TOKENS,
                settings.chunk_tokens,
                step=500,
                help="Smaller chunks give more detail but more requests",
            )
        )
        include_analysis = st.checkbox("Analyse document structure", value=True)
        include_quotes = st.checkbox("Extract key quotes", value=False)
        st.caption(f"Up to {MAX_CONCURRENCY} requests in parallel")

    uploaded = st.file_uploader(
        "Choose a PDF", type=["pdf"], help=f"Maximum {MAX_UPLOAD_MB} MB"
    )
    if uploaded is None:
        st.info("Upload a PDF to get started. Text-based PDFs only — scanned images need OCR first.")
        return

    data = uploaded.getvalue()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"{uploaded.name} is larger than {MAX_UPLOAD_MB} MB.")
        st.stop()

    try:
        document = _document_for(data, uploaded.name, chunk_tokens)
    except PDFError as exc:
        st.error(str(exc))
        st.stop()

    info_left, info_right, info_third = st.columns(3)
    info_left.metric("Pages", document.page_count)
    info_right.metric("Characters", f"{document.chars:,}")
    info_third.metric("Chunks", len(document.chunks))
    if document.empty_pages:
        st.caption(
            f"{len(document.empty_pages)} page(s) had no extractable text "
            f"(likely images): {', '.join(map(str, document.empty_pages[:10]))}"
        )

    signature = (
        f"{uploaded.name}:{len(data)}:{chunk_tokens}:{model}:{style}:"
        f"{include_analysis}:{include_quotes}"
    )
    previous = st.session_state.get("result")
    if previous and st.session_state.get("result_signature") != signature:
        st.warning("Settings or file changed — summarise again to refresh the results below.")
    elif previous:
        _render_summary(previous, document)
        if st.button("Summarise again", width="stretch"):
            st.session_state.pop("result", None)
            st.rerun()
        return

    estimate = estimate_processing_time(len(document.chunks))
    st.caption(f"Estimated time: {estimate} · model `{model}`")
    if not st.button("Summarise", type="primary", width="stretch"):
        return

    progress_bar = st.progress(0.0, text="Starting…")

    def on_progress(fraction: float, message: str) -> None:
        progress_bar.progress(min(1.0, max(0.0, fraction)), text=message)

    try:
        summarizer = PDFSummarizer(model, settings=get_settings(model, chunk_tokens))
        result = summarizer.run(
            document,
            style=style,
            include_analysis=include_analysis,
            include_quotes=include_quotes,
            progress=on_progress,
        )
    except SummarizerError as exc:
        progress_bar.empty()
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected
        progress_bar.empty()
        st.error(f"Summarisation failed: {type(exc).__name__}: {exc}")
        return

    progress_bar.empty()
    st.session_state.result = result
    st.session_state.result_signature = signature
    st.rerun()


if __name__ == "__main__":
    main()
