# 📄 PDF AI Summarizer

Upload a PDF, pick a summary style, and get a structured summary — built end to end
with **LangChain** and **Google Gemini**, served by **Streamlit**.

Text is extracted page by page, cleaned, split into token-bounded chunks, summarized
in parallel (map), then reduced into one coherent summary (reduce). Document analysis
and verbatim key quotes are optional extras.

```
PDF ──► extract ──► clean ──► chunk (by tokens)
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
          summary 1           summary 2    …     summary N        map (parallel)
              └───────────────────┼───────────────────┘
                                  ▼
                      hierarchical combine ──► final summary     reduce
                                  │
                       analysis + key quotes (optional)
```

## Quick start

Requires Python 3.10+ (verified on 3.14.7).

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env              # then put your key in .env
python scripts/check_setup.py     # verifies key, model, git hygiene
streamlit run app.py
```

PowerShell users can use the equivalent Windows commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # then put your key in .env
.\.venv\Scripts\python.exe scripts\check_setup.py   # verifies key, model, git hygiene
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Get a free key at <https://aistudio.google.com/apikey>.

```ini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Any model your key can access works — the sidebar lists them **live from the API**
rather than from a hardcoded list.

### Streamlit Community Cloud

The local `.env` file is not deployed with the app. In the app dashboard, open
**Settings > Secrets** and add:

```toml
GEMINI_API_KEY = "your-new-gemini-api-key"
GEMINI_MODEL = "gemini-3.6-flash"
```

Save the secrets and restart/reboot the app if it does not reload automatically.
Keep the key in Streamlit secrets; do not commit it to the repository.

## Command line

The same pipeline without Streamlit:

```powershell
.\.venv\Scripts\python.exe scripts\make_sample_pdf.py                 # creates sample_pdfs/sample.pdf
.\.venv\Scripts\python.exe scripts\summarize_cli.py sample_pdfs\sample.pdf --style brief
.\.venv\Scripts\python.exe scripts\summarize_cli.py report.pdf --style executive --analysis --quotes --out summary.txt
```

Exit code is `1` if any chunk failed, so it is safe to use in scripts and CI.

## Summary styles

| Style | Output |
| --- | --- |
| `brief` | 2–3 sentences: main point and the single most important takeaway |
| `detailed` | Topics, arguments/findings, details, conclusions |
| `bullet_points` | Scannable, grouped bullets |
| `executive` | Insights, implications, recommendations, risks — for decision makers |

To add a style, add an entry to `STYLES` and `STYLE_LABELS` in `prompts.py`.

## Project structure

```
app.py               Streamlit UI: settings, upload, progress, results, downloads
config.py            One source of truth: model, token budgets, limits, .env loading
pdf_processor.py     Extraction, cleaning, token counting, chunking (pure, testable)
prompts.py           System persona, styles, and all ChatPromptTemplates
summarizer.py        Gemini chains: map, hierarchical reduce, analysis, quotes
utils.py             Timing estimates, text/CSV export, filenames (pure)
scripts/check_setup.py     Phase 0 gate: key, model, secret hygiene
scripts/make_sample_pdf.py Generates a valid demo PDF with no dependencies
scripts/summarize_cli.py   Headless summarizer
tests/               48 tests, fake LLMs only — no network
docs/setup-verification.md Recorded environment and auth verification
```

## Testing

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The suite never calls the API: chains are exercised with
`FakeListChatModel` and `RunnableLambda`, the PDF layer with a stubbed
`PdfReader`, and the UI with `streamlit.testing.AppTest`.

## How it behaves under failure

- A chunk that fails (rate limit, safety block, empty answer) becomes an **error row**;
  the rest of the document still summarizes, and the UI lists what failed.
- If every chunk fails, the run stops with an explicit message instead of a fake summary.
- Very long documents are reduced in rounds (max 8 summaries per call) so the combine
  prompt never overflows.
- Password-protected or image-only PDFs produce a clear error instead of an empty result.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `GEMINI_API_KEY is not set` | `.env` missing next to `app.py`; copy `.env.example` |
| Empty summary / `finishReason=MAX_TOKENS` | Gemini 3.x spent the output budget on *thinking*. `config.py` sets `thinking_budget = 0`; if you change it, keep `output_tokens` ≥ 512 |
| `429` / quota errors | Wait a minute, lower "Max tokens per chunk", or switch model |
| `404` model not found | Pick another model in the sidebar — availability is per key |
| "No extractable text — the PDF may be scanned images" | Run OCR first (e.g. `ocrmypdf`) |
| `temperature … will be ignored` | That model uses fixed sampling; `config.TEMPERATURE` is `None` so the parameter is omitted |

## Notes

- `.env` is git-ignored; only `.env.example` is committed.
- Requests run up to 4 at a time (`config.MAX_CONCURRENCY`).
- Analysis and quotes read the first ~32k characters of a document to stay within
  one request.
