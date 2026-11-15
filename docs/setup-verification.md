# Phase 0 — Setup verification

Date: 2026-11-15
Project root: `D:\project\python\AI-Agent-Langchain` (its own git repository)
Validator: `scripts/check_setup.py` (standard library only — runs before any dependency is installed)

## Result

All 11 checks passed.

| Area | Check | Result |
| --- | --- | --- |
| Config | `.env` exists | PASS |
| Config | `GEMINI_API_KEY` set, non-placeholder, no whitespace | PASS |
| Config | `GEMINI_MODEL` set (`gemini-3.6-flash`) | PASS |
| Secrets | `.gitignore` covers `.env` | PASS |
| Secrets | `.env` ignored by git | PASS |
| Secrets | `.env` not tracked by git | PASS |
| Secrets | `.env.example` present for onboarding | PASS |
| API | Key authenticates; 49 models visible | PASS |
| API | `gemini-3.6-flash` available to this key | PASS |
| API | Live `generateContent` returns `'OK'` in 0.9 s | PASS |

## GitHub authentication

| Check | Result |
| --- | --- |
| Credential stored for `github.com` | PASS — Windows Credential Manager, `LegacyGeneric:target=git:https://github.com` |
| Token authenticates via API | PASS — identity `abidali1811-pixel` (User) |
| Token scopes | `gist`, `repo`, `workflow` — sufficient to push and create repos |
| Remote repository exists | PASS — created private via API after the first push returned `Repository not found` |
| First push (`main`) | PASS — `[new branch] main -> main` |
| Remote tree contains `.env` | No — only `.env.example` |

No interactive login was needed: an existing Git Credential Manager entry for the
account was reused, so no browser prompt appeared.

## Environment

| Item | Value |
| --- | --- |
| Python | 3.14.7 (`.venv`, no packages installed yet) |
| Git | 2.45.0.windows.1 |
| GitHub account | `abidali1811-pixel` |
| Remote | `https://github.com/abidali1811-pixel/pdf-summarizer-gemini.git` |
| Auth method | HTTPS + Git Credential Manager (`credential.helper=manager`) |
| SSH | Not configured (`git@github.com` → `Permission denied (publickey)`), therefore unused |

## Findings that affect later phases

1. **Gemini 3.x thinking tokens.** A request with `maxOutputTokens=32` returned an
   **empty** response with `finishReason=MAX_TOKENS` because 28 of 32 tokens went to
   internal "thinking". Fixes, both verified working on `gemini-3.6-flash`:
   - `thinkingConfig: {thinkingBudget: 0}` → 1 output token, ~0.9 s (used for the gate check)
   - a realistic `maxOutputTokens` (≥ 512) with thinking left on

   Phase 3 must set an explicit output budget and a `thinkingBudget` policy; the
   summarizer must never treat an empty `parts` list as a valid summary.
2. **Model list must not be hardcoded.** The reference project pins
   `gemini-3.1/3.5-flash-lite`, none of which are the configured model. Phase 1 adds
   `config.py`, and the UI populates its selector from `ListModels` filtered to
   `generateContent` support.
3. **Empty virtualenv.** Dependencies are installed in Phase 1. If `tiktoken` or
   `langchain-google-genai` have no Python 3.14 wheels, the venv is rebuilt on 3.12.
4. **Nested repository.** The parent `D:\project\python` is a separate, commit-less
   repository with unrelated staged files. This project is its own repository, so
   nothing from the parent can be published by accident.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\check_setup.py            # full gate, needs network
.\.venv\Scripts\python.exe scripts\check_setup.py --offline  # config + secret hygiene only
```

Exit code `0` = green, `1` = at least one failed check.
