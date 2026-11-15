"""Phase 0 setup validation for the PDF summarizer.

Checks, in order:

1. `.env` exists and defines a usable `GEMINI_API_KEY` (not a placeholder)
2. `.env` is ignored by git and not tracked
3. `GEMINI_MODEL` is set and the key can list models
4. the configured model is actually available to this key
5. one tiny live `generateContent` round-trip succeeds

Uses only the Python standard library: it must work before any dependency is
installed. The API key is never printed or written anywhere.

Usage:
    python scripts/check_setup.py           # full check (network)
    python scripts/check_setup.py --offline # skip network calls
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 30
PROMPT = "Reply with exactly one word: OK"
PLACEHOLDERS = {
    "",
    "your_gemini_api_key_here",
    "your-key",
    "changeme",
    "your_api_key",
}

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"\n       {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def parse_env(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser; mirrors python-dotenv for the cases we use."""
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        data[key] = value.strip().strip('"').strip("'")
    return data


def redact(secret: str) -> str:
    if len(secret) <= 10:
        return "*" * len(secret)
    return f"{secret[:6]}...{secret[-4:]}"


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def http_json(
    method: str, url: str, key: str, payload: dict | None = None
) -> tuple[int, dict]:
    """Call the Gemini API. Raises HTTPError with a decoded body for 4xx/5xx."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "x-goog-api-key": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.status, json.loads(response.read().decode("utf-8") or "{}")


def explain_error(exc: Exception) -> str:
    """Turn an API/network failure into an actionable instruction."""
    if isinstance(exc, urllib.error.HTTPError):
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(raw).get("error", {})
        except json.JSONDecodeError:
            error = {}
        status = error.get("status", "")
        message = error.get("message", raw[:200])
        hints = {
            "UNAUTHENTICATED": "the key is malformed or revoked -- regenerate it at "
            "https://aistudio.google.com/apikey",
            "INVALID_ARGUMENT": "the key was rejected (API_KEY_INVALID) -- check for "
            "extra spaces or a truncated copy",
            "PERMISSION_DENIED": "the key is restricted (API not enabled or wrong "
            "project) -- enable the Generative Language API on that project",
            "NOT_FOUND": "the model does not exist for this API version -- fix "
            "GEMINI_MODEL in .env",
            "RESOURCE_EXHAUSTED": "quota or rate limit hit -- wait a minute, or use a "
            "different model",
        }
        hint = hints.get(status, "")
        return f"HTTP {exc.code} {status}: {message}" + (f"\n       -> {hint}" if hint else "")
    if isinstance(exc, urllib.error.URLError):
        return f"network error: {exc.reason} -- check internet access / proxy / firewall"
    return f"{type(exc).__name__}: {exc}"


def available_models(key: str) -> list[str]:
    models: list[str] = []
    page_token = ""
    while True:
        url = f"{API_ROOT}/models?pageSize=100"
        if page_token:
            url += f"&pageToken={page_token}"
        _, payload = http_json("GET", url, key)
        for model in payload.get("models", []):
            name = str(model.get("name", "")).removeprefix("models/")
            methods = model.get("supportedGenerationMethods", [])
            if name and "generateContent" in methods:
                models.append(name)
        page_token = payload.get("nextPageToken") or ""
        if not page_token:
            return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="skip all network calls"
    )
    args = parser.parse_args()

    print("Phase 0 setup validation")
    print(f"project root: {ROOT}")

    # 1. config -------------------------------------------------------------
    section("Configuration")
    env_exists = record(".env exists", ENV_FILE.exists(), str(ENV_FILE))
    env = parse_env(ENV_FILE)

    api_key = env.get("GEMINI_API_KEY", "").strip()
    key_ok = record(
        "GEMINI_API_KEY is set",
        bool(api_key),
        f"{redact(api_key)} ({len(api_key)} chars)" if api_key else "missing from .env",
    )
    if key_ok:
        placeholder = api_key.lower() in PLACEHOLDERS or api_key.startswith("your")
        key_ok = record("GEMINI_API_KEY is not a placeholder", not placeholder)
        if " " in api_key:
            key_ok = record("GEMINI_API_KEY has no whitespace", False, "strip spaces/quotes")

    model = env.get("GEMINI_MODEL", "").strip()
    record("GEMINI_MODEL is set", bool(model), model or "missing from .env")

    # 2. secret hygiene -----------------------------------------------------
    section("Secret hygiene")
    ignore_rules: list[str] = []
    ignore_file = ROOT / ".gitignore"
    if ignore_file.exists():
        ignore_rules = [
            line.strip()
            for line in ignore_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    record(
        ".gitignore covers .env",
        any(rule in {".env", "*env"} or rule.startswith(".env") for rule in ignore_rules),
    )
    if (ROOT / ".git").exists():
        ignored = git("check-ignore", "-q", ".env")
        record(".env is ignored by git", ignored.returncode == 0)
        tracked = git("ls-files", "--error-unmatch", ".env")
        record(".env is not tracked by git", tracked.returncode != 0)
        if (ROOT / ".env.example").exists():
            record(".env.example exists for onboarding", True)
    else:
        print("[SKIP] not a git repository yet")

    # 3. network ------------------------------------------------------------
    if args.offline:
        section("API")
        print("[SKIP] --offline given, no network calls made")
    elif not (key_ok and env_exists):
        section("API")
        print("[SKIP] fix the configuration above first")
    else:
        section("API")
        models: list[str] = []
        try:
            started = time.perf_counter()
            models = available_models(api_key)
            elapsed = time.perf_counter() - started
            record(
                "key authenticates with the Gemini API",
                True,
                f"{len(models)} models available ({elapsed:.1f}s)",
            )
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            record("key authenticates with the Gemini API", False, explain_error(exc))

        if models:
            flash = [m for m in models if "flash" in m]
            print(f"       flash models: {', '.join(flash[:8]) or 'none'}")
            model_ok = record(
                f"model '{model}' is available to this key",
                model in models,
                "" if model in models else f"pick one of: {', '.join(models[:8])}",
            )
            if model_ok:
                try:
                    started = time.perf_counter()
                    _, payload = http_json(
                        "POST",
                        f"{API_ROOT}/models/{model}:generateContent",
                        api_key,
                        {
                            "contents": [{"parts": [{"text": PROMPT}]}],
                            "generationConfig": {
                                # Gemini 3.x spends output tokens on "thinking" first;
                                # a tiny budget yields empty text with finishReason
                                # MAX_TOKENS. 512 leaves room for both.
                                "maxOutputTokens": 512,
                                "temperature": 0,
                                # Thinking off: one word is the whole task.
                                "thinkingConfig": {"thinkingBudget": 0},
                            },
                        },
                    )
                    elapsed = time.perf_counter() - started
                    candidate = payload.get("candidates", [{}])[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    finish = candidate.get("finishReason", "?")
                    usage = payload.get("usageMetadata", {})
                    record(
                        "live generateContent call succeeds",
                        bool(text),
                        f"{elapsed:.1f}s -> {text[:60]!r} (finish={finish})",
                    )
                    if not text and finish == "MAX_TOKENS":
                        print(
                            "       -> output budget consumed by thinking tokens; "
                            "raise maxOutputTokens or set thinkingBudget=0"
                        )
                    if usage:
                        print(
                            f"       tokens: {usage.get('promptTokenCount', '?')} in / "
                            f"{usage.get('candidatesTokenCount', '?')} out / "
                            f"{usage.get('thoughtsTokenCount', 0)} thinking"
                        )
                except Exception as exc:  # noqa: BLE001
                    record("live generateContent call succeeds", False, explain_error(exc))

    # summary ---------------------------------------------------------------
    failures = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 64)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"RESULT: all {len(results)} checks passed -- Phase 0 gate is green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
