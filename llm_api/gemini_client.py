from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

try:
    from google import genai  # type: ignore[import-not-found]
except ModuleNotFoundError:
    genai = None  # type: ignore[assignment]

deprecated_genai = None  # type: ignore[assignment]
if genai is None:
    try:
        import google.generativeai as deprecated_genai  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        deprecated_genai = None  # type: ignore[assignment]


from .prompts import PromptStrategy, build_simplification_prompt


def simplify_with_gemini(
    *,
    text: str,
    target_level: str,
    strategy: PromptStrategy = "zero_shot",
    model: str = "gemini-3.1-pro-preview",
    api_key_env: str = "GOOGLE_API_KEY",
) -> str:
    """
    Simplify `text` using a Gemini model (Gemini 3 / 3.1).

    Requires environment variable GOOGLE_API_KEY (or `api_key_env`).
    """
    load_dotenv()
    api_key = os.getenv(api_key_env) or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env} (or GEMINI_API_KEY)")

    prompt = build_simplification_prompt(text=text, target_level=target_level, strategy=strategy)

    out: Optional[str] = None

    if genai is not None:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)
        out = getattr(resp, "text", None)
    elif deprecated_genai is not None:
        deprecated_genai.configure(api_key=api_key)
        m = deprecated_genai.GenerativeModel(model_name=model)
        resp = m.generate_content(prompt)
        out = getattr(resp, "text", None)
    else:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency for Gemini. Install project requirements first:\n"
            "  python -m pip install -r requirements.txt"
        )

    if not out or not out.strip():
        raise RuntimeError("Gemini returned empty content.")

    return out.strip()

