from __future__ import annotations

import os

from dotenv import load_dotenv

from .openrouter import chat_completion_text, extract_simplified_json
from .prompts import PromptStrategy, build_simplification_messages


def simplify_with_gemini(
    *,
    text: str,
    target_level: str,
    strategy: PromptStrategy = "zero_shot",
    model: str = "google/gemini-2.5-pro-preview",
    api_key_env: str = "OPENROUTER_API_KEY",
    temperature: float | None = None,
    timeout_s: float = 60.0,
    json_object: bool = True,
) -> str:
    """
    Simplify `text` using a Google Gemini model via OpenRouter.

    Uses the same chat message format as the OpenAI path. Requires
    OPENROUTER_API_KEY (or `api_key_env`). Model IDs follow OpenRouter
    (e.g. ``google/gemini-2.5-pro-preview``).

    When ``json_object`` is True (default), uses API JSON mode and parses
    ``{"simplified": "..."}``. Set False if the model rejects ``response_format``.
    """
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    messages = build_simplification_messages(
        text=text, target_level=target_level, strategy=strategy, json_object=json_object
    )
    raw = chat_completion_text(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        timeout_s=timeout_s,
        response_format={"type": "json_object"} if json_object else None,
    )
    return extract_simplified_json(raw) if json_object else raw
