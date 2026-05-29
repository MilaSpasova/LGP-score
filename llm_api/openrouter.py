from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError as e: 
    raise ModuleNotFoundError(
        "Missing dependency 'openai'. Install project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def extract_simplified_json(content: str) -> str:
    """
    Parse assistant content as JSON ``{\"simplified\": \"...\"}``.
    Strips optional markdown fences if the model adds them anyway.
    """
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)

    try:
        obj: Any = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("JSON response must be an object with key 'simplified'.")
    raw = obj.get("simplified")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("JSON must contain non-empty string field 'simplified'.")
    return raw.strip()


def openrouter_client(*, api_key: str, timeout_s: float = 60.0) -> OpenAI:
    """OpenAI SDK client pointed at OpenRouter (Chat Completions API)."""
    kwargs: dict = {
        "api_key": api_key,
        "base_url": OPENROUTER_BASE_URL,
        "timeout": timeout_s,
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer or title:
        headers: dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def chat_completion_text(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    timeout_s: float = 60.0,
    response_format: dict[str, str] | None = None,
    max_tokens: int | None = 1200,
) -> str:
    """Run a chat completion on OpenRouter and return non-empty assistant text."""
    client = openrouter_client(api_key=api_key, timeout_s=timeout_s)
    create_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if top_p is not None:
        create_kwargs["top_p"] = top_p
    if seed is not None:
        create_kwargs["seed"] = seed
    if response_format is not None:
        create_kwargs["response_format"] = response_format
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**create_kwargs)

    content: Optional[str] = None
    if resp.choices and resp.choices[0].message:
        content = resp.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("OpenRouter returned empty content.")

    return content.strip()
