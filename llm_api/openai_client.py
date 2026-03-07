from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'openai'. Install project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e


from .prompts import PromptStrategy, build_simplification_messages


def simplify_with_openai(
    *,
    text: str,
    target_level: str,
    strategy: PromptStrategy = "zero_shot",
    model: str = "gpt-5.2-chat-latest",
    api_key_env: str = "OPENAI_API_KEY",
    timeout_s: float = 60.0,
) -> str:
    """
    Simplify `text` using an OpenAI chat model (GPT-5.2 or newer).

    Requires environment variable OPENAI_API_KEY (or `api_key_env`).
    """
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    client = OpenAI(api_key=api_key, timeout=timeout_s)
    messages = build_simplification_messages(text=text, target_level=target_level, strategy=strategy)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    content: Optional[str] = None
    if resp.choices and resp.choices[0].message:
        content = resp.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("OpenAI returned empty content.")

    return content.strip()

