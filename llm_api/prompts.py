from __future__ import annotations

from typing import Literal

PromptStrategy = Literal["zero_shot", "few_shot", "chain_of_thought"]


_SYSTEM_INSTRUCTIONS = (
    "You simplify English texts for learners. "
    "Output ONLY the simplified text (no title, no bullets, no explanations). "
    "Preserve the original meaning and factual content. "
    "Do not add new facts. "
    "Keep names, numbers, and key entities consistent."
)

# JSON mode: OpenAI-style APIs require the word "JSON" in the prompt when using json_object response_format.
_SYSTEM_INSTRUCTIONS_JSON = (
    "You simplify English texts for learners. "
    "Preserve the original meaning and factual content. Do not add new facts. "
    "Keep names, numbers, and key entities consistent. "
    "Respond with valid JSON only: one object with a single key \"simplified\" (a string) "
    "whose value is the simplified passage. No markdown, no code fences, no other keys. "
    "The simplified string must not include labels like 'Simplified:' — only the passage text."
)

_FEW_SHOT_EXAMPLES = (
    "Example:\n"
    "Original: The committee convened to deliberate on the proposal.\n"
    "Simplified: The group met to talk about the plan.\n\n"
    "Example:\n"
    "Original: The drought severely impacted agricultural output.\n"
    "Simplified: The long dry period greatly reduced how much food farms produced.\n"
)

_FEW_SHOT_EXAMPLES_JSON = (
    "Example JSON object:\n"
    '{"simplified": "The group met to talk about the plan."}\n'
    "for original: The committee convened to deliberate on the proposal.\n\n"
    "Example JSON object:\n"
    '{"simplified": "The long dry period greatly reduced how much food farms produced."}\n'
    "for original: The drought severely impacted agricultural output.\n"
)


def build_simplification_messages(
    *,
    text: str,
    target_level: str,
    strategy: PromptStrategy,
    json_object: bool = True,
) -> list[dict[str, str]]:
    system = _SYSTEM_INSTRUCTIONS_JSON if json_object else _SYSTEM_INSTRUCTIONS

    if strategy == "zero_shot":
        user = f"Simplify the text for reading level: {target_level}.\n\nTEXT:\n{text}"
        if json_object:
            user += '\n\nReturn JSON: {"simplified": "<your simplified text here>"}'
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if strategy == "few_shot":
        examples = _FEW_SHOT_EXAMPLES_JSON if json_object else _FEW_SHOT_EXAMPLES
        user = (
            f"{examples}\n\n"
            f"Now simplify the next text for reading level: {target_level}.\n\n"
            f"TEXT:\n{text}"
        )
        if json_object:
            user += '\n\nReturn JSON: {"simplified": "<your simplified text here>"}'
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if strategy == "chain_of_thought":
        user = (
            f"Simplify the text for reading level: {target_level}.\n"
            "First, silently identify hard words/phrases and simplify sentence structure. "
        )
        if json_object:
            user += 'Then respond with JSON only: {"simplified": "<final simplified passage>"}.\n\n'
        else:
            user += "Then output ONLY the final simplified text.\n\n"
        user += f"TEXT:\n{text}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    raise ValueError(f"Unknown strategy: {strategy}")


def build_simplification_prompt(*, text: str, target_level: str, strategy: PromptStrategy) -> str:
    base = (
        "You simplify English texts for learners.\n"
        "Output ONLY the simplified text (no title, no bullets, no explanations).\n"
        "Preserve the original meaning and factual content.\n"
        "Do not add new facts.\n"
        "Keep names, numbers, and key entities consistent.\n"
    )

    if strategy == "zero_shot":
        return f"{base}\nSimplify the text for reading level: {target_level}.\n\nTEXT:\n{text}"

    if strategy == "few_shot":
        return (
            f"{base}\n"
            f"{_FEW_SHOT_EXAMPLES}\n\n"
            f"Now simplify the next text for reading level: {target_level}.\n\n"
            f"TEXT:\n{text}"
        )

    if strategy == "chain_of_thought":
        return (
            f"{base}\n"
            f"Simplify the text for reading level: {target_level}.\n"
            "First, silently identify hard words/phrases and simplify sentence structure. "
            "Then output ONLY the final simplified text.\n\n"
            f"TEXT:\n{text}"
        )

    raise ValueError(f"Unknown strategy: {strategy}")

