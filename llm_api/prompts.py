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

_FEW_SHOT_EXAMPLES = (
    "Example:\n"
    "Original: The committee convened to deliberate on the proposal.\n"
    "Simplified: The group met to talk about the plan.\n\n"
    "Example:\n"
    "Original: The drought severely impacted agricultural output.\n"
    "Simplified: The long dry period greatly reduced how much food farms produced.\n"
)


def build_simplification_messages(
    *, text: str, target_level: str, strategy: PromptStrategy
) -> list[dict[str, str]]:
    if strategy == "zero_shot":
        user = f"Simplify the text for reading level: {target_level}.\n\nTEXT:\n{text}"
        return [{"role": "system", "content": _SYSTEM_INSTRUCTIONS}, {"role": "user", "content": user}]

    if strategy == "few_shot":
        user = (
            f"{_FEW_SHOT_EXAMPLES}\n\n"
            f"Now simplify the next text for reading level: {target_level}.\n\n"
            f"TEXT:\n{text}"
        )
        return [{"role": "system", "content": _SYSTEM_INSTRUCTIONS}, {"role": "user", "content": user}]

    if strategy == "chain_of_thought":
        user = (
            f"Simplify the text for reading level: {target_level}.\n"
            "First, silently identify hard words/phrases and simplify sentence structure. "
            "Then output ONLY the final simplified text.\n\n"
            f"TEXT:\n{text}"
        )
        return [{"role": "system", "content": _SYSTEM_INSTRUCTIONS}, {"role": "user", "content": user}]

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

