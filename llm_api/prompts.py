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

# JSON mode: OpenAI-style APIs require the word "JSON" in the prompt when
# using json_object response_format.
_SYSTEM_INSTRUCTIONS_JSON = (
    "You simplify English texts for learners. "
    "Preserve the original meaning and factual content. Do not add new facts. "
    "Keep names, numbers, and key entities consistent. "
    "Respond with valid JSON only: one object with a single key \"simplified\" (a string) "
    "whose value is the simplified passage. No markdown, no code fences, no other keys. "
    "The simplified string must not include labels like 'Simplified:' — only the passage text."
)

# These examples are external human simplifications adapted from held-out
# sentence pairs in the ASSET dataset (Alva-Manchego et al., ACL 2020:
# https://aclanthology.org/2020.acl-main.424/; release:
# https://github.com/facebookresearch/asset). They are intentionally not
# sampled from the OneStopEnglish evaluation corpus, which avoids benchmark
# leakage while still giving the model a concrete demonstration of the
# desired simplification style.

_FEW_SHOT_EXAMPLES = (
    "Example 1:\n"
    "Original: Since 2000, the recipient of the Kate Greenaway Medal has also been presented with the Colin Mears Award to the value of £5000.\n"
    "Simplified: Since 2000, the winner of the Kate Greenaway Medal also receives the Colin Mears Award. The value of the prize is £5000.\n\n"
    "Example 2:\n"
    "Original: Admission to Tsinghua is extremely competitive.\n"
    "Simplified: Getting admitted to Tsinghua is difficult. There is a lot of competition.\n"
)

_FEW_SHOT_EXAMPLES_JSON = (
    "Example JSON object:\n"
    '{"simplified": "Since 2000, the winner of the Kate Greenaway Medal also receives the Colin Mears Award. The value of the prize is £5000."}\n'
    "for original: Since 2000, the recipient of the Kate Greenaway Medal has also been presented with the Colin Mears Award to the value of £5000.\n\n"
    "Example JSON object:\n"
    '{"simplified": "Getting admitted to Tsinghua is difficult. There is a lot of competition."}\n'
    "for original: Admission to Tsinghua is extremely competitive.\n"
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
            "First, silently identify hard words or phrases, simplify sentence structure, "
            "and check that no important facts are lost. "
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
            "First, silently identify hard words or phrases, simplify sentence structure, "
            "and check that no important facts are lost. "
            "Then output ONLY the final simplified text.\n\n"
            f"TEXT:\n{text}"
        )

    raise ValueError(f"Unknown strategy: {strategy}")
