from __future__ import annotations

import re


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def count_sentences(text: str) -> int:
    sentence_endings = re.findall(r"[.!?]+", text)
    if sentence_endings:
        return len(sentence_endings)
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    return max(1, len(non_empty_lines))


def count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0

    syllable_count = len(VOWEL_GROUP_RE.findall(cleaned))
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and syllable_count > 1:
        syllable_count -= 1
    if cleaned.endswith("es") and len(cleaned) > 2 and syllable_count > 1:
        syllable_count -= 1
    return max(1, syllable_count)


def flesch_kincaid_grade(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return float("nan")
    sentence_count = max(1, count_sentences(text))
    syllable_count = sum(count_syllables(word) for word in words)
    return 0.39 * (len(words) / sentence_count) + 11.8 * (syllable_count / len(words)) - 15.59


def mtld_segment_count(tokens: list[str], *, threshold: float = 0.72) -> float:
    if not tokens:
        return float("nan")

    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for token in tokens:
        token_count += 1
        types.add(token)
        if len(types) / token_count <= threshold:
            factors += 1.0
            types = set()
            token_count = 0

    if token_count > 0:
        ttr = len(types) / token_count
        factors += 0.0 if ttr == 1.0 else (1.0 - ttr) / (1.0 - threshold)
    return float("nan") if factors == 0 else len(tokens) / factors


def measure_mtld(text: str, *, threshold: float = 0.72) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return float("nan")
    forward = mtld_segment_count(tokens, threshold=threshold)
    backward = mtld_segment_count(list(reversed(tokens)), threshold=threshold)
    return (forward + backward) / 2.0
