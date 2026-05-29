from __future__ import annotations

import re
from functools import lru_cache
from collections import Counter

try:
    import spacy
    from spacy.lang.en import English
except ModuleNotFoundError:  
    spacy = None
    English = None


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LEVEL_SUFFIX_RE = re.compile(r"(?:^|[\s_-])(adv|advanced|ele|elementary|int|intermediate)$")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Small built-in fallback so the methodology pipeline still runs when spaCy
# is unavailable in a constrained environment.
_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

LEVEL_ALIASES: dict[str, str] = {
    "adv": "Advanced",
    "advanced": "Advanced",
    "advance": "Advanced",
    "ele": "Elementary",
    "elementary": "Elementary",
    "int": "Intermediate",
    "intermediate": "Intermediate",
}


def canonical_story_key(text: str) -> str:
    stem = str(text).rsplit(".", maxsplit=1)[0]
    cleaned = _NON_ALNUM_RE.sub(" ", stem.lower()).strip()
    cleaned = _LEVEL_SUFFIX_RE.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def normalize_level_name(label: str) -> str:
    key = str(label).strip().lower()
    if key not in LEVEL_ALIASES:
        raise ValueError(f"Unknown level label: {label}")
    return LEVEL_ALIASES[key]


def infer_level_from_title(title: str) -> str:
    stem = str(title).rsplit(".", maxsplit=1)[0]
    match = re.search(r"(adv|advanced|ele|elementary|int|intermediate)$", stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer reading level from title: {title}")
    return normalize_level_name(match.group(1))


@lru_cache(maxsize=1)
def get_spacy_nlp():
    if spacy is None or English is None:
        return None
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        nlp = English()
        nlp.add_pipe("sentencizer")
    return nlp


def iter_content_lemmas(text: str, *, drop_stopwords: bool = True) -> list[str]:
    nlp = get_spacy_nlp()
    if nlp is None:
        tokens = [token.lower() for token in _WORD_RE.findall(text)]
        if drop_stopwords:
            tokens = [token for token in tokens if token not in _FALLBACK_STOPWORDS]
        return tokens

    doc = nlp(text)

    lemmas: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if drop_stopwords and token.is_stop:
            continue
        lemma = token.lemma_.strip().lower() if token.lemma_ else ""
        if not lemma or lemma == "-pron-":
            lemma = token.text.strip().lower()
        if not lemma.isalpha():
            continue
        lemmas.append(lemma)
    return lemmas


def count_content_lemmas(text: str, *, drop_stopwords: bool = True) -> Counter[str]:
    return Counter(iter_content_lemmas(text, drop_stopwords=drop_stopwords))
