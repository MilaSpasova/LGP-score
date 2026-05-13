from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from collections import Counter

import pandas as pd

from .preprocessing import count_content_lemmas, iter_content_lemmas


DEFAULT_AVL_PATH = Path("data") / "lexicons" / "acadCore.xlsx"


@lru_cache(maxsize=1)
def load_avl_core_lemmas(path: str | Path = DEFAULT_AVL_PATH) -> set[str]:
    avl_path = Path(path).resolve()
    # Gardner and Davies' AVL "core" list is used here as a practical proxy
    # for Beck et al.'s Tier 2 vocabulary: cross-domain, high-utility, and
    # less domain-specific than technical terminology.
    raw = pd.read_excel(avl_path, sheet_name="list")
    if "word" not in raw.columns:
        raise ValueError(f"{avl_path.name}: missing expected 'word' column in AVL sheet")
    return {
        str(word).strip().lower()
        for word in raw["word"].dropna()
        if str(word).strip()
    }


def _stringify_lemma_list(items: list[str]) -> str:
    return "; ".join(items)


def summarize_tier2_proxy(text: str, *, avl_path: str | Path = DEFAULT_AVL_PATH) -> dict[str, object]:
    lemmas = iter_content_lemmas(text, drop_stopwords=True)
    lexical_token_count = len(lemmas)
    if lexical_token_count == 0:
        return {
            "tier2_proxy_token_count": 0.0,
            "tier2_proxy_type_count": 0.0,
            "tier2_proxy_token_ratio": float("nan"),
            "tier2_proxy_type_ratio": float("nan"),
            "tier2_proxy_low_frequency_type_count": 0.0,
            "tier2_proxy_low_frequency_types": "",
            "tier2_proxy_matched_types": "",
        }

    avl_core = load_avl_core_lemmas(avl_path)
    lemma_counts: Counter[str] = count_content_lemmas(text, drop_stopwords=True)
    matched_tokens = [lemma for lemma in lemmas if lemma in avl_core]
    unique_lemmas = set(lemmas)
    matched_types = unique_lemmas & avl_core
    low_frequency_types = sorted([lemma for lemma in matched_types if lemma_counts[lemma] <= 1])
    matched_types_sorted = sorted(matched_types)

    return {
        "tier2_proxy_token_count": float(len(matched_tokens)),
        "tier2_proxy_type_count": float(len(matched_types)),
        "tier2_proxy_token_ratio": len(matched_tokens) / lexical_token_count,
        "tier2_proxy_type_ratio": len(matched_types) / len(unique_lemmas),
        "tier2_proxy_low_frequency_type_count": float(len(low_frequency_types)),
        "tier2_proxy_low_frequency_types": _stringify_lemma_list(low_frequency_types),
        "tier2_proxy_matched_types": _stringify_lemma_list(matched_types_sorted),
    }
