from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .preprocessing import iter_content_lemmas
from setup_mrc_database import setup_mrc_database


@lru_cache(maxsize=1)
def get_mrc_lookup() -> pd.DataFrame:
    mrc_df = setup_mrc_database()
    available = [column for column in ("aoa", "conc", "img") if column in mrc_df.columns]
    # Duplicate surface forms can appear in the raw lexicon. Grouping here gives
    # one normalized lookup row per lemma for the local scoring pipeline.
    lookup = (
        mrc_df.assign(word=mrc_df["word"].astype(str).str.lower().str.strip())
        .dropna(subset=["word"])
        .groupby("word", as_index=True)[available]
        .mean()
    )
    return lookup


def summarize_text_psycholinguistics(text: str) -> dict[str, float]:
    lemmas = iter_content_lemmas(text, drop_stopwords=True)
    lexical_token_count = len(lemmas)
    if lexical_token_count == 0:
        return {
            "aoa": float("nan"),
            "concreteness": float("nan"),
            "imageability": float("nan"),
            "lexical_token_count": 0.0,
            "matched_lexical_token_count": 0.0,
            "lexical_coverage": float("nan"),
        }

    lookup = get_mrc_lookup()
    matched = lookup.reindex(lemmas)
    matched_rows = matched.dropna(how="all")
    matched_count = float(len(matched_rows))

    # Coverage is useful when interpreting LLM results because some generated
    # vocabulary may fall outside the MRC lexicon and therefore dilute averages.
    return {
        "aoa": float(matched_rows["aoa"].mean()) if "aoa" in matched_rows else float("nan"),
        "concreteness": (
            float(matched_rows["conc"].mean()) if "conc" in matched_rows else float("nan")
        ),
        "imageability": float(matched_rows["img"].mean()) if "img" in matched_rows else float("nan"),
        "lexical_token_count": float(lexical_token_count),
        "matched_lexical_token_count": matched_count,
        "lexical_coverage": matched_count / lexical_token_count,
    }
