from __future__ import annotations

import pandas as pd
try:
    from datasets import load_dataset
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e


LEVEL_ID_TO_TEXT: dict[int, str] = {0: "Elementary", 1: "Intermediate", 2: "Advance"}
LEVEL_TEXT_TO_ID: dict[str, int] = {v: k for k, v in LEVEL_ID_TO_TEXT.items()}


def load_onestop_english(*, include_splits: tuple[str, ...] = ("train", "test")) -> pd.DataFrame:
    """
    Load the OneStopEnglish corpus from Hugging Face as a single DataFrame.

    Dataset: SetFit/onestop_english
    Schema: text (str), label (int: 0/1/2), label_text (str)
    """
    ds = load_dataset("SetFit/onestop_english")

    frames: list[pd.DataFrame] = []
    for split in include_splits:
        if split not in ds:
            raise KeyError(f"Split '{split}' not found. Available: {list(ds.keys())}")
        df = ds[split].to_pandas()
        df["split"] = split
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    expected = {"text", "label", "label_text", "split"}
    missing = expected.difference(out.columns)
    if missing:
        raise ValueError(f"Unexpected schema. Missing columns: {sorted(missing)}")

    out = out.rename(columns={"label": "level_id", "label_text": "level"})

    out["level_id"] = pd.to_numeric(out["level_id"], errors="coerce")
    out["text"] = out["text"].astype(str)
    out["level"] = out["level"].astype(str)

    out = out.dropna(subset=["level_id", "text"]).copy()
    out["level_id"] = out["level_id"].astype(int)

    known_ids = set(LEVEL_ID_TO_TEXT.keys())
    unknown_ids = sorted(set(out["level_id"].unique()) - known_ids)
    if unknown_ids:
        raise ValueError(f"Unknown level_id values found: {unknown_ids}")

    out["level"] = out["level"].replace(LEVEL_ID_TO_TEXT)

    out = out[["text", "level", "level_id", "split"]]
    out = out.reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_onestop_english()
    print(df.head(5))
    print(df["level"].value_counts())
