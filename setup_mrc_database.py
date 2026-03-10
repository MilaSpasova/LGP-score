from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from datasets import load_dataset
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e


MRC_DATASET_ID = "StephanAkkerman/MRC-psycholinguistic-database"

# Map verbose MRC column names to shorter, analysis-friendly names.
MRC_COLUMN_MAP: dict[str, str] = {
    "word": "word",
    "kf written frequency": "kf_freq",
    "familiarity": "fam",
    "concreteness": "conc",
    "imageability": "img",
    # Dataset variants have used multiple AoA column names over time.
    "age of acquisition": "aoa",
    "age of acquisition rating": "aoa",
    "number of letters": "nlet",
}

# Columns that should be treated as numeric and where 0 means "missing".
MRC_NUMERIC_COLS = ["fam", "conc", "img", "aoa", "kf_freq"]

# Columns that must have at least one non-null value to retain a row.
MRC_REQUIRED_ANY = ["fam", "conc", "img"]


def setup_mrc_database(
    dataset_id: str = MRC_DATASET_ID,
    required_any: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and clean the MRC psycholinguistic database for LGP analysis.

    Steps:
    - Load the Hugging Face dataset and convert the train split to a DataFrame.
    - Lowercase and strip column names for robustness.
    - Rename key columns using MRC_COLUMN_MAP.
    - Clean the 'word' column (strip spaces, remove '&').
    - Convert selected columns to numeric and treat 0 as missing (NaN).
    - Drop rows where all of the required_any columns are missing.

    Parameters
    ----------
    dataset_id:
        Hugging Face dataset identifier.
    required_any:
        Columns of interest where at least one non-null value is required.
        Defaults to ["fam", "conc", "img"].
    """
    print("--- Loading MRC psycholinguistic database ---")
    ds = load_dataset(dataset_id)
    df = ds["train"].to_pandas()

    # Normalise column names.
    df.columns = [str(col).lower().strip() for col in df.columns]

    # Rename selected columns.
    df = df.rename(columns=MRC_COLUMN_MAP)

    # Clean 'word' column.
    if "word" in df.columns:
        df["word"] = df["word"].astype(str).str.replace("&", "", regex=False).str.strip()

    # Convert numeric columns and treat 0 as missing.
    for col in MRC_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    # Keep only rows with at least one psycholinguistic rating.
    required_cols = required_any or MRC_REQUIRED_ANY
    df_filtered = df.dropna(subset=required_cols, how="all").copy()

    print(f"Success! Final columns for analysis: {list(df_filtered.columns[:10])}...")
    print(f"Filtered to {len(df_filtered)} words with psycholinguistic ratings.")

    return df_filtered


if __name__ == "__main__":
    mrc_df = setup_mrc_database()

    # Show a small preview of complete cases (fam, conc, img all present).
    complete_cases = mrc_df.dropna(subset=["fam", "conc", "img"])

    print("\nPreview of Cleaned Data (Complete Cases):")
    print(complete_cases[["word", "fam", "conc", "img"]].head(10))

