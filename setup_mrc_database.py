import pandas as pd
import numpy as np
from datasets import load_dataset


def setup_mrc_database():
    print("--- Cleaning MRC for Thesis Analysis ---")
    dataset = load_dataset("StephanAkkerman/MRC-psycholinguistic-database")
    df = dataset["train"].to_pandas()

    # 1. Force lowercase
    df.columns = [str(col).lower().strip() for col in df.columns]

    # 2. Updated Mapping based on your last output
    # We map the long names to short, consistent keys
    name_map = {
        "word": "word",
        "kf written frequency": "kf_freq",
        "familiarity": "fam",
        "concreteness": "conc",
        "imageability": "img",  # Matches the 'imageability' col found
        "age of acquisition": "aoa",
        "number of letters": "nlet",
    }

    df = df.rename(columns=name_map)

    # 3. Clean 'word' column
    if "word" in df.columns:
        df["word"] = (
            df["word"].astype(str).str.replace("&", "", regex=False).str.strip()
        )

    # 4. Numeric conversion & handle 0s
    # We use 'img' here because we renamed 'imageability' to 'img' above
    check_cols = ["fam", "conc", "img", "aoa", "kf_freq"]

    for col in check_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    # 5. FILTER: Keep only words with usable data
    df_filtered = df.dropna(subset=["fam", "conc", "img"], how="all").copy()

    print(f"Success! Final columns for analysis: {list(df_filtered.columns[:10])}...")
    print(f"Filtered to {len(df_filtered)} words with psycholinguistic ratings.")

    return df_filtered


if __name__ == "__main__":
    try:
        mrc_df = setup_mrc_database()

        # Display the best-rated words (those that have all three major scores)
        complete_cases = mrc_df.dropna(subset=["fam", "conc", "img"])

        print("\nPreview of Cleaned Data (Complete Cases):")
        print(complete_cases[["word", "fam", "conc", "img"]].head(10))

    except Exception as e:
        print(f"An error occurred: {e}")
