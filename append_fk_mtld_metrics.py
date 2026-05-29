from __future__ import annotations
"""legacy"""
import argparse
import csv
from pathlib import Path

import pandas as pd

from lgp_pipeline.text_metrics import flesch_kincaid_grade, measure_mtld

DEFAULT_FK_FEATURE_ID = 200001
DEFAULT_MTLD_FEATURE_ID = 200002
FK_FEATURE_NAME = "Readability: Flesch-Kincaid Grade Level"
MTLD_FEATURE_NAME = "Lexical Diversity: Measure of Textual Lexical Diversity (MTLD)"


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
    if "\t" in sample:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode file: {path}")


def build_filename_candidates(title: str) -> list[str]:
    candidates = [title]
    if " " in title:
        candidates.append(title.split(" ", maxsplit=1)[1])
    candidates.append(title.replace(" ", "_"))
    return list(dict.fromkeys(candidates))


def resolve_text_path(title: str, texts_dir: Path) -> Path:
    for candidate in build_filename_candidates(title):
        p = texts_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f"No text file found in {texts_dir} for title: {title}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append Flesch-Kincaid and MTLD scores as new feature rows "
            "to a results CSV/TSV (e.g. exportResultsServlet)."
        )
    )
    parser.add_argument("--results-path", required=True, help="Path to results CSV/TSV file.")
    parser.add_argument("--texts-dir", required=True, help="Directory containing text files.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path. If omitted, overwrites --results-path in place.",
    )
    parser.add_argument("--id-col", default="Text_id", help="Text id column in results file.")
    parser.add_argument("--title-col", default="Text_Title", help="Text title column in results file.")
    parser.add_argument(
        "--title-filter",
        default=None,
        help="Optional substring filter on text title (e.g. '-ele.txt').",
    )
    parser.add_argument(
        "--titles",
        default=None,
        help="Optional comma-separated exact titles to process.",
    )
    parser.add_argument("--feature-id-fk", type=int, default=DEFAULT_FK_FEATURE_ID)
    parser.add_argument("--feature-id-mtld", type=int, default=DEFAULT_MTLD_FEATURE_ID)
    parser.set_defaults(replace_existing=True)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        dest="replace_existing",
        help="Remove existing FK/MTLD rows before adding fresh values (default behavior).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_false",
        dest="replace_existing",
        help="Keep existing FK/MTLD rows and append new ones (may create duplicates).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_path = Path(args.results_path).resolve()
    texts_dir = Path(args.texts_dir).resolve()
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {results_path}")
    if not texts_dir.is_dir():
        raise FileNotFoundError(f"Text directory does not exist: {texts_dir}")

    delimiter = detect_delimiter(results_path)
    df = pd.read_csv(results_path, sep=delimiter)

    for col in (args.id_col, args.title_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {results_path}")

    pairs = (
        df[[args.id_col, args.title_col]]
        .drop_duplicates()
        .rename(columns={args.id_col: "text_id", args.title_col: "text_title"})
    )

    if args.title_filter:
        pairs = pairs[pairs["text_title"].astype(str).str.contains(args.title_filter, regex=False)]

    if args.titles:
        wanted = {x.strip() for x in args.titles.split(",") if x.strip()}
        pairs = pairs[pairs["text_title"].astype(str).isin(wanted)]

    if pairs.empty:
        raise ValueError("No matching texts after applying filters.")

    # Keep track of original row order so we can reinsert new rows within each text block.
    df = df.copy()
    df["_row_order"] = range(len(df))

    new_rows: list[dict[str, object]] = []
    missing_titles: list[str] = []

    for row in pairs.itertuples(index=False):
        text_id = row.text_id
        text_title = str(row.text_title)
        try:
            text_path = resolve_text_path(text_title, texts_dir)
            text = read_text_file(text_path)
        except FileNotFoundError:
            missing_titles.append(text_title)
            continue

        fk_score = flesch_kincaid_grade(text)
        mtld_score = measure_mtld(text)

        base = {
            args.id_col: text_id,
            "Tags": "",
            args.title_col: text_title,
        }
        new_rows.append(
            {
                **base,
                "Feature_id": args.feature_id_fk,
                "Feature_Name": FK_FEATURE_NAME,
                "Value": fk_score,
            }
        )
        new_rows.append(
            {
                **base,
                "Feature_id": args.feature_id_mtld,
                "Feature_Name": MTLD_FEATURE_NAME,
                "Value": mtld_score,
            }
        )

    if not new_rows:
        raise ValueError("No score rows generated. Check titles/text directory mapping.")

    if args.replace_existing and "Feature_Name" in df.columns:
        df = df[~df["Feature_Name"].isin({FK_FEATURE_NAME, MTLD_FEATURE_NAME})].copy()

    new_df = pd.DataFrame(new_rows)

    # Insert FK/MTLD directly after each text's existing block instead of appending globally.
    if not new_df.empty:
        if "_row_order" in df.columns:
            max_row_order = (
                df.groupby([args.id_col, args.title_col], dropna=False)["_row_order"]
                .max()
                .to_dict()
            )
        else:
            max_row_order = {}

        per_text_insert_count: dict[tuple[object, object], int] = {}
        new_order_values: list[int] = []
        for row in new_df.itertuples(index=False):
            key = (getattr(row, args.id_col), getattr(row, args.title_col))
            base = int(max_row_order.get(key, len(df)))
            offset = per_text_insert_count.get(key, 0) + 1
            per_text_insert_count[key] = offset
            new_order_values.append(base + offset)

        new_df = new_df.copy()
        new_df["_row_order"] = new_order_values

    out_df = pd.concat([df, new_df], ignore_index=True)
    if "_row_order" in out_df.columns:
        out_df = out_df.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"])

    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else results_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep=delimiter, index=False)

    print(f"Processed texts: {len(new_rows) // 2}")
    print(f"Added rows: {len(new_rows)}")
    print(f"Output written to: {output_path}")
    if missing_titles:
        print(f"Missing text files for {len(missing_titles)} titles.")
        print("First missing titles:")
        for title in missing_titles[:10]:
            print(f"- {title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
