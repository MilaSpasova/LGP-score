from __future__ import annotations
"""Rank prompt and temperature conditions from experiment results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lgp_pipeline.text_metrics import flesch_kincaid_grade


TEXT_COLUMNS = [
    "aoa",
    "concreteness",
    "imageability",
    "fk_grade",
    "mtld",
    "cli",
    "tier2_proxy_token_ratio",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_variant(df: pd.DataFrame) -> pd.Series:
    return (
        df["provider"].astype(str)
        + "::"
        + df["model"].astype(str)
        + "::"
        + df["strategy"].astype(str)
        + "::temp="
        + df["temperature"].astype(str)
    )


def summarize_generation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["json_success"] = df["error"].isna() & df["simplified_text"].fillna("").astype(str).str.strip().ne("")
    df["generated_fk_grade"] = df["simplified_text"].fillna("").map(
        lambda text: flesch_kincaid_grade(text) if text else float("nan")
    )
    df["variant"] = _build_variant(df)

    summary = (
        df.groupby(["provider", "model", "strategy", "temperature", "variant"], dropna=False)
        .agg(
            rows=("json_success", "size"),
            json_success_rate=("json_success", "mean"),
            fk_grade_mean=("generated_fk_grade", "mean"),
            fk_grade_std=("generated_fk_grade", "std"),
        )
        .reset_index()
    )
    return summary


def _normalized_distance(row: pd.Series, baseline: pd.Series, scales: pd.Series) -> float:
    pieces: list[float] = []
    for column in TEXT_COLUMNS:
        if column not in row or pd.isna(row[column]) or pd.isna(baseline[column]):
            continue
        scale = scales[column]
        if pd.isna(scale) or scale == 0:
            scale = 1.0
        pieces.append(abs(float(row[column]) - float(baseline[column])) / float(scale))
    return float(np.mean(pieces)) if pieces else float("nan")


def enrich_with_methodology_outputs(
    summary: pd.DataFrame,
    *,
    text_metrics_path: Path | None,
    pairwise_path: Path | None,
) -> pd.DataFrame:
    enriched = summary.copy()

    if text_metrics_path and text_metrics_path.is_file():
        text_df = pd.read_csv(text_metrics_path)
        llm_df = text_df[text_df["source_type"] == "llm_simplification"].copy()
        if not llm_df.empty:
            llm_summary = llm_df.groupby("variant", dropna=False)[TEXT_COLUMNS].mean().reset_index()
            human_elem = (
                text_df[
                    (text_df["source_type"] == "human_reference")
                    & (text_df["level"] == "Elementary")
                ][TEXT_COLUMNS]
                .mean()
            )
            scales = text_df[TEXT_COLUMNS].std(ddof=0)
            llm_summary["human_benchmark_distance"] = llm_summary.apply(
                lambda row: _normalized_distance(row, human_elem, scales),
                axis=1,
            )
            llm_summary["human_benchmark_score"] = 1.0 / (1.0 + llm_summary["human_benchmark_distance"])
            enriched = enriched.merge(
                llm_summary[["variant", *TEXT_COLUMNS, "human_benchmark_distance", "human_benchmark_score"]],
                on="variant",
                how="left",
            )

    if pairwise_path and pairwise_path.is_file():
        pair_df = pd.read_csv(pairwise_path)
        llm_pairs = pair_df[pair_df["comparison_type"] == "llm_simplification"].copy()
        if not llm_pairs.empty:
            pair_summary = (
                llm_pairs.groupby("variant", dropna=False)
                .agg(
                    semantic_similarity_sbert=("semantic_similarity_sbert", "mean"),
                    delta_cli=("delta_cli", "mean"),
                    delta_aoa=("delta_aoa", "mean"),
                    delta_concreteness=("delta_concreteness", "mean"),
                    delta_imageability=("delta_imageability", "mean"),
                    delta_fk_grade=("delta_fk_grade", "mean"),
                    delta_mtld=("delta_mtld", "mean"),
                    delta_tier2_proxy_token_ratio=("delta_tier2_proxy_token_ratio", "mean"),
                )
                .reset_index()
            )
            enriched = enriched.merge(pair_summary, on="variant", how="left")

    return enriched


def rank_conditions(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["stability_score"] = 1.0 / (1.0 + ranked["fk_grade_std"].fillna(ranked["fk_grade_std"].max()))

    component_columns = ["json_success_rate", "stability_score"]
    if "semantic_similarity_sbert" in ranked.columns:
        component_columns.append("semantic_similarity_sbert")
    if "human_benchmark_score" in ranked.columns:
        component_columns.append("human_benchmark_score")

    ranked["overall_score"] = ranked[component_columns].mean(axis=1, skipna=True)
    sort_columns = ["overall_score", "json_success_rate"]
    ascending = [False, False]
    if "semantic_similarity_sbert" in ranked.columns:
        sort_columns.append("semantic_similarity_sbert")
        ascending.append(False)
    if "human_benchmark_score" in ranked.columns:
        sort_columns.append("human_benchmark_score")
        ascending.append(False)
    sort_columns.append("fk_grade_std")
    ascending.append(True)

    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="stable").reset_index(drop=True)
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank prompt strategy / temperature conditions for the thesis pilot experiment. "
            "Uses generation reliability first, and optionally enriches rankings with "
            "methodology-analysis outputs such as SBERT and human-benchmark closeness."
        )
    )
    parser.add_argument("--experiments", required=True, help="CSV produced by run_onestop_english.py experiments")
    parser.add_argument("--text-metrics", default=None, help="Optional methodology_text_metrics.csv")
    parser.add_argument("--pairwise", default=None, help="Optional methodology_pairwise_comparisons.csv")
    parser.add_argument("--output-dir", default="outputs/pilot_ranking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(output_dir)

    summary = summarize_generation_csv(Path(args.experiments).resolve())
    enriched = enrich_with_methodology_outputs(
        summary,
        text_metrics_path=Path(args.text_metrics).resolve() if args.text_metrics else None,
        pairwise_path=Path(args.pairwise).resolve() if args.pairwise else None,
    )
    ranked = rank_conditions(enriched)

    summary_path = output_dir / "pilot_condition_summary.csv"
    ranked_path = output_dir / "pilot_condition_ranked.csv"
    summary.to_csv(summary_path, index=False)
    ranked.to_csv(ranked_path, index=False)

    print(summary_path)
    print(ranked_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
