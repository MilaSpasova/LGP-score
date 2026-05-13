from __future__ import annotations
"""Run the thesis-local metric pipeline over human and optional LLM texts."""

import argparse
import os
from pathlib import Path

import pandas as pd

from lgp_pipeline.preprocessing import canonical_story_key, normalize_level_name
from lgp_pipeline.psycholinguistics import summarize_text_psycholinguistics
from lgp_pipeline.tier2 import summarize_tier2_proxy
from lgp_pipeline.text_metrics import flesch_kincaid_grade, measure_mtld
from setup_onestop_english import load_onestop_english_aligned, resolve_onestop_corpus_dir


PRIMARY_COLUMNS = ["aoa", "concreteness", "imageability"]


def _parse_semicolon_terms(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    parts = [chunk.strip() for chunk in str(value).split(";")]
    return {part for part in parts if part}


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_corpus_level(level: str) -> str:
    return normalize_level_name("advanced" if level == "Advance" else level)


def _load_human_reference_rows(corpus_root: str | os.PathLike[str]) -> pd.DataFrame:
    corpus = load_onestop_english_aligned(corpus_root).copy()
    corpus["level"] = corpus["level"].map(_normalize_corpus_level)
    corpus["story_key"] = corpus["story_id"].map(canonical_story_key)
    corpus["source_type"] = "human_reference"
    corpus["variant"] = corpus["level"].str.lower().map(lambda level: f"human::{level}")
    return corpus[["story_id", "story_key", "level", "text", "source_type", "variant"]]


def _load_simplifications(path: str | os.PathLike[str]) -> pd.DataFrame:
    simplifications = pd.read_csv(path).copy()
    required = {"story_id", "provider", "model", "source_text", "simplified_text"}
    missing = required - set(simplifications.columns)
    if missing:
        raise ValueError(f"Simplifications CSV is missing required columns: {sorted(missing)}")

    simplifications["story_key"] = simplifications["story_id"].astype(str).map(canonical_story_key)
    simplifications["level"] = "Elementary"
    simplifications["source_type"] = "llm_simplification"
    simplifications["text"] = simplifications["simplified_text"].fillna("")
    strategy_series = simplifications.get("strategy", pd.Series(["zero_shot"] * len(simplifications)))
    temperature_series = simplifications.get("temperature", pd.Series([float("nan")] * len(simplifications)))
    simplifications["variant"] = (
        simplifications["provider"].astype(str)
        + "::"
        + simplifications["model"].astype(str)
        + "::"
        + strategy_series.astype(str)
        + "::temp="
        + temperature_series.astype(str)
    )
    return simplifications


def _compute_local_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        metrics = summarize_text_psycholinguistics(str(row.text))
        metrics.update(summarize_tier2_proxy(str(row.text)))
        metrics["fk_grade"] = flesch_kincaid_grade(str(row.text))
        metrics["mtld"] = measure_mtld(str(row.text))
        records.append(metrics)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _add_cli_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    for column in PRIMARY_COLUMNS:
        mean = float(scored[column].mean(skipna=True))
        std = float(scored[column].std(skipna=True, ddof=0))
        if std == 0 or pd.isna(std):
            scored[f"{column}_z"] = 0.0
        else:
            scored[f"{column}_z"] = (scored[column] - mean) / std

    scored["cli"] = (
        0.25 * scored["concreteness_z"]
        + 0.25 * scored["imageability_z"]
        - 0.50 * scored["aoa_z"]
    )
    return scored


def _build_pairwise_comparisons(text_metrics: pd.DataFrame, *, include_sbert: bool) -> pd.DataFrame:
    compute_similarity = None
    if include_sbert:
        from lgp_pipeline.semantic import compute_sbert_cosine_similarity

        compute_similarity = compute_sbert_cosine_similarity

    advanced = (
        text_metrics[text_metrics["level"] == "Advanced"][
            [
                "story_key",
                "text",
                "aoa",
                "concreteness",
                "imageability",
                "fk_grade",
                "mtld",
                "cli",
                "tier2_proxy_token_ratio",
                "tier2_proxy_matched_types",
                "tier2_proxy_low_frequency_types",
            ]
        ]
        .rename(
            columns={
                "text": "original_text",
                "aoa": "original_aoa",
                "concreteness": "original_concreteness",
                "imageability": "original_imageability",
                "fk_grade": "original_fk_grade",
                "mtld": "original_mtld",
                "cli": "original_cli",
                "tier2_proxy_token_ratio": "original_tier2_proxy_token_ratio",
                "tier2_proxy_matched_types": "original_tier2_proxy_matched_types",
                "tier2_proxy_low_frequency_types": "original_tier2_proxy_low_frequency_types",
            }
        )
    )

    simplified = text_metrics[text_metrics["level"] == "Elementary"].copy()
    joined = simplified.merge(advanced, how="inner", on="story_key")

    comparison_rows: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        semantic_similarity = (
            compute_similarity(str(row.original_text), str(row.text))
            if compute_similarity is not None
            else float("nan")
        )
        comparison_rows.append(
            {
                "story_key": row.story_key,
                "comparison_type": row.source_type,
                "variant": getattr(row, "variant", "human::elementary"),
                "semantic_similarity_sbert": semantic_similarity,
                "delta_cli": row.cli - row.original_cli,
                "delta_aoa": row.aoa - row.original_aoa,
                "delta_concreteness": row.concreteness - row.original_concreteness,
                "delta_imageability": row.imageability - row.original_imageability,
                "delta_fk_grade": row.fk_grade - row.original_fk_grade,
                "delta_mtld": row.mtld - row.original_mtld,
                "delta_tier2_proxy_token_ratio": (
                    row.tier2_proxy_token_ratio - row.original_tier2_proxy_token_ratio
                ),
                # These AVL lists make the Tier 2 proxy more interpretable by
                # showing which cross-domain academic lemmas are retained,
                # added, or removed in the simplified version.
                "tier2_types_retained": "; ".join(
                    sorted(
                        _parse_semicolon_terms(row.tier2_proxy_matched_types)
                        & _parse_semicolon_terms(row.original_tier2_proxy_matched_types)
                    )
                ),
                "tier2_types_removed": "; ".join(
                    sorted(
                        _parse_semicolon_terms(row.original_tier2_proxy_matched_types)
                        - _parse_semicolon_terms(row.tier2_proxy_matched_types)
                    )
                ),
                "tier2_types_added": "; ".join(
                    sorted(
                        _parse_semicolon_terms(row.tier2_proxy_matched_types)
                        - _parse_semicolon_terms(row.original_tier2_proxy_matched_types)
                    )
                ),
                "low_frequency_tier2_types_retained": "; ".join(
                    sorted(
                        _parse_semicolon_terms(row.tier2_proxy_low_frequency_types)
                        & _parse_semicolon_terms(row.original_tier2_proxy_low_frequency_types)
                    )
                ),
            }
        )
    return pd.DataFrame(comparison_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis methodology pipeline using one local metric workflow for "
            "Advanced, Elementary, and optional LLM-generated texts."
        )
    )
    parser.add_argument(
        "--aligned-corpus",
        default=None,
        metavar="DIR",
        help="OneStopEnglishCorpus root. Defaults to OSE_CORPUS_ROOT or ./data/OneStopEnglishCorpus.",
    )
    parser.add_argument(
        "--simplifications-csv",
        default=None,
        help="Optional CSV of LLM simplifications generated from the corpus.",
    )
    parser.add_argument(
        "--text-output",
        default=os.path.join("outputs", "methodology_text_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        default=os.path.join("outputs", "methodology_pairwise_comparisons.csv"),
    )
    parser.add_argument(
        "--skip-sbert",
        action="store_true",
        help="Skip SBERT similarity scoring.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_root = resolve_onestop_corpus_dir(explicit=args.aligned_corpus)

    human_rows = _load_human_reference_rows(corpus_root)

    llm_metrics = pd.DataFrame()
    if args.simplifications_csv:
        simplifications = _load_simplifications(args.simplifications_csv)
        pilot_story_keys = set(simplifications["story_key"].astype(str))
        if pilot_story_keys:
            human_rows = human_rows[human_rows["story_key"].isin(pilot_story_keys)].copy()
        llm_metrics = _compute_local_metrics(
            simplifications[["story_id", "story_key", "level", "text", "source_type", "variant"]]
        )

    human_metrics = _compute_local_metrics(human_rows)

    all_text_metrics = pd.concat([human_metrics, llm_metrics], ignore_index=True, sort=False)
    all_text_metrics = _add_cli_scores(all_text_metrics)

    pairwise = _build_pairwise_comparisons(all_text_metrics, include_sbert=not args.skip_sbert)

    text_output = Path(args.text_output).resolve()
    comparison_output = Path(args.comparison_output).resolve()
    _ensure_parent_dir(text_output)
    _ensure_parent_dir(comparison_output)

    all_text_metrics.to_csv(text_output, index=False)
    pairwise.to_csv(comparison_output, index=False)

    print(f"Wrote text-level metrics to: {text_output}")
    print(f"Wrote pairwise comparisons to: {comparison_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
