from __future__ import annotations
"""Create compact summary tables from experiment and methodology outputs."""

import argparse
from pathlib import Path

import pandas as pd

from lgp_pipeline.text_metrics import flesch_kincaid_grade


TEXT_METRIC_COLUMNS = [
    "aoa",
    "concreteness",
    "imageability",
    "fk_grade",
    "mtld",
    "cli",
    "tier2_proxy_token_ratio",
]
PAIRWISE_COLUMNS = [
    "semantic_similarity_sbert",
    "delta_cli",
    "delta_aoa",
    "delta_concreteness",
    "delta_imageability",
    "delta_fk_grade",
    "delta_mtld",
    "delta_tier2_proxy_token_ratio",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flatten_summary(df: pd.DataFrame, value_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_columns, dropna=False)[value_columns]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part) and str(part) != "nan").rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def summarize_text_metrics(path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(path)
    summary = _flatten_summary(df, TEXT_METRIC_COLUMNS, ["source_type", "level", "variant"])
    out = output_dir / "text_metric_summary.csv"
    summary.to_csv(out, index=False)
    return out


def summarize_pairwise(path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(path)
    summary = _flatten_summary(df, PAIRWISE_COLUMNS, ["comparison_type", "variant"])
    out = output_dir / "pairwise_summary.csv"
    summary.to_csv(out, index=False)
    return out


def summarize_prompt_experiments(path: Path, output_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(path).copy()
    # JSON validity is the operational success criterion for Experiment 1.
    df["json_success"] = df["error"].isna() & df["simplified_text"].fillna("").astype(str).str.strip().ne("")
    df["generated_fk_grade"] = df["simplified_text"].fillna("").map(
        lambda text: flesch_kincaid_grade(text) if text else float("nan")
    )

    grouped = (
        df.groupby(["provider", "model", "strategy", "temperature"], dropna=False)
        .agg(
            rows=("json_success", "size"),
            json_success_rate=("json_success", "mean"),
            fk_grade_mean=("generated_fk_grade", "mean"),
            fk_grade_std=("generated_fk_grade", "std"),
        )
        .reset_index()
    )

    ranked = grouped.sort_values(
        ["json_success_rate", "fk_grade_std", "fk_grade_mean"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    summary_out = output_dir / "prompt_experiment_summary.csv"
    ranked_out = output_dir / "prompt_experiment_ranked.csv"
    grouped.to_csv(summary_out, index=False)
    ranked.to_csv(ranked_out, index=False)
    return summary_out, ranked_out


def build_markdown_report(created_files: list[Path], output_dir: Path) -> Path:
    lines = ["# Result Summary", ""]
    for path in created_files:
        lines.append(f"- `{path.name}`")
    lines.append("")
    report = output_dir / "summary_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready summary tables from methodology and experiment outputs."
    )
    parser.add_argument("--text-metrics", default="outputs/methodology_text_metrics.csv")
    parser.add_argument("--pairwise", default="outputs/methodology_pairwise_comparisons.csv")
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--output-dir", default="outputs/summaries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(output_dir)

    created: list[Path] = []
    text_metrics_path = Path(args.text_metrics).resolve()
    pairwise_path = Path(args.pairwise).resolve()

    if text_metrics_path.is_file():
        created.append(summarize_text_metrics(text_metrics_path, output_dir))
    if pairwise_path.is_file():
        created.append(summarize_pairwise(pairwise_path, output_dir))
    if args.experiments:
        experiments_path = Path(args.experiments).resolve()
        if experiments_path.is_file():
            summary_out, ranked_out = summarize_prompt_experiments(experiments_path, output_dir)
            created.extend([summary_out, ranked_out])

    created.append(build_markdown_report(created, output_dir))

    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
