from __future__ import annotations
"""Run paired non-parametric statistical tests over the final thesis conditions."""

from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
STATS_DIR = OUTPUTS / "statistical_tests"

HUMAN_VARIANT = "human::elementary"
FEW_SHOT_VARIANT = "openai::openai/gpt-5.2::few_shot::temp=0.5"
ZERO_SHOT_VARIANT = "openai::openai/gpt-5.2::zero_shot::temp=0.0"


TEXT_METRICS = {
    "AoA": ("aoa_h", "aoa_f", "aoa_z"),
    "FKGL": ("fk_h", "fk_f", "fk_z"),
    "CLI": ("cli_h", "cli_f", "cli_z"),
    "MTLD": ("mtld_h", "mtld_f", "mtld_z"),
    "Tier2Proxy": ("tier2_h", "tier2_f", "tier2_z"),
}

PAIRWISE_METRICS = {
    "SBERT": ("sbert_h", "sbert_f", "sbert_z"),
    "DeltaCLI": ("dcli_h", "dcli_f", "dcli_z"),
    "DeltaAoA": ("daoa_h", "daoa_f", "daoa_z"),
    "DeltaFKGL": ("dfk_h", "dfk_f", "dfk_z"),
    "DeltaMTLD": ("dmtld_h", "dmtld_f", "dmtld_z"),
    "DeltaTier2Proxy": ("dtier2_h", "dtier2_f", "dtier2_z"),
}


def holm_adjust(ps: list[float]) -> list[float]:
    """Apply Holm's step-down correction to a list of p-values."""
    ranked = sorted(enumerate(ps), key=lambda item: item[1])
    adjusted = [0.0] * len(ps)
    running_max = 0.0
    m = len(ps)
    for rank, (idx, p) in enumerate(ranked, start=1):
        value = min(1.0, (m - rank + 1) * p)
        value = max(value, running_max)
        adjusted[idx] = value
        running_max = value
    return adjusted


def rank_biserial(x: pd.Series, y: pd.Series) -> float:
    """Compute the matched-pairs rank-biserial correlation for Wilcoxon comparisons."""
    differences = (x - y)
    differences = differences[differences != 0]
    if differences.empty:
        return 0.0
    ranks = rankdata(differences.abs())
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def _load_selected_rows(path: Path, *, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame["variant"] == variant].copy()


def load_text_level() -> pd.DataFrame:
    """Load one aligned text-level row per story for the three final conditions."""
    human_s = _load_selected_rows(OUTPUTS / "human_baseline_text_metrics.csv", variant=HUMAN_VARIANT)
    few_s = _load_selected_rows(OUTPUTS / "full_batch_fewshot05_text_metrics.csv", variant=FEW_SHOT_VARIANT)
    zero_s = _load_selected_rows(OUTPUTS / "full_batch_zeroshot00_text_metrics.csv", variant=ZERO_SHOT_VARIANT)

    merged = human_s[
        ["story_key", "aoa", "cli", "tier2_proxy_token_ratio", "fk_grade", "mtld"]
    ].rename(
        columns={
            "aoa": "aoa_h",
            "cli": "cli_h",
            "tier2_proxy_token_ratio": "tier2_h",
            "fk_grade": "fk_h",
            "mtld": "mtld_h",
        }
    )
    merged = merged.merge(
        few_s[["story_key", "aoa", "cli", "tier2_proxy_token_ratio", "fk_grade", "mtld"]].rename(
            columns={
                "aoa": "aoa_f",
                "cli": "cli_f",
                "tier2_proxy_token_ratio": "tier2_f",
                "fk_grade": "fk_f",
                "mtld": "mtld_f",
            }
        ),
        on="story_key",
    )
    merged = merged.merge(
        zero_s[["story_key", "aoa", "cli", "tier2_proxy_token_ratio", "fk_grade", "mtld"]].rename(
            columns={
                "aoa": "aoa_z",
                "cli": "cli_z",
                "tier2_proxy_token_ratio": "tier2_z",
                "fk_grade": "fk_z",
                "mtld": "mtld_z",
            }
        ),
        on="story_key",
    )
    return merged


def load_pairwise() -> pd.DataFrame:
    """Load one aligned pairwise-comparison row per story for the three final conditions."""
    human = pd.read_csv(OUTPUTS / "human_baseline_pairwise_comparisons.csv")
    few = pd.read_csv(OUTPUTS / "full_batch_fewshot05_pairwise.csv")
    zero = pd.read_csv(OUTPUTS / "full_batch_zeroshot00_pairwise.csv")

    human = human[human["comparison_type"] == "human_reference"].copy()
    few = few[few["comparison_type"] == "llm_simplification"].copy()
    zero = zero[zero["comparison_type"] == "llm_simplification"].copy()

    merged = human[
        [
            "story_key",
            "semantic_similarity_sbert",
            "delta_cli",
            "delta_aoa",
            "delta_fk_grade",
            "delta_mtld",
            "delta_tier2_proxy_token_ratio",
        ]
    ].rename(
        columns={
            "semantic_similarity_sbert": "sbert_h",
            "delta_cli": "dcli_h",
            "delta_aoa": "daoa_h",
            "delta_fk_grade": "dfk_h",
            "delta_mtld": "dmtld_h",
            "delta_tier2_proxy_token_ratio": "dtier2_h",
        }
    )
    merged = merged.merge(
        few[
            [
                "story_key",
                "semantic_similarity_sbert",
                "delta_cli",
                "delta_aoa",
                "delta_fk_grade",
                "delta_mtld",
                "delta_tier2_proxy_token_ratio",
            ]
        ].rename(
            columns={
                "semantic_similarity_sbert": "sbert_f",
                "delta_cli": "dcli_f",
                "delta_aoa": "daoa_f",
                "delta_fk_grade": "dfk_f",
                "delta_mtld": "dmtld_f",
                "delta_tier2_proxy_token_ratio": "dtier2_f",
            }
        ),
        on="story_key",
    )
    merged = merged.merge(
        zero[
            [
                "story_key",
                "semantic_similarity_sbert",
                "delta_cli",
                "delta_aoa",
                "delta_fk_grade",
                "delta_mtld",
                "delta_tier2_proxy_token_ratio",
            ]
        ].rename(
            columns={
                "semantic_similarity_sbert": "sbert_z",
                "delta_cli": "dcli_z",
                "delta_aoa": "daoa_z",
                "delta_fk_grade": "dfk_z",
                "delta_mtld": "dmtld_z",
                "delta_tier2_proxy_token_ratio": "dtier2_z",
            }
        ),
        on="story_key",
    )
    return merged


def analyze_block(
    df: pd.DataFrame,
    metric_map: dict[str, tuple[str, str, str]],
    block_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run omnibus Friedman tests and Holm-corrected Wilcoxon post-hoc tests."""
    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for metric, cols in metric_map.items():
        human_scores, few_scores, zero_scores = [df[col] for col in cols]
        friedman = friedmanchisquare(human_scores, few_scores, zero_scores)
        kendalls_w = friedman.statistic / (len(df) * 2)

        omnibus_rows.append(
            {
                "block": block_name,
                "metric": metric,
                "n": len(df),
                "friedman_chi2": friedman.statistic,
                "friedman_p": friedman.pvalue,
                "kendalls_w": kendalls_w,
            }
        )

        comparisons = [
            ("Human Elementary", "LLM Few-shot 0.5", human_scores, few_scores),
            ("Human Elementary", "LLM Zero-shot 0.0", human_scores, zero_scores),
            ("LLM Few-shot 0.5", "LLM Zero-shot 0.0", few_scores, zero_scores),
        ]
        raw_ps: list[float] = []
        temporary: list[dict[str, object]] = []
        for left, right, x, y in comparisons:
            result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
            temporary.append(
                {
                    "block": block_name,
                    "metric": metric,
                    "left_condition": left,
                    "right_condition": right,
                    "n": len(df),
                    "wilcoxon_w": result.statistic,
                    "raw_p": result.pvalue,
                    "rank_biserial": rank_biserial(pd.Series(x), pd.Series(y)),
                }
            )
            raw_ps.append(result.pvalue)

        adjusted = holm_adjust(raw_ps)
        for row, holm_p in zip(temporary, adjusted):
            row["holm_p"] = holm_p
            pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def main() -> int:
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    text = load_text_level()
    pairwise = load_pairwise()

    text_omnibus, text_pairwise = analyze_block(text, TEXT_METRICS, "text_level")
    pair_omnibus, pair_pairwise = analyze_block(pairwise, PAIRWISE_METRICS, "pairwise")

    text_omnibus.to_csv(STATS_DIR / "text_level_friedman_tests.csv", index=False)
    text_pairwise.to_csv(STATS_DIR / "text_level_wilcoxon_posthoc.csv", index=False)
    pair_omnibus.to_csv(STATS_DIR / "pairwise_friedman_tests.csv", index=False)
    pair_pairwise.to_csv(STATS_DIR / "pairwise_wilcoxon_posthoc.csv", index=False)

    print("Saved statistical test outputs to", STATS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
