from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from llm_api.gemini_client import simplify_with_gemini
from llm_api.openai_client import simplify_with_openai
from llm_api.prompts import PromptStrategy
from setup_onestop_english import load_onestop_english_aligned, resolve_onestop_corpus_dir


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _slugify_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text).strip())
    safe = "_".join(chunk for chunk in safe.split("_") if chunk)
    return safe or "text"


def _variant_folder_name(*, provider: str, model: str, strategy: str, temperature: float) -> str:
    model_slug = _slugify_filename(model.replace("/", "__"))
    return f"{provider}__{model_slug}__{strategy}__temp_{temperature}"


def _story_output_name(row: pd.Series, fallback_index: int) -> str:
    story_id = row.get("story_id", None)
    if story_id is not None and str(story_id).strip():
        return _slugify_filename(str(story_id))
    return f"text_{fallback_index:03d}"


def _write_generated_text(
    *,
    export_root: str | None,
    provider: str,
    model: str,
    strategy: str,
    temperature: float,
    row: pd.Series,
    fallback_index: int,
    simplified_text: Optional[str],
) -> None:
    if not export_root or not simplified_text or not str(simplified_text).strip():
        return
    folder = (
        Path(export_root).resolve()
        / _variant_folder_name(
            provider=provider,
            model=model,
            strategy=strategy,
            temperature=temperature,
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    name = _story_output_name(row, fallback_index)
    out_path = folder / f"{name}.txt"
    out_path.write_text(str(simplified_text).strip() + "\n", encoding="utf-8")


def _call_with_retries(
    fn: Callable[..., str], *, max_retries: int, base_sleep_s: float, **kwargs
) -> Tuple[Optional[str], Optional[str]]:
    for attempt in range(max_retries + 1):
        try:
            return fn(**kwargs), None
        except Exception as e:  # noqa: BLE001
            if attempt >= max_retries:
                return None, repr(e)
            time.sleep(base_sleep_s * (2**attempt))
    return None, "Unknown error"


def _parse_temperatures(raw: str) -> list[float]:
    vals: list[float] = []
    for chunk in raw.split(","):
        s = chunk.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("No temperatures parsed from --temperatures.")
    return vals


def _normalize_level_name(level: str) -> str:
    normalized = str(level).strip().lower()
    mapping = {
        "advanced": "Advance",
        "advance": "Advance",
        "adv": "Advance",
        "elementary": "Elementary",
        "ele": "Elementary",
        "intermediate": "Intermediate",
        "int": "Intermediate",
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown reading level: {level}")
    return mapping[normalized]


def _filter_source_levels(df: pd.DataFrame, raw_levels: str) -> pd.DataFrame:
    if raw_levels.strip().lower() == "all":
        return df
    # The corpus loader exposes three aligned levels, but thesis experiments
    # usually start from Advanced and generate a target Elementary version.
    selected = {_normalize_level_name(chunk.strip()) for chunk in raw_levels.split(",") if chunk.strip()}
    return df[df["level"].isin(selected)].copy()


def _sample_rows(df: pd.DataFrame, *, limit: int, sample_seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df.copy()
    rng = random.Random(sample_seed)
    indices = list(df.index)
    rng.shuffle(indices)
    selected = indices[:limit]
    return df.loc[selected].sort_index().copy()


def cmd_batch(args: argparse.Namespace) -> int:
    try:
        corpus_root = resolve_onestop_corpus_dir(explicit=args.aligned_corpus)
    except FileNotFoundError as e:
        print(f"batch: {e}", file=sys.stderr)
        return 1
    df = load_onestop_english_aligned(corpus_root)
    df = _filter_source_levels(df, args.source_levels)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows: list[dict[str, object]] = []
    for i, r in df.reset_index(drop=False).iterrows():
        source_id = int(r["index"])
        source_text = str(r["text"])
        level = str(r["level"])
        # Keeping source and target levels separate makes the experimental
        # setup explicit in the output CSV instead of baking assumptions into it.
        target_level = _normalize_level_name(args.target_level) if args.target_level else level
        level_id = int(r["level_id"])
        split = str(r["split"])

        common = {
            "source_id": source_id,
            "level": level,
            "level_id": level_id,
            "split": split,
            "source_text": source_text,
            "target_level": target_level,
            "strategy": args.strategy,
            "created_at": _utc_now_iso(),
        }
        if "story_id" in df.columns:
            common["story_id"] = str(r["story_id"])

        if args.provider in ("openai", "both"):
            simplified, err = _call_with_retries(
                simplify_with_openai,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep_s,
                text=source_text,
                target_level=target_level,
                strategy=args.strategy,  # type: ignore[arg-type]
                model=args.openai_model,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            rows.append(
                {
                    **common,
                    "provider": "openai",
                    "model": args.openai_model,
                    "temperature": args.temperature,
                    "simplified_text": simplified,
                    "error": err,
                }
            )
            _write_generated_text(
                export_root=args.export_text_dir,
                provider="openai",
                model=args.openai_model,
                strategy=args.strategy,
                temperature=args.temperature,
                row=r,
                fallback_index=i,
                simplified_text=simplified,
            )

        if args.provider in ("gemini", "both"):
            simplified, err = _call_with_retries(
                simplify_with_gemini,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep_s,
                text=source_text,
                target_level=target_level,
                strategy=args.strategy,  # type: ignore[arg-type]
                model=args.gemini_model,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            rows.append(
                {
                    **common,
                    "provider": "gemini",
                    "model": args.gemini_model,
                    "temperature": args.temperature,
                    "simplified_text": simplified,
                    "error": err,
                }
            )
            _write_generated_text(
                export_root=args.export_text_dir,
                provider="gemini",
                model=args.gemini_model,
                strategy=args.strategy,
                temperature=args.temperature,
                row=r,
                fallback_index=i,
                simplified_text=simplified,
            )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} source texts...")

    out_df = pd.DataFrame(rows)
    _ensure_parent_dir(args.output)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    strategy_values = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strategy_values:
        raise ValueError("No strategies parsed from --strategies.")
    strategies = [s for s in strategy_values]
    valid_strategies = {"zero_shot", "few_shot", "chain_of_thought"}
    unknown = sorted(set(strategies) - valid_strategies)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")

    temperatures = _parse_temperatures(args.temperatures)

    try:
        corpus_root = resolve_onestop_corpus_dir(explicit=args.aligned_corpus)
    except FileNotFoundError as e:
        print(f"experiments: {e}", file=sys.stderr)
        return 1
    df = load_onestop_english_aligned(corpus_root)
    advanced = _filter_source_levels(df, args.source_levels)
    if args.limit > 0:
        advanced = _sample_rows(advanced, limit=args.limit, sample_seed=args.sample_seed)
    advanced = advanced.reset_index(drop=True)

    providers = ["openai", "gemini"] if args.provider == "both" else [args.provider]
    rows: list[dict[str, object]] = []
    total = len(advanced) * len(strategies) * len(temperatures) * len(providers)
    done = 0
    for source_id, r in advanced.iterrows():
        source_text = str(r["text"])
        story_id = str(r["story_id"]) if "story_id" in advanced.columns else None
        for strategy in strategies:
            for temperature in temperatures:
                strategy_typed: PromptStrategy = strategy  # type: ignore[assignment]
                # This grid is the reproducible prompt/temperature search used
                # to decide which generation setup to carry into later analysis.
                common = {
                    "source_id": int(source_id),
                    "story_id": story_id,
                    "split": "aligned",
                    "level": str(r["level"]),
                    "level_id": int(r["level_id"]),
                    "target_level": _normalize_level_name(args.target_level),
                    "source_text": source_text,
                    "strategy": strategy,
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "created_at": _utc_now_iso(),
                }

                if "openai" in providers:
                    simplified, err = _call_with_retries(
                        simplify_with_openai,
                        max_retries=args.max_retries,
                        base_sleep_s=args.base_sleep_s,
                        text=source_text,
                        target_level=_normalize_level_name(args.target_level),
                        strategy=strategy_typed,
                        model=args.openai_model,
                        temperature=temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                    )
                    rows.append(
                        {
                            **common,
                            "provider": "openai",
                            "model": args.openai_model,
                            "simplified_text": simplified,
                            "error": err,
                        }
                    )
                    _write_generated_text(
                        export_root=args.export_text_dir,
                        provider="openai",
                        model=args.openai_model,
                        strategy=strategy,
                        temperature=temperature,
                        row=r,
                        fallback_index=source_id,
                        simplified_text=simplified,
                    )
                    done += 1

                if "gemini" in providers:
                    simplified, err = _call_with_retries(
                        simplify_with_gemini,
                        max_retries=args.max_retries,
                        base_sleep_s=args.base_sleep_s,
                        text=source_text,
                        target_level=_normalize_level_name(args.target_level),
                        strategy=strategy_typed,
                        model=args.gemini_model,
                        temperature=temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                    )
                    rows.append(
                        {
                            **common,
                            "provider": "gemini",
                            "model": args.gemini_model,
                            "simplified_text": simplified,
                            "error": err,
                        }
                    )
                    _write_generated_text(
                        export_root=args.export_text_dir,
                        provider="gemini",
                        model=args.gemini_model,
                        strategy=strategy,
                        temperature=temperature,
                        row=r,
                        fallback_index=source_id,
                        simplified_text=simplified,
                    )
                    done += 1

                if done % 20 == 0:
                    print(f"Progress: {done}/{total}")

    out_df = pd.DataFrame(rows)
    _ensure_parent_dir(args.output)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    text = "The committee convened to deliberate on the proposal."
    level = "Elementary"

    openai_out = simplify_with_openai(
        text=text,
        target_level=level,
        strategy="zero_shot",
        model=args.openai_model,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print("OpenAI track OK")
    print(openai_out[:200].replace("\n", " "))

    gemini_out = simplify_with_gemini(
        text=text,
        target_level=level,
        strategy="zero_shot",
        model=args.gemini_model,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print("Gemini track OK")
    print(gemini_out[:200].replace("\n", " "))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OneStopEnglish text simplification via OpenRouter (OpenAI + Gemini model IDs)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser(
        "batch",
        help="Run simplification on official aligned OneStopEnglishCorpus rows (on disk).",
    )
    p_batch.add_argument("--provider", choices=["openai", "gemini", "both"], default="both")
    p_batch.add_argument(
        "--source-levels",
        default="all",
        help="Comma-separated source levels to simplify, or 'all'. Example: Advanced",
    )
    p_batch.add_argument(
        "--target-level",
        default=None,
        help="Target reading level. Defaults to the source level when omitted.",
    )
    p_batch.add_argument(
        "--strategy",
        choices=["zero_shot", "few_shot", "chain_of_thought"],
        default="zero_shot",
    )
    p_batch.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of source texts to process (0 = all).",
    )
    p_batch.add_argument(
        "--openai-model",
        default="openai/gpt-5.2",
        help="OpenRouter model id for the OpenAI track.",
    )
    p_batch.add_argument(
        "--gemini-model",
        default="google/gemini-2.5-pro-preview",
        help="OpenRouter model id for the Gemini track.",
    )
    p_batch.add_argument(
        "--output",
        default=os.path.join("outputs", "onestop_english_simplifications.csv"),
    )
    p_batch.add_argument(
        "--export-text-dir",
        default=None,
        help="Optional folder where each generated simplification is also written as a separate .txt file.",
    )
    p_batch.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    p_batch.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter.")
    p_batch.add_argument("--seed", type=int, default=42, help="Optional deterministic seed.")
    p_batch.add_argument("--max-retries", type=int, default=3)
    p_batch.add_argument("--base-sleep-s", type=float, default=1.5)
    p_batch.add_argument(
        "--aligned-corpus",
        default=None,
        metavar="DIR",
        help=(
            "OneStopEnglishCorpus root (or Texts-Together-OneCSVperFile). "
            "If omitted: use OSE_CORPUS_ROOT, else ./data/OneStopEnglishCorpus if it exists."
        ),
    )
    p_batch.set_defaults(func=cmd_batch)

    p_exp = sub.add_parser(
        "experiments",
        help="Grid over strategies and temperatures for aligned Advanced texts only.",
    )
    p_exp.add_argument(
        "--aligned-corpus",
        required=True,
        metavar="DIR",
        help="OneStopEnglishCorpus root (or Texts-Together-OneCSVperFile); must exist.",
    )
    p_exp.add_argument(
        "--source-levels",
        default="Advanced",
        help="Comma-separated source levels to use for the experiment grid.",
    )
    p_exp.add_argument(
        "--target-level",
        default="Elementary",
        help="Target reading level for generated simplifications.",
    )
    p_exp.add_argument(
        "--strategies",
        default="zero_shot,few_shot,chain_of_thought",
        help="Comma-separated strategies.",
    )
    p_exp.add_argument(
        "--temperatures",
        default="0.0,0.2,0.5,0.8,1.0",
        help="Comma-separated temperatures.",
    )
    p_exp.add_argument("--provider", choices=["openai", "gemini", "both"], default="both")
    p_exp.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of source texts to sample for a pilot run (0 = use all selected texts).",
    )
    p_exp.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling a pilot subset with --limit.",
    )
    p_exp.add_argument("--openai-model", default="openai/gpt-5.2")
    p_exp.add_argument("--gemini-model", default="google/gemini-2.5-pro-preview")
    p_exp.add_argument("--top-p", type=float, default=0.9)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument(
        "--output",
        default=os.path.join("outputs", "onestop_english_advanced_experiments.csv"),
    )
    p_exp.add_argument(
        "--export-text-dir",
        default=None,
        help="Optional folder where each experiment output is also written as a separate .txt file for local review or external export.",
    )
    p_exp.add_argument("--max-retries", type=int, default=3)
    p_exp.add_argument("--base-sleep-s", type=float, default=1.5)
    p_exp.set_defaults(func=cmd_experiments)

    p_smoke = sub.add_parser("smoke", help="Quick OpenRouter connectivity check (both tracks).")
    p_smoke.add_argument("--openai-model", default="openai/gpt-5.2")
    p_smoke.add_argument("--gemini-model", default="google/gemini-2.5-pro-preview")
    p_smoke.add_argument("--temperature", type=float, default=0.2)
    p_smoke.add_argument("--top-p", type=float, default=0.9)
    p_smoke.add_argument("--seed", type=int, default=42)
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    load_dotenv()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
