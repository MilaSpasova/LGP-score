from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
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


def cmd_batch(args: argparse.Namespace) -> int:
    try:
        corpus_root = resolve_onestop_corpus_dir(explicit=args.aligned_corpus)
    except FileNotFoundError as e:
        print(f"batch: {e}", file=sys.stderr)
        return 1
    df = load_onestop_english_aligned(corpus_root)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows: list[dict[str, object]] = []
    for i, r in df.reset_index(drop=False).iterrows():
        source_id = int(r["index"])
        source_text = str(r["text"])
        level = str(r["level"])
        level_id = int(r["level_id"])
        split = str(r["split"])

        common = {
            "source_id": source_id,
            "level": level,
            "level_id": level_id,
            "split": split,
            "source_text": source_text,
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
                target_level=level,
                strategy=args.strategy,  # type: ignore[arg-type]
                model=args.openai_model,
                temperature=args.temperature,
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

        if args.provider in ("gemini", "both"):
            simplified, err = _call_with_retries(
                simplify_with_gemini,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep_s,
                text=source_text,
                target_level=level,
                strategy=args.strategy,  # type: ignore[arg-type]
                model=args.gemini_model,
                temperature=args.temperature,
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
    advanced = df[df["level_id"] == 2].copy().reset_index(drop=True)
    if len(advanced) != 189:
        print(f"Warning: expected 189 Advanced rows, found {len(advanced)}.")

    rows: list[dict[str, object]] = []
    total = len(advanced) * len(strategies) * len(temperatures) * 2
    done = 0
    for source_id, r in advanced.iterrows():
        source_text = str(r["text"])
        story_id = str(r["story_id"]) if "story_id" in advanced.columns else None
        for strategy in strategies:
            for temperature in temperatures:
                strategy_typed: PromptStrategy = strategy  # type: ignore[assignment]
                common = {
                    "source_id": int(source_id),
                    "story_id": story_id,
                    "split": "aligned",
                    "level": "Advance",
                    "level_id": 2,
                    "source_text": source_text,
                    "strategy": strategy,
                    "temperature": temperature,
                    "created_at": _utc_now_iso(),
                }

                simplified, err = _call_with_retries(
                    simplify_with_openai,
                    max_retries=args.max_retries,
                    base_sleep_s=args.base_sleep_s,
                    text=source_text,
                    target_level="Advance",
                    strategy=strategy_typed,
                    model=args.openai_model,
                    temperature=temperature,
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
                done += 1

                simplified, err = _call_with_retries(
                    simplify_with_gemini,
                    max_retries=args.max_retries,
                    base_sleep_s=args.base_sleep_s,
                    text=source_text,
                    target_level="Advance",
                    strategy=strategy_typed,
                    model=args.gemini_model,
                    temperature=temperature,
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
    )
    print("OpenAI track OK")
    print(openai_out[:200].replace("\n", " "))

    gemini_out = simplify_with_gemini(
        text=text,
        target_level=level,
        strategy="zero_shot",
        model=args.gemini_model,
        temperature=args.temperature,
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
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: provider/model default).",
    )
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
        "--strategies",
        default="zero_shot,few_shot,chain_of_thought",
        help="Comma-separated strategies.",
    )
    p_exp.add_argument(
        "--temperatures",
        default="0.0,0.2,0.5,0.8,1.0",
        help="Comma-separated temperatures.",
    )
    p_exp.add_argument("--openai-model", default="openai/gpt-5.2")
    p_exp.add_argument("--gemini-model", default="google/gemini-2.5-pro-preview")
    p_exp.add_argument(
        "--output",
        default=os.path.join("outputs", "onestop_english_advanced_experiments.csv"),
    )
    p_exp.add_argument("--max-retries", type=int, default=3)
    p_exp.add_argument("--base-sleep-s", type=float, default=1.5)
    p_exp.set_defaults(func=cmd_experiments)

    p_smoke = sub.add_parser("smoke", help="Quick OpenRouter connectivity check (both tracks).")
    p_smoke.add_argument("--openai-model", default="openai/gpt-5.2")
    p_smoke.add_argument("--gemini-model", default="google/gemini-2.5-pro-preview")
    p_smoke.add_argument("--temperature", type=float, default=0.2)
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    load_dotenv()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
