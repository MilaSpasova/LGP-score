from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from typing import Callable, Literal, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from setup_onestop_english import load_onestop_english

from llm_api.gemini_client import simplify_with_gemini
from llm_api.openai_client import simplify_with_openai


Provider = Literal["openai", "gemini", "both"]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OneStopEnglish through OpenAI/Gemini simplification.")
    parser.add_argument("--provider", choices=["openai", "gemini", "both"], default="both")
    parser.add_argument("--strategy", choices=["zero_shot", "few_shot", "chain_of_thought"], default="zero_shot")
    parser.add_argument("--limit", type=int, default=10, help="Number of source texts to process (0 = all).")
    parser.add_argument("--openai-model", default="gpt-5.2-chat-latest")
    parser.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--output", default=os.path.join("outputs", "onestop_english_simplifications.csv"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--base-sleep-s", type=float, default=1.5)
    args = parser.parse_args()

    load_dotenv()

    df = load_onestop_english()
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

        if args.provider in ("openai", "both"):
            simplified, err = _call_with_retries(
                simplify_with_openai,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep_s,
                text=source_text,
                target_level=level,
                strategy=args.strategy,  # type: ignore[arg-type]
                model=args.openai_model,
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

        if args.provider in ("gemini", "both"):
            simplified, err = _call_with_retries(
                simplify_with_gemini,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep_s,
                text=source_text,
                target_level=level,
                strategy=args.strategy,  # type: ignore[arg-type]
                model=args.gemini_model,
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

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} source texts...")

    out_df = pd.DataFrame(rows)
    _ensure_parent_dir(args.output)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

