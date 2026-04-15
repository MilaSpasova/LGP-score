from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

LEVEL_ID_TO_TEXT: dict[int, str] = {0: "Elementary", 1: "Intermediate", 2: "Advance"}

_ALIGNED_SUBDIR = "Texts-Together-OneCSVperFile"
_ALIGNED_COL_TO_LEVEL_ID: tuple[tuple[str, int], ...] = (
    ("Elementary", 0),
    ("Intermediate", 1),
    ("Advanced", 2),
)

_CORPUS_CLONE_HINT = (
    "Clone the official corpus, for example:\n"
    "  git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git data/OneStopEnglishCorpus\n"
    "Then pass --aligned-corpus, set OSE_CORPUS_ROOT, or run from the project root with that folder present."
)


def resolve_onestop_corpus_dir(*, explicit: str | os.PathLike[str] | None = None) -> Path:
    """
    Resolve the OneStopEnglishCorpus root directory (repo root or a folder containing
    ``Texts-Together-OneCSVperFile``).

    Precedence: non-empty ``explicit`` path, then environment variable ``OSE_CORPUS_ROOT``,
    then ``./data/OneStopEnglishCorpus`` under the current working directory if it exists.

    Raises
    ------
    FileNotFoundError
        If no existing directory is resolved.
    """
    if explicit is not None and str(explicit).strip():
        p = Path(explicit).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"OneStop corpus path is not an existing directory: {p}\n{_CORPUS_CLONE_HINT}"
        )

    env = os.environ.get("OSE_CORPUS_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"OSE_CORPUS_ROOT is set but is not an existing directory: {p}\n{_CORPUS_CLONE_HINT}"
        )

    default = (Path.cwd() / "data" / "OneStopEnglishCorpus").resolve()
    if default.is_dir():
        return default

    raise FileNotFoundError(
        "Could not find OneStopEnglishCorpus. "
        "Pass a path, set OSE_CORPUS_ROOT, or create data/OneStopEnglishCorpus.\n"
        f"{_CORPUS_CLONE_HINT}"
    )


def _resolve_aligned_csv_dir(corpus_dir: str | os.PathLike[str]) -> Path:
    root = Path(corpus_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Not a directory: {root}\n"
            "Clone or download the official corpus, then set the path to the repo root "
            "or to the folder Texts-Together-OneCSVperFile inside it:\n"
            "  https://github.com/nishkalavallabhi/OneStopEnglishCorpus"
        )
    nested = root / _ALIGNED_SUBDIR
    if nested.is_dir():
        return nested
    return root


def _read_one_aligned_csv(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Could not read CSV: {path}")


def _joined_article_text(series: pd.Series) -> str:
    parts: list[str] = []
    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return "\n\n".join(parts)


def load_onestop_english_aligned(corpus_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """
    Load the **official** OneStop English release with guaranteed cross-level alignment.

    Point ``corpus_dir`` at either the repository root or the folder
    ``Texts-Together-OneCSVperFile`` from
    https://github.com/nishkalavallabhi/OneStopEnglishCorpus — one CSV per article,
    three columns (Elementary / Intermediate / Advanced), paragraph-aligned rows.

    Returns a long DataFrame with columns ``text``, ``level``, ``level_id``, ``split``,
    and ``story_id``. ``split`` is ``\"aligned\"`` for every row. The CSV column name
    **Advanced** is stored as level **Advance** (``level_id`` 2) for a single consistent
    label across the project.
    """
    csv_dir = _resolve_aligned_csv_dir(corpus_dir)
    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No .csv files under {csv_dir}")

    rows: list[dict[str, object]] = []
    for p in paths:
        story_id = p.stem
        df = _read_one_aligned_csv(p)
        df.columns = pd.Index([str(c).strip() for c in df.columns])
        missing = {c for c, _ in _ALIGNED_COL_TO_LEVEL_ID if c not in df.columns}
        if missing:
            raise ValueError(f"{p.name}: missing expected columns {sorted(missing)}")

        for col, lid in _ALIGNED_COL_TO_LEVEL_ID:
            text = _joined_article_text(df[col])
            rows.append(
                {
                    "text": text,
                    "level": LEVEL_ID_TO_TEXT[lid],
                    "level_id": lid,
                    "split": "aligned",
                    "story_id": story_id,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["story_id", "level_id"], kind="stable").reset_index(drop=True)
    return out


if __name__ == "__main__":
    explicit_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        root = resolve_onestop_corpus_dir(explicit=explicit_arg)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    df = load_onestop_english_aligned(root)
    print(df.head(5))
    print(df["level"].value_counts())
