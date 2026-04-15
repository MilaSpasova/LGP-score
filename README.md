# Benchmarking Linguistic Growth Potential (LGP)

> **A Comparative Analysis of LLM-Simplified vs. Human-Simplified Texts**

This repository contains the computational framework, datasets, and analysis pipeline for evaluating whether Large Language Models (LLMs) preserve the "growth gap" necessary for effective language acquisition in children.

## 📌 Project Overview
Traditional text simplification often focuses solely on readability, potentially removing the "challenge" required for learning. This project implements **Krashen’s Input Hypothesis ($i+1$)** by defining and measuring **Linguistic Growth Potential (LGP)**—the balance between decodable text and the introduction of new lexical/syntactic structures.

## 🛠 Tech Stack & Methodology
- **Language:** Python 3.x
- **Models:** OpenRouter (OpenAI + Google Gemini model IDs, e.g. `openai/gpt-5.2`, `google/gemini-2.5-pro-preview`)
- **Corpora:** Newsela, OneStopEnglish
- **Psycholinguistic Databases:** MRC Psycholinguistic Database (AoA and related norms in MRC where available)

### Key Features
* **LGP Scoring Pipeline:** A custom Python engine that tokenizes text and cross-references tokens with psycholinguistic variables:
    * **Age of Acquisition (AoA):** Targeting the "growth window" (1–2 years above current age).
    * **Concreteness & Imageability:** Ensuring new vocabulary remains bridgeable to existing knowledge.
    * **Tier 2 Vocabulary:** Automated identification of high-utility academic words.
* **Prompt Engineering Suite:** Evaluation of **Zero-shot**, **Few-shot**, and **Chain-of-Thought (CoT)** strategies, including **Prompt Sketching** for targeted linguistic constraints.
* **Interactive Analysis Dashboard:** A visual tool for educators to input text and receive:
    * Side-by-side AI vs. Human comparison.
    * Visual "Tier" and "LGP" mapping.
    * Instructional "frustration" alerts (for $i+3$ levels).

## Evaluation Logic
The LGP score will be calculated by weighting words within the $i+1$ window. The goal is to maximize pedagogical value while avoiding the "frustration level" ($i+3$, or 4+ years above the target age).

---

## How everything works

### Repository layout

```
LGP-score/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API keys (copy to .env)
├── .gitignore
│
├── setup_mrc_database.py     # Loads and cleans the MRC psycholinguistic database
├── setup_onestop_english.py  # OneStopEnglish: official aligned corpus on disk only
├── onestop_english_exploration.ipynb  # EDA: official aligned corpus (local clone required)
├── mrc_exploration.ipynb
├── run_onestop_english.py    # CLI: batch / experiments / smoke (OpenRouter OpenAI + Gemini)
│
└── llm_api/                  # LLM integration for text simplification
    ├── __init__.py
    ├── prompts.py            # Zero-shot, Few-shot, Chain-of-Thought templates
    ├── openrouter.py         # OpenAI SDK → OpenRouter Chat Completions
    ├── openai_client.py      # ChatGPT-class models via OpenRouter (`openai/...`)
    └── gemini_client.py      # Gemini via OpenRouter (`google/...`)
```

### 1. MRC psycholinguistic database (`setup_mrc_database.py`)

The **MRC database** provides psycholinguistic variables for English words (e.g. Age of Acquisition, Concreteness, Imageability, Familiarity). The script:

- Loads the dataset from Hugging Face: `StephanAkkerman/MRC-psycholinguistic-database` (train split).
- Normalises column names and renames key ones using a fixed mapping, e.g.:
  - `age of acquisition` → `aoa`
  - `concreteness` → `conc`
  - `imageability` → `img`
  - `familiarity` → `fam`
  - `kf written frequency` → `kf_freq`
- Cleans the `word` column (removes `&`, strips whitespace).
- Converts psycholinguistic columns to numeric, treating `0` as missing.
- Drops rows where **all** of `fam`, `conc`, and `img` are missing (so every retained word has at least one rating).

Under the hood the main entrypoint is:

```python
from setup_mrc_database import setup_mrc_database

mrc_df = setup_mrc_database()
```

**CLI usage:** Run `python setup_mrc_database.py` to test the load and see a small preview of words with complete `fam`/`conc`/`img` ratings. The returned DataFrame is intended for use in the LGP pipeline (e.g. mapping tokens to AoA/conc/img for scoring).

### 2. OneStopEnglish corpus (`setup_onestop_english.py`)

**OneStopEnglish** is used here only from the **official** [OneStopEnglishCorpus](https://github.com/nishkalavallabhi/OneStopEnglishCorpus): folder **`Texts-Together-OneCSVperFile`** (or the repo root), **189 articles × 3 aligned levels**. There is **no** Hugging Face fallback in this repo.

**Loader:** `load_onestop_english_aligned(corpus_dir)` — returns a DataFrame with columns `text`, `level`, `level_id` (0/1/2), `split` (always `aligned`), and `story_id` (one per article). The CSV column **Advanced** is stored as level **Advance**.

**Resolving the corpus directory** (same rules for `python setup_onestop_english.py` and `run_onestop_english.py batch`):

1. Explicit path: first CLI argument to `setup_onestop_english.py`, or `--aligned-corpus DIR` for `batch`.
2. Else environment variable **`OSE_CORPUS_ROOT`** (must be an existing directory).
3. Else **`./data/OneStopEnglishCorpus`** under the current working directory, if that folder exists.

If nothing resolves, the CLI prints an error with clone instructions.

**Clone the corpus** (not in git):

```bash
git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git data/OneStopEnglishCorpus
```

**Exploration notebook:** [`onestop_english_exploration.ipynb`](onestop_english_exploration.ipynb) sets `OSE_CORPUS_ROOT` to `data/OneStopEnglishCorpus` by default. Open Jupyter with the project folder as the working directory, or change that variable in the notebook to your clone path.

**CLI preview:** From the project root with the clone in place:

```bash
python setup_onestop_english.py
```

Or: `python setup_onestop_english.py /path/to/OneStopEnglishCorpus`

**In code:** `from setup_onestop_english import load_onestop_english_aligned, resolve_onestop_corpus_dir`.

### 3. LLM API clients (`llm_api/`)

Both simplification tracks use **[OpenRouter](https://openrouter.ai/)**’s OpenAI-compatible Chat Completions API, so one key can route to OpenAI-hosted and Google-hosted models.

| File | Purpose |
|------|--------|
| `prompts.py` | Builds chat messages for a given **strategy** (see below). Used by both clients. |
| `openrouter.py` | Base URL `https://openrouter.ai/api/v1`, optional `OPENROUTER_HTTP_REFERER` / `OPENROUTER_APP_TITLE`. |
| `openai_client.py` | Default model `openai/gpt-5.2` (override with any OpenRouter `openai/...` id). Needs `OPENROUTER_API_KEY`. |
| `gemini_client.py` | Default model `google/gemini-2.5-pro-preview` (override with any OpenRouter `google/...` id). Same key. |

**Prompt strategies** (thesis-relevant):

- **`zero_shot`** — Ask the model to simplify for a given reading level with no examples.
- **`few_shot`** — Include one or two example (original → simplified) pairs before the target text.
- **`chain_of_thought`** — Ask the model to (silently) identify hard words/phrases and simplify structure, then output only the final simplified text.

**Example (single call):**

```python
from llm_api.openai_client import simplify_with_openai
from llm_api.gemini_client import simplify_with_gemini

text = "The committee convened to deliberate on the proposal."
level = "Elementary"

# OpenAI track on OpenRouter
simple_openai = simplify_with_openai(text=text, target_level=level, strategy="zero_shot")

# Gemini track on OpenRouter
simple_gemini = simplify_with_gemini(text=text, target_level=level, strategy="zero_shot")
```

### 4. OneStopEnglish CLI (`run_onestop_english.py`)

Subcommands:

| Subcommand | Purpose |
|------------|--------|
| `batch` | Run the corpus through one or both LLMs; one row per (source text × provider) in a CSV. |
| `experiments` | Grid over strategies and temperatures for **aligned Advanced** texts only (requires `--aligned-corpus`). Default CSV: `outputs/onestop_english_advanced_experiments.csv`. |
| `smoke` | Quick OpenRouter check: one short call on each track (replaces a separate test script). |

**`batch` — what it does**

1. Resolves the corpus directory (`--aligned-corpus`, then `OSE_CORPUS_ROOT`, then `./data/OneStopEnglishCorpus`).
2. Loads aligned rows via `load_onestop_english_aligned`.
3. Optionally limits rows (`--limit`).
4. For each text, calls the OpenAI and/or Gemini **OpenRouter** tracks (with retries).
5. Writes to a CSV (default: `outputs/onestop_english_simplifications.csv`).

**`batch` options:**

| Option | Meaning |
|--------|--------|
| `--provider` | `openai`, `gemini`, or `both` |
| `--strategy` | `zero_shot`, `few_shot`, or `chain_of_thought` |
| `--limit N` | Process only the first N texts (0 = all rows, 189 × 3 = 567) |
| `--openai-model` | OpenRouter id, e.g. `openai/gpt-5.2` |
| `--gemini-model` | OpenRouter id, e.g. `google/gemini-2.5-pro-preview` |
| `--output` | Output CSV path |
| `--temperature` | Optional sampling temperature |
| `--max-retries` | Retries per API call on failure |
| `--base-sleep-s` | Base delay for exponential backoff |
| `--aligned-corpus DIR` | Override corpus root; otherwise env / default path (see §2) |

Run `python run_onestop_english.py experiments -h` for the full experiment grid flags (`--strategies`, `--temperatures`, etc.).

---

## Getting started

1. **Clone the repository** (if you have not already):
   ```bash
   git clone <repo-url>
   cd LGP-score
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Configure API keys:** Copy `.env.example` to `.env` and set your [OpenRouter](https://openrouter.ai/) key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
   Optionally set `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` for OpenRouter rankings.

5. **Quick checks:**
   - MRC (needs internet for Hugging Face):
     ```bash
     python setup_mrc_database.py
     ```
   - OneStopEnglish (needs local clone under `data/OneStopEnglishCorpus`, or set `OSE_CORPUS_ROOT` / pass a path):
     ```bash
     python setup_onestop_english.py
     ```

   Optional OpenRouter check (needs `OPENROUTER_API_KEY` in `.env`):
   ```bash
   python run_onestop_english.py smoke
   ```

6. **OneStop English exploration notebook:** Clone the official corpus (see §2), then run [`onestop_english_exploration.ipynb`](onestop_english_exploration.ipynb) with the project as the working directory.

7. **Run a small batch** (e.g. 5 rows, both providers; corpus path resolved like §2):
   ```bash
   python run_onestop_english.py batch --provider both --strategy zero_shot --limit 5
   ```
   Output is written to `outputs/onestop_english_simplifications.csv`.

8. **Run full batch** (all aligned rows, both LLMs):
   ```bash
   python run_onestop_english.py batch --provider both --limit 0
   ```
   This will take a while and use API credits; use `--limit` for testing first.

---

## Data flow (high level)

1. **Human benchmark:** OneStopEnglish gives you texts already simplified by humans at three levels.
2. **AI simplifications:** The same (or similar) source content can be sent through OpenRouter to ChatGPT-class and Gemini models with a target level and strategy; `run_onestop_english.py batch` (or `experiments`) does this at scale.
3. **LGP pipeline (planned):** Tokenise both human- and AI-simplified texts, map words to the MRC database (AoA, concreteness, imageability), then compute LGP scores to compare how much “growth potential” (i+1) each version retains.

The current code covers loading MRC, loading OneStopEnglish, calling both LLMs with configurable strategies, and saving batch results; the actual LGP scoring and dashboard are to be added on top of this.

---
