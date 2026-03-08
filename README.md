# Benchmarking Linguistic Growth Potential (LGP)

> **A Comparative Analysis of LLM-Simplified vs. Human-Simplified Texts**

This repository contains the computational framework, datasets, and analysis pipeline for evaluating whether Large Language Models (LLMs) preserve the "growth gap" necessary for effective language acquisition in children.

## 📌 Project Overview
Traditional text simplification often focuses solely on readability, potentially removing the "challenge" required for learning. This project implements **Krashen’s Input Hypothesis ($i+1$)** by defining and measuring **Linguistic Growth Potential (LGP)**—the balance between decodable text and the introduction of new lexical/syntactic structures.

## 🛠 Tech Stack & Methodology
- **Language:** Python 3.x
- **Models:** GPT-5.2, Gemini 3
- **Corpora:** Newsela, OneStopEnglish
- **Psycholinguistic Databases:** MRC Psycholinguistic Database, AoA norms (Kuperman et al.)

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
├── setup_onestop_english.py  # Loads the OneStopEnglish corpus from Hugging Face
├── run_batch_onestop_english.py  # Runs OpenAI + Gemini on OSE texts, saves CSV
│
└── llm_api/                  # LLM integration for text simplification
    ├── __init__.py
    ├── prompts.py            # Zero-shot, Few-shot, Chain-of-Thought templates
    ├── openai_client.py      # GPT-5.2 (or newer) via OpenAI API
    └── gemini_client.py      # Gemini 3 / 3.1 via Google API
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

**OneStopEnglish** is a corpus of human-simplified texts at three reading levels (Elementary, Intermediate, Advance). It serves as the human benchmark for comparing LLM simplifications.

- **Source:** Hugging Face `SetFit/onestop_english` (567 texts: 192 train + 375 test)
- **Columns in the returned DataFrame:** `text`, `level` (name), `level_id` (0/1/2), `split` (train/test)

**Usage:** Run `python setup_onestop_english.py` to preview the data, or import and call `load_onestop_english()` from your own scripts.

### 3. LLM API clients (`llm_api/`)

Two providers are integrated so you can generate **AI-simplified** versions of texts and compare them to human-simplified ones.

| File | Purpose |
|------|--------|
| `prompts.py` | Builds the simplification prompt for a given **strategy** (see below). Used by both clients. |
| `openai_client.py` | Calls the OpenAI Chat Completions API (default model: `gpt-5.2-chat-latest`). Needs `OPENAI_API_KEY`. |
| `gemini_client.py` | Calls the Gemini API (default: `gemini-3.1-pro-preview`). Uses `google-genai` if available; falls back to `google-generativeai`. Needs `GOOGLE_API_KEY` or `GEMINI_API_KEY`. |

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

# OpenAI (GPT-5.2 or newer)
simple_openai = simplify_with_openai(text=text, target_level=level, strategy="zero_shot")

# Gemini (3 or 3.1)
simple_gemini = simplify_with_gemini(text=text, target_level=level, strategy="zero_shot")
```

### 4. Batch script (`run_batch_onestop_english.py`)

This script runs the **OneStopEnglish** corpus through one or both LLMs and writes a CSV with source text, level, provider, model, strategy, and the simplified output (or error).

**What it does:**

1. Loads OneStopEnglish (train + test).
2. Optionally limits the number of texts (`--limit`).
3. For each text, calls OpenAI and/or Gemini (with retries).
4. Writes one row per (source text × provider) to a CSV (default: `outputs/onestop_english_simplifications.csv`).

**Options:**

| Option | Meaning |
|--------|--------|
| `--provider` | `openai`, `gemini`, or `both` |
| `--strategy` | `zero_shot`, `few_shot`, or `chain_of_thought` |
| `--limit N` | Process only the first N texts (0 = all 567) |
| `--openai-model` | e.g. `gpt-5.2-chat-latest` |
| `--gemini-model` | e.g. `gemini-3.1-pro-preview` |
| `--output` | Output CSV path |
| `--max-retries` | Retries per API call on failure |
| `--base-sleep-s` | Base delay for exponential backoff |

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

4. **Configure API keys:** Copy `.env.example` to `.env` and set your keys:
   ```
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   ```
   (Gemini also accepts `GEMINI_API_KEY`.)

5. **Quick checks:**
   ```bash
   python setup_mrc_database.py
   python setup_onestop_english.py
   ```

6. **Run a small batch** (e.g. 5 texts, both providers):
   ```bash
   python run_batch_onestop_english.py --provider both --strategy zero_shot --limit 5
   ```
   Output is written to `outputs/onestop_english_simplifications.csv`.

7. **Run full comparison** (all 567 texts, both LLMs):
   ```bash
   python run_batch_onestop_english.py --provider both --limit 0
   ```
   This will take a while and use API credits; use `--limit` for testing first.

---

## Data flow (high level)

1. **Human benchmark:** OneStopEnglish gives you texts already simplified by humans at three levels.
2. **AI simplifications:** The same (or similar) source content can be sent to GPT-5.2 and Gemini with a target level and strategy; the batch script does this for every OSE text.
3. **LGP pipeline (planned):** Tokenise both human- and AI-simplified texts, map words to the MRC database (AoA, concreteness, imageability), then compute LGP scores to compare how much “growth potential” (i+1) each version retains.

The current code covers loading MRC, loading OneStopEnglish, calling both LLMs with configurable strategies, and saving batch results; the actual LGP scoring and dashboard are to be added on top of this.

---
