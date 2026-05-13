# Benchmarking Linguistic Growth Potential (LGP)

This repository contains the computational pipeline for the thesis comparison between human simplification and LLM simplification on the OneStopEnglish corpus.

The current implementation is fully local for evaluation. An earlier version of the workflow planned to use CTAP as an external feature-extraction service, but the final repository no longer depends on that platform.

## Structure

```text
LGP-score/
|-- dashboard/
|-- run_onestop_english.py
|-- analyze_thesis_metrics.py
|-- rank_prompt_configs.py
|-- summarize_thesis_results.py
|-- append_fk_mtld_metrics.py
|-- setup_mrc_database.py
|-- setup_onestop_english.py
|-- methodology_analysis.py        # legacy wrapper
|-- rank_prompt_experiments.py     # legacy wrapper
|-- summarize_results.py           # legacy wrapper
|-- FKandMTLD.py                   # legacy wrapper
|-- setup_onestopenglish.py        # legacy wrapper
|-- lgp_pipeline/
|-- llm_api/
|-- notebooks/
|-- data/
`-- outputs/
```

Use the repo as four layers:
- `run_onestop_english.py`: generation entrypoint for OpenRouter experiments
- `analyze_thesis_metrics.py`, `rank_prompt_configs.py`, `summarize_thesis_results.py`: analysis and reporting entrypoints
- `lgp_pipeline/`: reusable local metric logic
- `llm_api/`: provider clients and prompt templates
- `dashboard/`: teacher-facing web UI for benchmark review, vocabulary inspection, live simplification, and questionnaire capture

The similarly named older files are only compatibility wrappers so earlier commands still run:
- `methodology_analysis.py`
- `rank_prompt_experiments.py`
- `summarize_results.py`
- `FKandMTLD.py`
- `setup_onestopenglish.py`

## File Audit

Useful and active:
- `run_onestop_english.py`: main experiment runner
- `analyze_thesis_metrics.py`: core local analysis pipeline for Advanced, Elementary, and LLM texts
- `rank_prompt_configs.py`: ranks prompt and temperature conditions
- `summarize_thesis_results.py`: builds thesis-ready summary tables
- `append_fk_mtld_metrics.py`: auxiliary script for adding FK and MTLD to legacy exported result files
- `setup_mrc_database.py`: validates and previews the MRC resource
- `setup_onestop_english.py`: canonical OneStopEnglish loader
- `lgp_pipeline/`: preprocessing, psycholinguistics, Tier 2 proxy, text metrics, and SBERT
- `llm_api/`: OpenRouter client layer and prompt templates
- `notebooks/`: exploratory analysis only, not required for the pipeline

Useful but legacy:
- `methodology_analysis.py`
- `rank_prompt_experiments.py`
- `summarize_results.py`
- `FKandMTLD.py`
- `setup_onestopenglish.py`

Not part of the project code:
- `venv/`
- `__pycache__/`

## Core Inputs

### OneStopEnglish corpus

The code expects the official corpus under `data/OneStopEnglishCorpus`, or a path passed with `--aligned-corpus`, or `OSE_CORPUS_ROOT`.

### LLM simplification CSV

Optional input for the comparison stage. This is produced by `run_onestop_english.py` and then passed into `analyze_thesis_metrics.py`.

## Main Outputs

The main analysis script writes:
- [outputs/methodology_text_metrics.csv](D:/Year-3-Uni/thesis/Code/LGP-score/outputs/methodology_text_metrics.csv:1)
- [outputs/methodology_pairwise_comparisons.csv](D:/Year-3-Uni/thesis/Code/LGP-score/outputs/methodology_pairwise_comparisons.csv:1)

`methodology_text_metrics.csv` contains one row per text variant with:
- source metadata
- local psycholinguistic metrics for human and LLM texts
- `FK`
- `MTLD`
- z-scores
- `CLI`
- `tier2_proxy_token_ratio`

`methodology_pairwise_comparisons.csv` contains one row per simplified-vs-original comparison with:
- `delta_cli`
- `delta_aoa`
- `delta_concreteness`
- `delta_imageability`
- `delta_fk_grade`
- `delta_mtld`
- `delta_tier2_proxy_token_ratio`
- `semantic_similarity_sbert`

## Reproducible Command Set

### Launch the teacher dashboard

```powershell
venv\Scripts\streamlit run dashboard/streamlit_app.py
```

The dashboard provides:
- benchmark review with `Human Advanced`, `Human Elementary`, `Few-shot 0.5`, and `Zero-shot 0.0`
- vocabulary review with retained / removed / added AVL terms
- an optional live simplification page
- an in-app teacher questionnaire
- local CSV / JSON feedback capture in `outputs/dashboard_feedback`

See [dashboard/README.md](D:/Year-3-Uni/thesis/Code/LGP-score/dashboard/README.md:1) for the expected input files.

### Validate setup

```powershell
python setup_mrc_database.py
python setup_onestop_english.py
python run_onestop_english.py smoke
```

### OpenRouter setup

Create a local `.env` file in the repo root with:

```powershell
OPENROUTER_API_KEY=sk-or-v1-your-real-key-here
```

Optional:

```powershell
OPENROUTER_HTTP_REFERER=https://your-project-url.example
OPENROUTER_APP_TITLE=LGP-score
```

### Run prompt and temperature experiment

```powershell
python run_onestop_english.py experiments `
  --aligned-corpus data/OneStopEnglishCorpus `
  --source-levels Advanced `
  --target-level Elementary `
  --strategies zero_shot,few_shot,chain_of_thought `
  --temperatures 0.0,0.2,0.5,0.8 `
  --output outputs/onestop_english_advanced_experiments.csv
```

### Run the 12-text pilot

```powershell
python run_onestop_english.py experiments `
  --aligned-corpus data/OneStopEnglishCorpus `
  --provider openai `
  --source-levels Advanced `
  --target-level Elementary `
  --strategies zero_shot,few_shot,chain_of_thought `
  --temperatures 0.0,0.2,0.5 `
  --limit 12 `
  --sample-seed 42 `
  --export-text-dir outputs/pilot_text_exports `
  --output outputs/pilot_12text_experiments.csv
```

### Rank pilot conditions

First pass:

```powershell
python rank_prompt_configs.py `
  --experiments outputs/pilot_12text_experiments.csv `
  --output-dir outputs/pilot_ranking
```

Deeper ranking with local metrics:

```powershell
python rank_prompt_configs.py `
  --experiments outputs/pilot_12text_experiments.csv `
  --text-metrics outputs/pilot_methodology_text_metrics.csv `
  --pairwise outputs/pilot_methodology_pairwise_comparisons.csv `
  --output-dir outputs/pilot_ranking_local
```

### Generate LLM simplifications for thesis comparison

```powershell
python run_onestop_english.py batch `
  --aligned-corpus data/OneStopEnglishCorpus `
  --provider both `
  --source-levels Advanced `
  --target-level Elementary `
  --strategy zero_shot `
  --temperature 0.5 `
  --top-p 0.9 `
  --seed 42 `
  --limit 0 `
  --export-text-dir outputs/simplification_text_exports `
  --output outputs/onestop_english_simplifications.csv
```

### Run thesis analysis

```powershell
python analyze_thesis_metrics.py `
  --aligned-corpus data/OneStopEnglishCorpus `
  --simplifications-csv outputs/onestop_english_simplifications.csv
```

### Run human-only benchmark

```powershell
python analyze_thesis_metrics.py `
  --aligned-corpus data/OneStopEnglishCorpus
```

### Build thesis summary tables

```powershell
python summarize_thesis_results.py `
  --text-metrics outputs/methodology_text_metrics.csv `
  --pairwise outputs/methodology_pairwise_comparisons.csv `
  --output-dir outputs/summaries
```

## Methodology Mapping

Primary thesis metrics:
- `AoA`
- `Concreteness`
- `Imageability`
- `Tier 2 proxy` from the AVL core list

Supporting metrics:
- `Flesch-Kincaid`
- `MTLD`
- `SBERT`

The local pipeline is reproducible if you keep fixed:
- corpus version
- OpenRouter model IDs
- prompt strategy
- `temperature`
- `top_p`
- `seed`

Remaining reproducibility risks:
- OpenRouter models may still change behind the same model ID over time
- the qualitative dashboard evaluation has not yet been completed, even though the dashboard and questionnaire are implemented
