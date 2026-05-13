# Codebase File Guide

This document explains what each important file in the `LGP-score` repository does, why it exists, and how it is useful to the thesis project.

The aim is not only to list files, but to explain how they fit together as a working system.

## 1. High-Level Structure

The repository can be understood as six layers:

1. `setup` files  
These load or validate the core data resources used by the project.

2. `generation` files  
These generate LLM simplifications through OpenRouter.

3. `analysis` files  
These compute the local metric pipeline for human and LLM texts.

4. `ranking and summary` files  
These compare prompt conditions and prepare compact thesis tables.

5. `internal pipeline modules`  
These contain reusable logic such as preprocessing, lexical metrics, and semantic similarity.

6. `supporting materials`  
These include notebooks, outputs, documentation, and legacy wrappers.

## 2. Root-Level Canonical Files

These are the files you should think of as the main active entrypoints.

### `run_onestop_english.py`

What it does:
- main command-line interface for LLM generation
- loads OneStopEnglish source texts
- runs prompt experiments
- runs larger simplification batches
- exports generated texts to folders
- writes experiment results to CSV

Why it exists:
- this is the operational front door of the generation workflow
- it centralizes how Advanced texts are sent to models and how outputs are recorded

Why it is useful:
- without it, there is no systematic way to create the LLM simplifications used in the thesis
- it makes experiments reproducible because it logs:
  - provider
  - model
  - strategy
  - temperature
  - top-p
  - seed
  - timestamps
  - errors

Main commands it supports:
- `smoke`
- `experiments`
- `batch`

When you use it:
- when running prompt pilots
- when generating the final LLM comparison dataset

### `analyze_thesis_metrics.py`

What it does:
- main local metric pipeline
- loads the OneStopEnglish human benchmark
- optionally loads generated LLM simplifications
- computes text-level metrics for every text
- computes pairwise deltas against the Advanced source text
- writes:
  - `methodology_text_metrics.csv`
  - `methodology_pairwise_comparisons.csv`

Why it exists:
- this file operationalizes the thesis methodology in code
- it is the central bridge between raw texts and thesis-ready metric outputs

Why it is useful:
- it applies one consistent local metric pipeline to:
  - human Advanced texts
  - human Elementary texts
  - LLM-generated texts
- it is the key file for answering the research questions computationally

Important internal responsibilities:
- compute local psycholinguistic summaries
- compute the Tier 2 proxy
- compute FKGL
- compute MTLD
- compute CLI
- compute SBERT similarity

When you use it:
- after you have human texts only
- after you have LLM outputs and want comparison metrics

### `rank_prompt_configs.py`

What it does:
- ranks prompt and temperature conditions from the pilot experiment
- uses generation reliability first
- optionally enriches ranking with deeper metric outputs

Why it exists:
- the thesis needs a principled way to choose a final prompt setting
- this file turns the raw pilot outputs into a comparative decision table

Why it is useful:
- it is how the current best prompt condition was selected
- it prevents the prompt decision from becoming subjective

How it works:
- reads the experiment CSV from `run_onestop_english.py`
- computes:
  - JSON success rate
  - FK mean
  - FK standard deviation
- optionally merges:
  - SBERT similarity
  - human benchmark closeness
- outputs ranked prompt conditions

When you use it:
- after the 12-text pilot
- after a deeper local analysis of pilot outputs

### `summarize_thesis_results.py`

What it does:
- creates compact summary tables from the detailed metric outputs
- summarizes text-level and pairwise outputs
- optionally summarizes prompt experiments too

Why it exists:
- raw document-level CSV files are too large to use directly in the thesis
- this file compresses them into reporting tables

Why it is useful:
- makes it easier to quote results in the thesis
- provides summary CSVs for figures, tables, and appendix use

When you use it:
- after `analyze_thesis_metrics.py`
- after prompt experiments if you want compact reporting tables

### `append_fk_mtld_metrics.py`

What it does:
- appends `Flesch-Kincaid Grade Level` and `Measure of Textual Lexical Diversity` rows to a legacy exported results file

Why it exists:
- this was useful when working with older exported result tables that did not already contain FK and MTLD values

Why it is useful:
- helps preserve compatibility with earlier workflows and archived data

Important note:
- this is not part of the main modern local thesis pipeline
- it is an auxiliary compatibility tool

When you use it:
- only if you are working with older tabular result exports

### `setup_mrc_database.py`

What it does:
- loads and cleans the MRC Psycholinguistic Database from Hugging Face
- standardizes column names
- converts key columns to numeric
- removes unusable rows

Why it exists:
- the local metric pipeline depends on the MRC lexical resource
- this file validates that the underlying psycholinguistic dataset is available and clean enough to use

Why it is useful:
- if something is wrong with the psycholinguistic source data, this file helps detect it early
- it also documents exactly how the raw resource is normalized before use

When you use it:
- when first setting up the environment
- when debugging MRC resource issues

### `setup_onestop_english.py`

What it does:
- canonical loader for the OneStopEnglish corpus
- resolves the corpus path
- loads the aligned CSV structure
- turns each story-level pair into one row

Why it exists:
- the rest of the project depends on the corpus being loaded consistently
- this file gives one official way to access the benchmark corpus

Why it is useful:
- avoids duplicated corpus-loading logic elsewhere
- ensures every other script uses the same text representation

When you use it:
- indirectly, through other scripts
- directly, if you want to inspect or validate corpus loading

## 3. Legacy Wrapper Files

These files exist mostly for backward compatibility.

### `methodology_analysis.py`

What it does:
- thin wrapper that forwards execution to `analyze_thesis_metrics.py`

Why it exists:
- older commands and earlier documentation used this filename

Why it is useful:
- prevents older workflows from breaking immediately

### `rank_prompt_experiments.py`

What it does:
- thin wrapper around `rank_prompt_configs.py`

Why it exists:
- keeps old command names working

### `summarize_results.py`

What it does:
- thin wrapper around `summarize_thesis_results.py`

Why it exists:
- backward compatibility for earlier result-summary commands

### `FKandMTLD.py`

What it does:
- thin wrapper around `append_fk_mtld_metrics.py`

Why it exists:
- the old file name was unclear but may still be referenced somewhere

### `setup_onestopenglish.py`

What it does:
- thin wrapper around `setup_onestop_english.py`

Why it exists:
- keeps older imports and commands alive

Important note about wrappers:
- they are useful for continuity
- they are not the best files to build future work around
- use the clearer canonical file names instead

## 4. `lgp_pipeline/` Internal Modules

This folder contains reusable logic for the local metric pipeline.

Think of it as the implementation core.

### `lgp_pipeline/preprocessing.py`

What it does:
- tokenization and lemmatization support
- content-word extraction
- story-key normalization helpers
- fallback preprocessing when spaCy is unavailable

Why it exists:
- almost every metric in the thesis depends on consistent lexical preprocessing

Why it is useful:
- avoids duplicating tokenization logic across analysis scripts
- makes the pipeline more reproducible

### `lgp_pipeline/psycholinguistics.py`

What it does:
- computes text-level psycholinguistic summaries
- aggregates lexical values like:
  - Age of Acquisition
  - Concreteness
  - Imageability
- calculates coverage-related information

Why it exists:
- this is the file that turns word-level psycholinguistic information into document-level metrics

Why it is useful:
- makes the MRC-based portion of the methodology concrete and reusable

### `lgp_pipeline/tier2.py`

What it does:
- loads the Academic Vocabulary List core lexicon
- computes the AVL-based Tier 2 proxy
- returns token and type counts and ratios

Why it exists:
- the thesis needed a reproducible approximation of Tier 2 vocabulary

Why it is useful:
- this is one of the most educationally important parts of the current pipeline
- it provides the main computational proxy for growth vocabulary

Important conceptual role:
- this is not a gold-standard Tier 2 detector
- it is a reproducible local proxy

### `lgp_pipeline/text_metrics.py`

What it does:
- computes text metrics such as:
  - `Flesch-Kincaid Grade Level`
  - `Measure of Textual Lexical Diversity`

Why it exists:
- these metrics are needed in several scripts
- centralizing them keeps behavior consistent

Why it is useful:
- supports both the main analysis and legacy compatibility tools

### `lgp_pipeline/semantic.py`

What it does:
- computes Sentence-BERT cosine similarity between source and simplified text

Why it exists:
- semantic preservation is a separate concern from readability and vocabulary

Why it is useful:
- helps detect whether a text became easier at the expense of meaning

### `lgp_pipeline/__init__.py`

What it does:
- exposes selected pipeline functions at package level

Why it exists:
- small convenience layer for imports

Why it is useful:
- keeps package access cleaner

## 5. `llm_api/` Internal Modules

This folder contains the model-facing logic for generation.

### `llm_api/openrouter.py`

What it does:
- low-level OpenRouter request logic
- handles authentication and HTTP communication

Why it exists:
- provider-specific network code should not be mixed directly into experiment logic

Why it is useful:
- isolates the API transport layer

### `llm_api/openai_client.py`

What it does:
- OpenAI-track simplification through OpenRouter

Why it exists:
- keeps the OpenAI model path separate from Gemini

Why it is useful:
- makes provider comparisons easier to manage

### `llm_api/gemini_client.py`

What it does:
- Gemini-track simplification through OpenRouter

Why it exists:
- same reason as above: separate provider-facing logic

Why it is useful:
- keeps the project modular if model providers change later

### `llm_api/prompts.py`

What it does:
- stores and builds prompt strategies such as:
  - zero-shot
  - few-shot
  - chain-of-thought

Why it exists:
- prompt logic should be explicit and versionable

Why it is useful:
- makes prompt experiments reproducible
- supports the thesis question about prompting strategy

### `llm_api/__init__.py`

What it does:
- package marker and lightweight import convenience

Why it exists:
- standard package organization

## 6. Data Files and Folders

### `data/OneStopEnglishCorpus/`

What it contains:
- the benchmark corpus used in the thesis

Why it exists:
- this is the main dataset

Why it is useful:
- provides aligned human simplifications across reading levels

### `data/lexicons/acadCore.xlsx`

What it contains:
- the Academic Vocabulary List core lexicon

Why it exists:
- supports the AVL-based Tier 2 proxy

Why it is useful:
- gives a reproducible local academic-vocabulary resource

### `data/advResults` and `data/elemResults`

What they contain:
- archived CTAP exports for the human benchmark

Why they exist:
- they preserve the earlier CTAP-based stage of the project

Why they are useful:
- useful for sanity checking the local fallback pipeline
- useful for method comparison

Important note:
- they are no longer part of the active main pipeline

## 7. Output Folders

### `outputs/`

What it contains:
- experiment CSVs
- local metric outputs
- summary tables
- exported generated text files
- figure exports for appendix and presentation use

Why it exists:
- keeps derived results separate from source code

Why it is useful:
- this folder is the evidence layer of the thesis

Important subgroups:
- `pilot_12text_experiments.csv`
- `pilot_ranking_local/`
- `human_baseline_summaries/`
- `appendix_tables/`
- `appendix_figures/`
- `presentation_figures/`
- `pilot_text_exports/`

## 8. Notebook Files

### `notebooks/llm_results_visualization.ipynb`

What it does:
- visualizes LLM prompt-condition results
- creates appendix-ready tables and figures

Why it exists:
- the thesis needs visual reporting, not only CSVs

Why it is useful:
- helps compare prompt conditions and top model outputs visually

### `notebooks/results_analysis.ipynb`

What it does:
- compares archived CTAP benchmark results with the current local human benchmark

Why it exists:
- validates that the shift from CTAP to a local pipeline did not destroy the general benchmark signal

Why it is useful:
- helpful for methodological justification

### `notebooks/mrc_exploration.ipynb`

What it does:
- exploratory notebook for understanding the MRC data resource

Why it exists:
- useful during development and debugging

Why it is useful:
- helps inspect the lexical resource behind psycholinguistic scores

### `notebooks/onestop_english_exploration.ipynb`

What it does:
- exploratory notebook for the OneStopEnglish corpus

Why it exists:
- supports early data understanding

Why it is useful:
- helpful for inspecting corpus structure and content

### `notebooks/README.md`

What it does:
- brief explanation that notebooks are exploratory only

Why it exists:
- clarifies that the notebooks are not part of the production pipeline

## 9. Documentation Files

### `README.md`

What it does:
- top-level usage and structure guide

Why it exists:
- gives a practical entrypoint to the repository

Why it is useful:
- first place to look when reusing the code

### `docs/checkpoint_explainer.md`

What it does:
- detailed conceptual explanation of the thesis and pipeline

Why it exists:
- helps you understand the project beyond just running scripts

Why it is useful:
- ideal for supervisor prep and self-study

### `docs/checkpoint_speaker_notes.md`

What it does:
- 10-minute speaking script for the supervisor checkpoint

Why it exists:
- presentation support

Why it is useful:
- helps structure how you explain the project verbally

## 10. Environment and Configuration Files

### `.env.example`

What it does:
- example environment file for API-related configuration

Why it exists:
- shows which variables need to be set locally

### `requirements.txt`

What it does:
- lists the Python dependencies

Why it exists:
- environment reproducibility

### `.gitignore`

What it does:
- excludes local environment and transient files from version control

Why it exists:
- keeps the repository cleaner

## 11. Files That Are Not Really Project Logic

### `venv/`

What it is:
- local Python environment

Why it exists:
- local execution support

Why it is not conceptually part of the codebase:
- it is environment state, not thesis logic

### `__pycache__/`

What it is:
- Python bytecode cache

Why it exists:
- runtime performance

Why it is not conceptually part of the codebase:
- generated automatically

## 12. Most Important Files To Understand First

If you want the shortest path to understanding the project, read these in order:

1. `README.md`
2. `docs/checkpoint_explainer.md`
3. `run_onestop_english.py`
4. `analyze_thesis_metrics.py`
5. `rank_prompt_configs.py`
6. `lgp_pipeline/tier2.py`
7. `lgp_pipeline/psycholinguistics.py`
8. `llm_api/prompts.py`

That order gives you:
- the big picture
- the generation flow
- the analysis flow
- the two most thesis-specific metric components

## 13. What The Codebase Is Really Optimized For

The repository is optimized for:
- transparent evaluation
- prompt comparison
- human versus LLM comparison
- thesis reporting

It is not optimized for:
- production deployment
- large-scale web application use
- advanced learned evaluation
- strict factuality checking

This is not a flaw. It reflects the actual thesis scope.

## 14. Final Practical Advice

For future work, treat these as the canonical active files:
- `run_onestop_english.py`
- `analyze_thesis_metrics.py`
- `rank_prompt_configs.py`
- `summarize_thesis_results.py`
- `setup_onestop_english.py`

Treat these as archival or compatibility files:
- `methodology_analysis.py`
- `rank_prompt_experiments.py`
- `summarize_results.py`
- `FKandMTLD.py`
- `setup_onestopenglish.py`
- `data/advResults`
- `data/elemResults`

That distinction will make the repository much easier to maintain and explain.
