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


## 🚀 Getting Started
1. Clone the repository: `git clone
