from __future__ import annotations

"""Plain-language evaluation materials for the teacher dashboard study."""

ESTIMATED_TIME = "about 12 to 15 minutes"

TASKS = [
    {
        "id": "T1",
        "title": "Read the original passage",
        "instruction": (
            "Read the original passage, which is on an advaced level, so you understand the topic and the main ideas."
        ),
    },
    {
        "id": "T2",
        "title": "Compare the simplified versions",
        "instruction": (
            "Read the three simplified versions and decide which one you would be most likely to use with elementary level students."
        ),
    },
    {
        "id": "T3",
        "title": "Explain your choice",
        "instruction": (
            "Briefly explain why you preferred that version and note any words or phrases that felt odd or unclear for that specific level."
        ),
    },
    {
        "id": "T4",
        "title": "Complete the final questionnaire",
        "instruction": (
            "After the three passages, answer a short questionnaire about ease of use and how easy it was to make your choices."
        ),
    },
]

QUESTIONNAIRE_ITEMS = [
    {
        "id": "Q1",
        "prompt": "The instructions were easy to follow.",
    },
    {
        "id": "Q2",
        "prompt": "It was easy to compare the rewritten versions.",
    },
    {
        "id": "Q3",
        "prompt": "It was easy to explain why I preferred one version over the others.",
    },
    {
        "id": "Q4",
        "prompt": "It was easy to point out words or phrases that felt odd or unclear.",
    },
    {
        "id": "Q5",
        "prompt": "Overall, the website was easy to use.",
    },
    {
        "id": "Q6",
        "prompt": "I could imagine using a tool like this when adapting texts for students.",
    },
]

OPEN_QUESTIONS = [
    {
        "id": "O1",
        "prompt": "What helped you decide which simplified version was best?",
    },
    {
        "id": "O2",
        "prompt": "Was there anything confusing or difficult about the process?",
    },
    {
        "id": "O3",
        "prompt": "Is there anything you would change or add before using a tool like this again?",
    },
]

LIKERT_OPTIONS = {
    "1": "Strongly disagree",
    "2": "Disagree",
    "3": "Not sure",
    "4": "Agree",
    "5": "Strongly agree",
}
