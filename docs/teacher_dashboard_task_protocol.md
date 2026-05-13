# Teacher Dashboard Task Protocol

Estimated time: `about 12 to 15 minutes`

## Study Flow

1. Read the original passage.
2. Compare three rewritten versions shown as `Version A`, `Version B`, and `Version C`.
3. Choose the version you would be most likely to use with students.
4. Explain why you made that choice.
5. If needed, type any word or phrase that felt odd or unclear and explain why.
6. Move to the next passage and repeat the same process.
7. Complete the final questionnaire.

## Passage Selection

The dashboard does not show random passages. It selects three passages automatically because they show large differences between the human simplified text and the AI outputs on vocabulary variety (`MTLD`), with an additional preference for passages that also differ on the academic-vocabulary proxy. This makes the teacher review more likely to reveal meaningful qualitative differences.

## What Teachers See

Teachers see:

- the original passage,
- three rewritten versions,
- a place to choose a preferred version,
- a free-text field for explaining the choice,
- three optional word-or-phrase comment slots,
- a short final questionnaire.

Teachers do not see:

- technical metric names,
- model names,
- graphs,
- raw numerical scores,
- which rewritten version is human-written or AI-generated.

## Data Saved

For each passage, the dashboard saves:

- the chosen anonymous version label,
- the hidden underlying source (`human`, `few-shot`, or `zero-shot`),
- the teacher’s explanation,
- any flagged words or phrases,
- the accompanying comments.

The final questionnaire is saved separately, and each session also has a JSON snapshot for later analysis.
