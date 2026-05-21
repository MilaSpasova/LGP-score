# Teacher Dashboard

This folder contains the simplified teacher-facing dashboard used for `Experiment D`.

## Purpose

The current version is designed for invited teachers with no technical setup knowledge. It focuses on one simple task:

- read an original passage,
- compare three anonymous rewritten versions,
- choose the version they prefer,
- explain why,
- flag any words or phrases that feel odd,
- complete a short final questionnaire.

The three study passages are chosen automatically because they show especially large differences between the human simplified text and the AI outputs on `MTLD`, with an additional bonus for differences in the academic-vocabulary proxy.

## Main Pages

- `Home`
- `Text Review`
- `Questionnaire`

## Run Locally

```powershell
venv\Scripts\streamlit run dashboard/streamlit_app.py
```

## Access Code

The app can require a shared access code. For local testing, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`.

Example:

```toml
dashboard_access_code = "my-teacher-code"
```

On Streamlit Community Cloud, paste the same secret into the app's **Secrets** settings.

## Google Sheets Storage

The app can also append responses to Google Sheets if you provide:

- `google_sheet_id`
- a `gcp_service_account` block in Streamlit secrets

The app writes to two worksheet tabs:

- `teacher_story_reviews`
- `teacher_questionnaire`

If the tabs do not exist yet, the app creates them automatically.

## Input Files

The app first looks for the bundled file:

- `dashboard/study_data.json`

If that file is not present, it rebuilds the study set from the metric CSV files in `outputs/`. This keeps the cloud deployment simple while still allowing local regeneration from the thesis outputs.

## Feedback Storage

Teacher responses are written to:

- `outputs/dashboard_feedback/teacher_story_reviews.csv`
- `outputs/dashboard_feedback/teacher_questionnaire.csv`

A full JSON snapshot of each teacher session is also saved in the same folder.

## Important Deployment Note

For local use, writing feedback to files is fine. On Streamlit Community Cloud, files created while the app is running are **not guaranteed to persist across user sessions**. This means the local CSV/JSON files should be treated as a fallback only. For a real teacher study, use Google Sheets or another external store.

## Supporting Materials

- [teacher_dashboard_task_protocol.md](D:/Year-3-Uni/thesis/Code/LGP-score/docs/teacher_dashboard_task_protocol.md:1)
- [teacher_dashboard_questionnaire.md](D:/Year-3-Uni/thesis/Code/LGP-score/docs/teacher_dashboard_questionnaire.md:1)
