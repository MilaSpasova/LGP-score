from __future__ import annotations

"""Simple teacher-facing dashboard for the thesis review study."""

import hashlib
import json
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
GSPREAD_IMPORT_ERROR = ""


def load_gspread_module():
    try:
        import gspread as gspread_module

        return gspread_module, ""
    except ModuleNotFoundError:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "gspread>=6.0.0",
                    "google-auth>=2.29.0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import gspread as gspread_module

            return gspread_module, ""
        except Exception:  # pragma: no cover - deployment fallback
            import traceback

            return None, traceback.format_exc()
    except Exception:  # pragma: no cover - surface unexpected import failures
        import traceback

        return None, traceback.format_exc()


gspread, GSPREAD_IMPORT_ERROR = load_gspread_module()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dashboard.evaluation_materials import (
        ESTIMATED_TIME,
        LIKERT_OPTIONS,
        OPEN_QUESTIONS,
        QUESTIONNAIRE_ITEMS,
        TASKS,
    )
except ModuleNotFoundError:
    from evaluation_materials import (  # type: ignore[no-redef]
        ESTIMATED_TIME,
        LIKERT_OPTIONS,
        OPEN_QUESTIONS,
        QUESTIONNAIRE_ITEMS,
        TASKS,
    )


OUTPUTS = ROOT / "outputs"
FEEDBACK_DIR = OUTPUTS / "dashboard_feedback"
STUDY_DATA_PATH = ROOT / "dashboard" / "study_data.json"

DEFAULT_FILES = {
    "human_text": OUTPUTS / "human_baseline_text_metrics.csv",
    "human_pairwise": OUTPUTS / "human_baseline_pairwise_comparisons.csv",
    "few_text": OUTPUTS / "full_batch_fewshot05_text_metrics.csv",
    "few_pairwise": OUTPUTS / "full_batch_fewshot05_pairwise.csv",
    "zero_text": OUTPUTS / "full_batch_zeroshot00_text_metrics.csv",
    "zero_pairwise": OUTPUTS / "full_batch_zeroshot00_pairwise.csv",
}

ANON_LABELS = ["Version A", "Version B", "Version C"]
PAGE_OPTIONS = ["Home", "Text Review", "Questionnaire"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_csv(DEFAULT_FILES["human_text"]),
        read_csv(DEFAULT_FILES["human_pairwise"]),
        read_csv(DEFAULT_FILES["few_text"]),
        read_csv(DEFAULT_FILES["few_pairwise"]),
        read_csv(DEFAULT_FILES["zero_text"]),
        read_csv(DEFAULT_FILES["zero_pairwise"]),
    )


@st.cache_data(show_spinner=False)
def load_bundled_study_items() -> list[dict[str, object]]:
    payload = read_json(STUDY_DATA_PATH)
    study_items = payload.get("study_items", [])
    return study_items if isinstance(study_items, list) else []


def pretty_story_title(story_key: str) -> str:
    return story_key.replace("wnl ", "").replace("-", " ").replace("_", " ").title()


def deterministic_order(story_key: str) -> list[str]:
    variants = ["human", "few", "zero"]
    digest = hashlib.md5(story_key.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    permutations = [
        ["human", "few", "zero"],
        ["human", "zero", "few"],
        ["few", "human", "zero"],
        ["few", "zero", "human"],
        ["zero", "human", "few"],
        ["zero", "few", "human"],
    ]
    return permutations[seed % len(permutations)]


def select_study_story_keys(
    human_pairwise: pd.DataFrame,
    few_pairwise: pd.DataFrame,
    zero_pairwise: pd.DataFrame,
    *,
    top_n: int = 3,
) -> list[str]:
    human = human_pairwise[human_pairwise["variant"].eq("human::elementary")].copy()
    few = few_pairwise[few_pairwise["variant"].astype(str).str.contains("few_shot", na=False)].copy()
    zero = zero_pairwise[zero_pairwise["variant"].astype(str).str.contains("zero_shot", na=False)].copy()

    merged = human[["story_key", "delta_mtld", "delta_tier2_proxy_token_ratio"]].rename(
        columns={
            "delta_mtld": "human_mtld",
            "delta_tier2_proxy_token_ratio": "human_tier2",
        }
    )
    merged = merged.merge(
        few[["story_key", "delta_mtld", "delta_tier2_proxy_token_ratio"]].rename(
            columns={
                "delta_mtld": "few_mtld",
                "delta_tier2_proxy_token_ratio": "few_tier2",
            }
        ),
        on="story_key",
        how="inner",
    )
    merged = merged.merge(
        zero[["story_key", "delta_mtld", "delta_tier2_proxy_token_ratio"]].rename(
            columns={
                "delta_mtld": "zero_mtld",
                "delta_tier2_proxy_token_ratio": "zero_tier2",
            }
        ),
        on="story_key",
        how="inner",
    )

    merged["few_mtld_gap"] = (merged["few_mtld"] - merged["human_mtld"]).abs()
    merged["zero_mtld_gap"] = (merged["zero_mtld"] - merged["human_mtld"]).abs()
    merged["few_tier2_gap"] = (merged["few_tier2"] - merged["human_tier2"]).abs()
    merged["zero_tier2_gap"] = (merged["zero_tier2"] - merged["human_tier2"]).abs()

    # MTLD is the primary selection target; Tier 2 differences add a smaller bonus.
    merged["selection_score"] = (
        merged[["few_mtld_gap", "zero_mtld_gap"]].mean(axis=1)
        + 50 * merged[["few_tier2_gap", "zero_tier2_gap"]].mean(axis=1)
    )
    merged = merged.sort_values("selection_score", ascending=False)
    return merged["story_key"].head(top_n).tolist()


def fetch_text(df: pd.DataFrame, *, story_key: str, variant: str) -> str:
    row = df[(df["story_key"] == story_key) & (df["variant"] == variant)]
    if row.empty:
        return ""
    return str(row.iloc[0]["text"])


@st.cache_data(show_spinner=False)
def build_study_items() -> list[dict[str, object]]:
    bundled = load_bundled_study_items()
    if bundled:
        return bundled

    human_text, human_pairwise, few_text, few_pairwise, zero_text, zero_pairwise = load_data()
    story_keys = select_study_story_keys(human_pairwise, few_pairwise, zero_pairwise, top_n=3)

    items: list[dict[str, object]] = []
    for story_key in story_keys:
        order = deterministic_order(story_key)
        version_pool = {
            "human": {
                "source": "human_simplified",
                "text": fetch_text(human_text, story_key=story_key, variant="human::elementary"),
            },
            "few": {
                "source": "few_shot_0.5",
                "text": fetch_text(
                    few_text,
                    story_key=story_key,
                    variant="openai::openai/gpt-5.2::few_shot::temp=0.5",
                ),
            },
            "zero": {
                "source": "zero_shot_0.0",
                "text": fetch_text(
                    zero_text,
                    story_key=story_key,
                    variant="openai::openai/gpt-5.2::zero_shot::temp=0.0",
                ),
            },
        }

        versions: list[dict[str, str]] = []
        for label, source_key in zip(ANON_LABELS, order, strict=True):
            versions.append(
                {
                    "label": label,
                    "source": str(version_pool[source_key]["source"]),
                    "text": str(version_pool[source_key]["text"]),
                }
            )

        items.append(
            {
                "story_key": story_key,
                "title": pretty_story_title(story_key),
                "original_text": fetch_text(human_text, story_key=story_key, variant="human::advanced"),
                "versions": versions,
            }
        )
    return items


def ensure_state(study_items: list[dict[str, object]]) -> None:
    if "page" in st.session_state:
        del st.session_state["page"]
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "scroll_target" not in st.session_state:
        st.session_state.scroll_target = ""
    if "current_story_index" not in st.session_state:
        st.session_state.current_story_index = 0
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = uuid.uuid4().hex[:8]
    if "submission_id" not in st.session_state:
        st.session_state.submission_id = uuid.uuid4().hex
    if "study_responses" not in st.session_state:
        responses: dict[str, dict[str, object]] = {}
        for item in study_items:
            story_key = str(item["story_key"])
            responses[story_key] = {
                "preferred_label": "",
                "reason": "",
                "flags": [
                    {"phrase": "", "comment": ""},
                    {"phrase": "", "comment": ""},
                    {"phrase": "", "comment": ""},
                ],
            }
        st.session_state.study_responses = responses
    if "questionnaire_saved" not in st.session_state:
        st.session_state.questionnaire_saved = False
    if "questionnaire_submitting" not in st.session_state:
        st.session_state.questionnaire_submitting = False
    if "access_granted" not in st.session_state:
        st.session_state.access_granted = False
    if "submission_error_messages" not in st.session_state:
        st.session_state.submission_error_messages = []
    if "questionnaire_draft" not in st.session_state:
        st.session_state.questionnaire_draft = {}


def get_access_codes() -> list[str]:
    try:
        single = st.secrets.get("dashboard_access_code", "")
        multiple = st.secrets.get("dashboard_access_codes", [])
    except Exception:
        return []

    codes: list[str] = []
    if isinstance(single, str) and single.strip():
        codes.append(single.strip())
    if isinstance(multiple, (list, tuple)):
        codes.extend(str(item).strip() for item in multiple if str(item).strip())
    return list(dict.fromkeys(codes))


def require_access_code() -> bool:
    codes = get_access_codes()
    if not codes:
        st.session_state.access_granted = True
        return True
    if st.session_state.get("access_granted"):
        return True

    st.title("Teacher Text Review")
    st.markdown("Please enter the access code you received from the researcher.")
    with st.form("access_code_form"):
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
        if submitted:
            if code.strip() in codes:
                st.session_state.access_granted = True
                st.rerun()
            else:
                st.error("That code is not correct. Please try again.")
    return False


def get_google_sheet_id() -> str:
    try:
        return str(st.secrets.get("google_sheet_id", "")).strip()
    except Exception:
        return ""


def get_google_service_account_info() -> dict[str, object] | None:
    try:
        raw = st.secrets.get("gcp_service_account")
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, Mapping):
        return dict(raw)
    return None


def get_gspread_client() -> gspread.Client | None:
    if gspread is None:
        return None
    info = get_google_service_account_info()
    if not info:
        return None
    try:
        return gspread.service_account_from_dict(info)
    except Exception:
        return None


def append_dataframe_to_google_sheet(sheet_name: str, frame: pd.DataFrame) -> tuple[bool, str]:
    if gspread is None:
        detail = GSPREAD_IMPORT_ERROR.strip().splitlines()[-1] if GSPREAD_IMPORT_ERROR.strip() else "gspread_not_installed"
        return False, f"gspread_not_installed: {detail}"
    sheet_id = get_google_sheet_id()
    client = get_gspread_client()
    if not sheet_id or client is None or frame.empty:
        if not sheet_id:
            return False, "missing_google_sheet_id"
        if client is None:
            return False, "google_client_init_failed"
        return False, "empty_frame"
    try:
        workbook = client.open_by_key(sheet_id)
        try:
            worksheet = workbook.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            worksheet = workbook.add_worksheet(
                title=sheet_name,
                rows=max(len(frame) + 10, 100),
                cols=max(len(frame.columns) + 2, 20),
            )
            worksheet.append_row(frame.columns.tolist(), value_input_option="USER_ENTERED")

        if worksheet.row_count == 0:
            worksheet.append_row(frame.columns.tolist(), value_input_option="USER_ENTERED")

        existing_header = worksheet.row_values(1)
        if not existing_header:
            worksheet.append_row(frame.columns.tolist(), value_input_option="USER_ENTERED")
            existing_header = worksheet.row_values(1)

        combined_header = list(existing_header)
        for column in frame.columns.tolist():
            if column not in combined_header:
                combined_header.append(column)
        if combined_header != existing_header:
            if worksheet.col_count < len(combined_header):
                worksheet.add_cols(len(combined_header) - worksheet.col_count)
            worksheet.update("A1", [combined_header])
            existing_header = combined_header

        aligned = frame.reindex(columns=existing_header, fill_value="")

        if "submission_id" in existing_header:
            submission_col = existing_header.index("submission_id") + 1
            existing_ids = worksheet.col_values(submission_col)[1:]
            frame_ids = set(aligned["submission_id"].astype(str).tolist())
            if frame_ids and frame_ids.issubset(set(existing_ids)):
                return True, "duplicate_skipped"

        rows = aligned.fillna("").astype(str).values.tolist()
        if rows:
            worksheet.append_rows(rows, value_input_option="USER_ENTERED")
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def save_story_response(story_key: str) -> None:
    responses = st.session_state.study_responses
    responses[story_key] = {
        "preferred_label": st.session_state.get(f"preferred_{story_key}", ""),
        "reason": st.session_state.get(f"reason_{story_key}", ""),
        "flags": [
            {
                "phrase": st.session_state.get(f"flag_phrase_{story_key}_{idx}", ""),
                "comment": st.session_state.get(f"flag_comment_{story_key}_{idx}", ""),
            }
            for idx in range(3)
        ],
    }
    st.session_state.study_responses = responses


def save_questionnaire_draft() -> None:
    draft = {
        "q_name": st.session_state.get("q_name", ""),
        "q_context": st.session_state.get("q_context", ""),
    }
    for item in QUESTIONNAIRE_ITEMS:
        draft[f"questionnaire_{item['id']}"] = st.session_state.get(f"questionnaire_{item['id']}", "")
    for item in OPEN_QUESTIONS:
        draft[f"questionnaire_{item['id']}"] = st.session_state.get(f"questionnaire_{item['id']}", "")
    st.session_state.questionnaire_draft = draft


def hydrate_questionnaire_fields() -> None:
    draft = st.session_state.get("questionnaire_draft", {})
    for key, value in draft.items():
        if key not in st.session_state or (not st.session_state.get(key) and value):
            st.session_state[key] = value


def validate_story_responses(study_items: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    for item in study_items:
        story_key = str(item["story_key"])
        response = st.session_state.study_responses.get(story_key, {})
        preferred = str(response.get("preferred_label", "")).strip()
        reason = str(response.get("reason", "")).strip()
        title = str(item["title"])
        if not preferred:
            errors.append(f"Choose a preferred rewritten version for '{title}'.")
        if not reason:
            errors.append(f"Explain why you chose that version for '{title}'.")
    return errors


def validate_questionnaire() -> list[str]:
    errors: list[str] = []
    for item in QUESTIONNAIRE_ITEMS:
        if not str(st.session_state.get(f"questionnaire_{item['id']}", "")).strip():
            errors.append(f"Answer question {item['id']}.")
    for item in OPEN_QUESTIONS:
        if not str(st.session_state.get(f"questionnaire_{item['id']}", "")).strip():
            errors.append(f"Answer question {item['id']}.")
    return errors


def save_all_feedback(study_items: list[dict[str, object]]) -> Path:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    participant_id = st.session_state.participant_id
    submission_id = st.session_state.submission_id

    story_rows: list[dict[str, object]] = []
    for item in study_items:
        story_key = str(item["story_key"])
        response = st.session_state.study_responses.get(story_key, {})
        version_map = {
            version["label"]: version["source"]
            for version in item["versions"]  # type: ignore[index]
        }
        row = {
            "timestamp": timestamp,
            "participant_id": participant_id,
            "submission_id": submission_id,
            "story_key": story_key,
            "story_title": item["title"],
            "preferred_label": response.get("preferred_label", ""),
            "preferred_source": version_map.get(response.get("preferred_label", ""), ""),
            "reason": response.get("reason", ""),
        }
        flags = response.get("flags", [])
        for idx, flag in enumerate(flags, start=1):
            row[f"flag_phrase_{idx}"] = flag.get("phrase", "")
            row[f"flag_comment_{idx}"] = flag.get("comment", "")
        story_rows.append(row)

    questionnaire_row = {
        "timestamp": timestamp,
        "participant_id": participant_id,
        "submission_id": submission_id,
        "name_or_initials": st.session_state.get("q_name", ""),
        "teaching_context": st.session_state.get("q_context", ""),
    }
    for item in QUESTIONNAIRE_ITEMS:
        questionnaire_row[item["id"]] = st.session_state.get(f"questionnaire_{item['id']}", "")
    for item in OPEN_QUESTIONS:
        questionnaire_row[item["id"]] = st.session_state.get(f"questionnaire_{item['id']}", "")

    story_path = FEEDBACK_DIR / "teacher_story_reviews.csv"
    questionnaire_path = FEEDBACK_DIR / "teacher_questionnaire.csv"
    session_path = FEEDBACK_DIR / f"teacher_session_{timestamp}_{participant_id}.json"

    story_df = pd.DataFrame(story_rows)
    questionnaire_df = pd.DataFrame([questionnaire_row])

    duplicate_local = False
    if questionnaire_path.is_file():
        try:
            existing_questionnaire = pd.read_csv(questionnaire_path, usecols=["submission_id"])
            duplicate_local = submission_id in existing_questionnaire["submission_id"].astype(str).tolist()
        except Exception:
            duplicate_local = False

    story_sheet_ok, story_sheet_status = append_dataframe_to_google_sheet("teacher_story_reviews", story_df)
    questionnaire_sheet_ok, questionnaire_sheet_status = append_dataframe_to_google_sheet("teacher_questionnaire", questionnaire_df)

    if not duplicate_local:
        if story_path.is_file():
            story_df.to_csv(story_path, mode="a", index=False, header=False)
        else:
            story_df.to_csv(story_path, index=False)

        if questionnaire_path.is_file():
            questionnaire_df.to_csv(questionnaire_path, mode="a", index=False, header=False)
        else:
            questionnaire_df.to_csv(questionnaire_path, index=False)

    session_payload = {
        "timestamp": timestamp,
        "participant_id": participant_id,
        "submission_id": submission_id,
        "study_items": study_items,
        "responses": st.session_state.study_responses,
        "questionnaire": questionnaire_row,
        "google_sheets": {
            "story_reviews_saved": story_sheet_ok,
            "story_reviews_status": story_sheet_status,
            "questionnaire_saved": questionnaire_sheet_ok,
            "questionnaire_status": questionnaire_sheet_status,
            "sheet_id_present": bool(get_google_sheet_id()),
        },
        "local_storage": {
            "duplicate_skipped": duplicate_local,
        },
    }
    session_path.write_text(json.dumps(session_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    st.session_state.google_sheets_status = session_payload["google_sheets"]
    return session_path


def apply_branding() -> None:
    st.set_page_config(page_title="Teacher Text Review", page_icon="📝", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px;}
        .task-card, .text-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            background: #ffffff;
        }
        .text-card {min-height: 430px;}
        .soft-note {
            padding: 0.9rem 1rem;
            border-radius: 10px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #7c2d12;
        }
        .small-muted {color: #6b7280; font-size: 0.95rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_scroll_anchor(anchor_id: str = "review-top") -> None:
    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)
    if st.session_state.get("scroll_target") == anchor_id:
        components.html(
            f"""
            <script>
            const anchorId = {json.dumps(anchor_id)};
            const scrollParentToAnchor = () => {{
              try {{
                const parentDoc = window.parent.document;
                const anchor = parentDoc.getElementById(anchorId);
                if (anchor) {{
                  anchor.scrollIntoView({{ behavior: 'auto', block: 'start' }});
                }}
                window.parent.location.hash = anchorId;
              }} catch (e) {{}}
            }};
            scrollParentToAnchor();
            setTimeout(scrollParentToAnchor, 50);
            setTimeout(scrollParentToAnchor, 150);
            setTimeout(scrollParentToAnchor, 300);
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_target = ""


def render_home(study_items: list[dict[str, object]]) -> None:
    st.title("Teacher Text Review")
    st.markdown(
        f"<div class='soft-note'><strong>Estimated time:</strong> {ESTIMATED_TIME}.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("### What you will do")

    for idx, task in enumerate(TASKS, start=1):
        st.markdown(
            (
                "<div class='task-card'>"
                f"<strong>Step {idx}. {task['title']}</strong><br>"
                f"<span class='small-muted'>{task['instruction']}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        f"You will review **{len(study_items)} passages**. For each one, you will read the original passage, compare **three anonymous rewritten versions**, choose the one you prefer, and explain why."
    )
    if st.button("Start the review", type="primary", width="stretch"):
        st.session_state.current_page = "Text Review"
        st.rerun()


def render_text_column(title: str, text: str) -> None:
    st.markdown(
        (
            "<div class='text-card'>"
            f"<h4 style='margin-top:0;'>{title}</h4>"
            f"<div style='line-height:1.7; white-space:pre-wrap;'>{text}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def hydrate_story_fields(story_key: str) -> None:
    saved = st.session_state.study_responses.get(story_key, {})
    if f"preferred_{story_key}" not in st.session_state or (
        not st.session_state.get(f"preferred_{story_key}") and saved.get("preferred_label")
    ):
        st.session_state[f"preferred_{story_key}"] = saved.get("preferred_label", "")
    if f"reason_{story_key}" not in st.session_state or (
        not st.session_state.get(f"reason_{story_key}") and saved.get("reason")
    ):
        st.session_state[f"reason_{story_key}"] = saved.get("reason", "")
    for idx, flag in enumerate(saved.get("flags", [])):
        phrase_key = f"flag_phrase_{story_key}_{idx}"
        comment_key = f"flag_comment_{story_key}_{idx}"
        if phrase_key not in st.session_state or (
            not st.session_state.get(phrase_key) and flag.get("phrase")
        ):
            st.session_state[phrase_key] = flag.get("phrase", "")
        if comment_key not in st.session_state or (
            not st.session_state.get(comment_key) and flag.get("comment")
        ):
            st.session_state[comment_key] = flag.get("comment", "")


def render_text_review(study_items: list[dict[str, object]]) -> None:
    render_scroll_anchor("review-top")
    idx = st.session_state.current_story_index
    item = study_items[idx]
    story_key = str(item["story_key"])
    hydrate_story_fields(story_key)

    st.title("Text Review")
    st.markdown(
        (
            "<div class='soft-note'>"
            f"<strong>Now reviewing passage {idx + 1} of {len(study_items)}:</strong> "
            f"{item['title']}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader(str(item["title"]))

    def render_nav_buttons(position: str) -> None:
        nav_left, nav_mid, nav_right = st.columns([1, 1, 1])
        with nav_left:
            if st.button(
                "Previous",
                disabled=idx == 0,
                width="stretch",
                key=f"prev_{position}_{story_key}",
                ):
                    save_story_response(story_key)
                    st.session_state.current_story_index = max(0, idx - 1)
                    st.session_state.scroll_target = "review-top"
                    st.rerun()
        with nav_mid:
            if idx < len(study_items) - 1:
                if st.button(
                    "Next",
                    type="primary",
                    width="stretch",
                    key=f"next_{position}_{story_key}",
                ):
                    save_story_response(story_key)
                    st.session_state.current_story_index = min(len(study_items) - 1, idx + 1)
                    st.session_state.scroll_target = "review-top"
                    st.rerun()
            else:
                if st.button(
                    "Go to questionnaire",
                    type="primary",
                    width="stretch",
                    key=f"questionnaire_{position}_{story_key}",
                ):
                    save_story_response(story_key)
                    st.session_state.current_page = "Questionnaire"
                    st.session_state.scroll_target = "questionnaire-top"
                    st.rerun()
        with nav_right:
            st.markdown("&nbsp;", unsafe_allow_html=True)

    render_nav_buttons("top")
    st.write("")

    version_labels = [version["label"] for version in item["versions"]]  # type: ignore[index]
    selected_version_label = st.radio(
        "Choose which rewritten version to read on the right.",
        version_labels,
        key=f"view_version_{story_key}",
        horizontal=True,
    )
    selected_version = next(
        version for version in item["versions"] if version["label"] == selected_version_label  # type: ignore[index]
    )

    col1, col2 = st.columns(2)
    with col1:
        render_text_column("Original passage", str(item["original_text"]))
    with col2:
        render_text_column(selected_version_label, str(selected_version["text"]))

    st.write("")
    st.markdown("### Your choice")
    st.caption("* Required")
    st.radio(
        "Which simplified version would you be most likely to use with students? *",
        ["", *version_labels],
        key=f"preferred_{story_key}",
        format_func=lambda value: "Select one option" if value == "" else value,
        on_change=partial(save_story_response, story_key),
    )
    st.text_area(
        "Why did you choose that version? *",
        key=f"reason_{story_key}",
        height=120,
        placeholder="You can mention things like clarity, word choice, tone, or what felt easiest to understand.",
        on_change=partial(save_story_response, story_key),
    )

    st.markdown("### Words or phrases you would still change")
    st.caption("If anything sounded odd or unclear, type the word or phrase and explain why. You can leave these blank if nothing stood out.")
    for flag_idx in range(3):
        left, right = st.columns([1, 2])
        with left:
            st.text_input(
                f"Word or phrase {flag_idx + 1}",
                key=f"flag_phrase_{story_key}_{flag_idx}",
                placeholder="Example: public interest",
                on_change=partial(save_story_response, story_key),
            )
        with right:
            st.text_input(
                f"Why does it stand out? {flag_idx + 1}",
                key=f"flag_comment_{story_key}_{flag_idx}",
                placeholder="Example: This phrase still feels too formal.",
                on_change=partial(save_story_response, story_key),
            )

    render_nav_buttons("bottom")


def render_questionnaire(study_items: list[dict[str, object]]) -> None:
    render_scroll_anchor("questionnaire-top")
    hydrate_questionnaire_fields()

    if st.session_state.get("questionnaire_submitting") and not st.session_state.get("questionnaire_saved"):
        save_path = save_all_feedback(study_items)
        st.session_state.questionnaire_saved = True
        st.session_state.questionnaire_submitting = False
        st.session_state.submission_error_messages = []
        st.session_state.saved_path = str(save_path)
        st.rerun()

    st.title("Final Questionnaire")
    st.markdown(
        "This last step asks about how easy the activity was and how comfortable you felt making your choices."
    )
    st.caption("* Required")

    st.text_input("Name or initials (optional)", key="q_name", on_change=save_questionnaire_draft)
    st.text_input("Teaching context or subject area (optional)", key="q_context", on_change=save_questionnaire_draft)

    st.markdown("### Quick questions")
    for item in QUESTIONNAIRE_ITEMS:
        st.radio(
            f"{item['prompt']} *",
            list(LIKERT_OPTIONS.keys()),
            key=f"questionnaire_{item['id']}",
            format_func=lambda value: LIKERT_OPTIONS[value],
            on_change=save_questionnaire_draft,
        )

    st.markdown("### Final comments")
    for item in OPEN_QUESTIONS:
        st.text_area(
            f"{item['prompt']} *",
            key=f"questionnaire_{item['id']}",
            height=110,
            on_change=save_questionnaire_draft,
        )

    if st.session_state.get("submission_error_messages"):
        for message in st.session_state["submission_error_messages"]:
            st.error(message)

    left, right = st.columns([1, 1])
    with left:
        if st.button(
            "Back to the texts",
            width="stretch",
            disabled=st.session_state.get("questionnaire_saved", False) or st.session_state.get("questionnaire_submitting", False),
        ):
            save_questionnaire_draft()
            st.session_state.current_page = "Text Review"
            st.session_state.scroll_target = "review-top"
            st.rerun()
    with right:
        if st.button(
            "Finish",
            type="primary",
            width="stretch",
            disabled=st.session_state.get("questionnaire_saved", False) or st.session_state.get("questionnaire_submitting", False),
        ):
            save_questionnaire_draft()
            story_errors = validate_story_responses(study_items)
            questionnaire_errors = validate_questionnaire()
            all_errors = story_errors + questionnaire_errors
            if all_errors:
                st.session_state.submission_error_messages = all_errors
                st.rerun()
            st.session_state.questionnaire_submitting = True
            st.session_state.submission_error_messages = []
            st.rerun()

    if st.session_state.get("questionnaire_saved"):
        st.success("Thank you. Your responses have been saved.")
        st.info("This session has already been submitted. Further submissions are disabled.")
        google_status = st.session_state.get("google_sheets_status", {})
        if google_status:
            if google_status.get("story_reviews_saved") and google_status.get("questionnaire_saved"):
                st.success("Google Sheets save: success.")
            else:
                st.warning(
                    "Google Sheets save did not fully succeed. "
                    f"Story reviews: {google_status.get('story_reviews_status', 'unknown')}. "
                    f"Questionnaire: {google_status.get('questionnaire_status', 'unknown')}."
                )
    elif st.session_state.get("questionnaire_submitting"):
        st.info("Submitting your responses. Please wait...")


def main() -> None:
    apply_branding()
    study_items = build_study_items()
    ensure_state(study_items)
    if not require_access_code():
        return

    with st.sidebar:
        st.header("Review Study")
        st.caption("Follow the steps in order.")
        current = st.session_state.current_page
        st.markdown(f"{'➡️' if current == 'Home' else '1.'} Home")
        st.markdown(f"{'➡️' if current == 'Text Review' else '2.'} Text Review")
        st.markdown(f"{'➡️' if current == 'Questionnaire' else '3.'} Questionnaire")
        st.caption(f"Time needed: {ESTIMATED_TIME}")

    page = st.session_state.current_page
    if page == "Home":
        render_home(study_items)
    elif page == "Text Review":
        render_text_review(study_items)
    else:
        render_questionnaire(study_items)


if __name__ == "__main__":
    main()
