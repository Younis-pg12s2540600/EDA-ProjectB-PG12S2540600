
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""

st.set_page_config(
    page_title="EDA Mini Project B Starter",
    page_icon="📈",
    layout="wide",
)

st.title("EDA Mini Project B — Time-Series Forecasting Starter")
st.caption("Starter app stops after data audit, time-series setup, baseline feature table, exports, and AI grader.")

# -----------------------------
# Helpers
# -----------------------------
def safe_json_dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def get_openrouter_key():
    try:
        secret_key = st.secrets["OPENROUTER_API_KEY"]
        if secret_key:
            return secret_key
    except Exception:
        pass

    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key

    return st.session_state.get("typed_openrouter_key", "")


def dataset_audit(dataframe):
    audit = pd.DataFrame({
        "column": dataframe.columns,
        "dtype": [str(dataframe[col].dtype) for col in dataframe.columns],
        "missing_percent": [round(float(dataframe[col].isna().mean() * 100), 3) for col in dataframe.columns],
        "unique_count": [int(dataframe[col].nunique(dropna=True)) for col in dataframe.columns],
    })
    return audit


def parse_ai_response(text):
    try:
        return json.loads(text), None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except Exception as exc:
            return None, str(exc)

    return None, "No JSON object found in AI response."


def make_baseline_features(ts_df, timestamp_col, target_col, horizon):
    work = ts_df[[timestamp_col, target_col]].copy()
    work = work.sort_values(timestamp_col)
    work["lag_1"] = work[target_col].shift(1)
    work["lag_24"] = work[target_col].shift(24)
    work["rolling_mean_24"] = work[target_col].shift(1).rolling(window=24, min_periods=24).mean()
    work["hour"] = work[timestamp_col].dt.hour
    work["weekend"] = work[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    work["month"] = work[timestamp_col].dt.month
    work["y_target"] = work[target_col].shift(-horizon)

    feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    feature_table = work.dropna(subset=feature_cols + ["y_target"]).copy()
    X = feature_table[feature_cols]
    y = feature_table["y_target"]
    return feature_table, X, y, feature_cols


def make_project_card(info, audit_summary):
    return f"""# Project Card — EDA Mini Project B

## Student
- Name: {info["student_name"]}
- ID: {info["student_id"]}

## Project
- Title: {info["project_title"]}
- Goal: {info["project_goal"]}

## App links
- Deployed Streamlit URL: {info["deployed_url"]}

## Dataset setup
- Dataset path: {info["dataset_path"]}
- Timestamp column: {info["timestamp_col"]}
- Target column: {info["target_col"]}
- Resampling rule: {info["resample_rule"]}
- Forecast horizon: {info["forecast_horizon"]}

## Prepared feature table
- Feature rows: {info["feature_rows"]}
- Feature columns: {", ".join(info["feature_cols"])}

## Audit summary
{audit_summary}

## Student additions checklist
- Add at least one forecasting model under the MODELING marker.
- Use a time-based train/test split.
- Create a metrics table named `results_df`.
- Add extra dashboard plots and KPIs under the DASHBOARD marker.
- Write insights before exporting final files.
"""


# -----------------------------
# Student info
# -----------------------------
st.header("1. Student and project information")

col_a, col_b = st.columns(2)
with col_a:
    student_name = st.text_input("Student name", value="Younis")
    student_id = st.text_input("Student ID", value="PG12S2540600")
    deployed_url = st.text_input("Deployed Streamlit app URL", value="")
with col_b:
    project_title = st.text_input("Project title", value="Electricity Consumption Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Prepare a time-series forecasting dashboard for daily electricity consumption using weather-related features.",
        height=100,
    )

# -----------------------------
# Load dataset
# -----------------------------
st.header("2. Load dataset")

dataset_path = st.text_input("Dataset path", value="data/dataset_sample.csv")

try:
    df = pd.read_csv(dataset_path)
    st.success(f"Loaded dataset from `{dataset_path}` with shape {df.shape[0]:,} rows × {df.shape[1]:,} columns.")
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset audit")
audit_df = dataset_audit(df)
st.dataframe(audit_df, use_container_width=True)

st.subheader("Missing values — top 10")
missing_top = audit_df.sort_values("missing_percent", ascending=False).head(10)
st.dataframe(missing_top[["column", "missing_percent"]], use_container_width=True)

# -----------------------------
# Time-series setup
# -----------------------------
st.header("3. Time-series setup")

default_timestamp = "date" if "date" in df.columns else df.columns[0]
numeric_candidate_cols = [col for col in df.columns if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0]
default_target = "daily_consumption" if "daily_consumption" in df.columns else (numeric_candidate_cols[0] if numeric_candidate_cols else df.columns[0])

col_c, col_d = st.columns(2)
with col_c:
    timestamp_col = st.selectbox(
        "Timestamp column",
        options=list(df.columns),
        index=list(df.columns).index(default_timestamp),
    )
with col_d:
    target_col = st.selectbox(
        "Target column",
        options=list(df.columns),
        index=list(df.columns).index(default_target),
    )

work_df = df.copy()
work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col], errors="coerce")
work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")

before_rows = len(work_df)
work_df = work_df.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
after_rows = len(work_df)

st.write(f"Rows before cleaning: **{before_rows:,}**")
st.write(f"Rows after dropping invalid timestamp/target rows: **{after_rows:,}**")

if after_rows == 0:
    st.error("No valid rows remain after timestamp and target cleaning.")
    st.stop()

min_time = work_df[timestamp_col].min()
max_time = work_df[timestamp_col].max()
st.write(f"Time coverage: **{min_time}** to **{max_time}**")

st.subheader("Optional resampling and horizon")
resample_choice = st.selectbox(
    "Resampling option",
    options=["None", "D", "H", "W", "M"],
    index=0,
    help="Use None to keep the original rows. D=daily, H=hourly, W=weekly, M=monthly.",
)
forecast_horizon = st.number_input("Forecast horizon in periods", min_value=1, max_value=365, value=1, step=1)

ts_df = work_df[[timestamp_col, target_col]].copy()
if resample_choice != "None":
    ts_df = (
        ts_df.set_index(timestamp_col)
        .resample(resample_choice)[target_col]
        .mean()
        .reset_index()
        .dropna(subset=[target_col])
    )
    st.info(f"Applied resampling rule `{resample_choice}` using mean aggregation.")

st.subheader("Target over time")
fig, ax = plt.subplots()
ax.plot(ts_df[timestamp_col], ts_df[target_col])
ax.set_xlabel(timestamp_col)
ax.set_ylabel(target_col)
ax.set_title(f"{target_col} over time")
st.pyplot(fig)

# -----------------------------
# Baseline feature table only
# -----------------------------
st.header("4. Baseline feature table")

feature_table, X, y, feature_cols = make_baseline_features(
    ts_df=ts_df,
    timestamp_col=timestamp_col,
    target_col=target_col,
    horizon=int(forecast_horizon),
)

st.write(f"Prepared feature table rows: **{len(feature_table):,}**")
st.write(f"X shape: **{X.shape}**")
st.write(f"y length: **{len(y):,}**")
st.dataframe(feature_table.head(20), use_container_width=True)

# -----------------------------
# STUDENT ADDITIONS: MODELING
# -----------------------------
st.header("5. STUDENT ADDITIONS — Modeling")

st.warning(
    "Starter code intentionally does not train models or calculate metrics. "
    "Add your forecasting models, time-based split, predictions, and metrics under this marker."
)

st.code(
    """
# STUDENT ADDITIONS: MODELING
# Add your code here.
# Required output for grading:
# results_df = a pandas DataFrame containing model names and metrics.
# Example columns: model, MAE, RMSE, MAPE
results_df = None
""",
    language="python",
)

results_df = None

# -----------------------------
# STUDENT ADDITIONS: DASHBOARD
# -----------------------------
st.header("6. STUDENT ADDITIONS — Dashboard")

st.info("Add extra plots, KPIs, interpretation, and error analysis under this marker.")

st.code(
    """
# STUDENT ADDITIONS: DASHBOARD
# Add visual evidence here.
# Suggested ideas:
# - Actual vs predicted line chart
# - Error distribution
# - Monthly/weekday patterns
# - Model comparison table
""",
    language="python",
)

student_insights = st.text_area(
    "Student insights and interpretation",
    value="Add your final modeling insights here after you complete your model and metrics.",
    height=140,
)

# -----------------------------
# Exports
# -----------------------------
st.header("7. Export files")

has_metrics_table = isinstance(results_df, pd.DataFrame)
results_table = [] if results_df is None else results_df.to_dict(orient="records")

evidence = {
    "student_name": student_name,
    "student_id": student_id,
    "deployed_url": deployed_url,
    "project_title": project_title,
    "project_goal": project_goal,
    "dataset_path": dataset_path,
    "dataset_rows": int(df.shape[0]),
    "dataset_columns": int(df.shape[1]),
    "timestamp_col": timestamp_col,
    "target_col": target_col,
    "time_min": str(min_time),
    "time_max": str(max_time),
    "resample_rule": resample_choice,
    "forecast_horizon": int(forecast_horizon),
    "clean_rows_after_timestamp_target_drop": int(after_rows),
    "feature_rows": int(len(feature_table)),
    "feature_cols": feature_cols,
    "baseline_features_created": True,
    "has_metrics_table": has_metrics_table,
    "results_table": results_table,
    "student_insights": student_insights,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}

submission_json = safe_json_dumps(evidence)

audit_summary = audit_df.to_markdown(index=False)
project_card = make_project_card(
    {
        "student_name": student_name,
        "student_id": student_id,
        "deployed_url": deployed_url,
        "project_title": project_title,
        "project_goal": project_goal,
        "dataset_path": dataset_path,
        "timestamp_col": timestamp_col,
        "target_col": target_col,
        "resample_rule": resample_choice,
        "forecast_horizon": int(forecast_horizon),
        "feature_rows": int(len(feature_table)),
        "feature_cols": feature_cols,
    },
    audit_summary=audit_summary,
)

col_e, col_f = st.columns(2)
with col_e:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col_f:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

st.subheader("Submission JSON preview")
st.json(evidence)

# -----------------------------
# AI grader
# -----------------------------
st.header("8. AI grader out of 80")

st.write(f"Model: `{OPENROUTER_MODEL}`")

typed_key = st.text_input(
    "OpenRouter API key",
    type="password",
    key="typed_openrouter_key",
    help="Used only for this session. Prefer Streamlit Secrets for deployment.",
)

api_key = get_openrouter_key()

grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
    "<insert submission.json contents here>",
    submission_json,
)

with st.expander("Show AI grader prompt"):
    st.text(grader_prompt)

if st.button("Run AI grader"):
    if not api_key:
        st.error("No OpenRouter API key found. Add it in Streamlit Secrets, environment variable, or the password input.")
    else:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": deployed_url or "http://localhost:8501",
                    "X-Title": project_title or "EDA Mini Project B",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": grader_prompt,
                        }
                    ],
                    "temperature": 0,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            raw_output = payload["choices"][0]["message"]["content"]
            parsed, parse_error = parse_ai_response(raw_output)
            if parsed is not None:
                st.success("Parsed AI grader JSON:")
                st.json(parsed)
            else:
                st.warning(f"Could not parse JSON: {parse_error}")
                st.text(raw_output)
        except Exception as exc:
            st.error(f"AI grader request failed: {exc}")
