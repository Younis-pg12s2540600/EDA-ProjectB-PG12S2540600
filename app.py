import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
st.caption("Time-series forecasting app with data audit, feature engineering, models, dashboard evidence, exports, and AI grader.")

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

st.success(
    "This section adds student forecasting work: extra features, a time-based split, "
    "hyperparameter tuning, multiple models, predictions, and metrics."
)

if len(feature_table) < 80:
    st.error("Not enough feature rows to train and test models reliably. Try a smaller forecast horizon.")
    results_df = None
    predictions_df = pd.DataFrame()
    best_model_name = None
    best_model = None
    student_feature_cols = feature_cols
    hyperparameter_summary = []
else:
    model_df = feature_table.copy()

    # Extra student-created time-series features beyond the starter baseline.
    model_df["dayofweek"] = model_df[timestamp_col].dt.dayofweek
    model_df["quarter"] = model_df[timestamp_col].dt.quarter
    model_df["lag_7"] = model_df[target_col].shift(7)
    model_df["rolling_mean_7"] = model_df[target_col].shift(1).rolling(7).mean()
    model_df["rolling_std_7"] = model_df[target_col].shift(1).rolling(7).std()

    student_feature_cols = feature_cols + [
        "dayofweek",
        "quarter",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
    ]

    model_df = model_df.dropna(subset=student_feature_cols + ["y_target"]).copy()

    X_model = model_df[student_feature_cols]
    y_model = model_df["y_target"]

    # Time-based train/test split: earlier observations train, later observations test.
    split_index = int(len(model_df) * 0.8)

    X_train = X_model.iloc[:split_index]
    X_test = X_model.iloc[split_index:]
    y_train = y_model.iloc[:split_index]
    y_test = y_model.iloc[split_index:]

    st.subheader("Time-based train/test split")
    st.write(f"Training rows: **{len(X_train):,}**")
    st.write(f"Testing rows: **{len(X_test):,}**")
    st.write(
        "The split is chronological, so the models train on earlier dates and test on later dates. "
        "This reduces look-ahead leakage in a forecasting project."
    )

    # Small hyperparameter grids to keep Streamlit Cloud fast and reliable.
    model_grids = {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0, 50.0]},
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_estimators=120),
            "params": {
                "max_depth": [3, 6, None],
                "min_samples_leaf": [1, 3, 5],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3],
                "n_estimators": [80, 120],
            },
        },
    }

    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    prediction_store = {}
    fitted_models = {}
    hyperparameter_summary = []

    rng = np.random.default_rng(42)

    for model_name, config in model_grids.items():
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        preds = best_estimator.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mape = float(np.nanmean(np.abs((y_test.values - preds) / np.where(y_test.values == 0, np.nan, y_test.values))) * 100)
        r2 = r2_score(y_test, preds)

        # Simple bootstrap confidence interval for RMSE on the test period.
        squared_errors = (y_test.values - preds) ** 2
        boot_rmse = []
        for _ in range(300):
            sample = rng.choice(squared_errors, size=len(squared_errors), replace=True)
            boot_rmse.append(float(np.sqrt(np.mean(sample))))
        rmse_ci_low, rmse_ci_high = np.percentile(boot_rmse, [2.5, 97.5])

        results.append(
            {
                "model": model_name,
                "MAE": round(float(mae), 3),
                "RMSE": round(float(rmse), 3),
                "RMSE_CI_low": round(float(rmse_ci_low), 3),
                "RMSE_CI_high": round(float(rmse_ci_high), 3),
                "MAPE_%": round(float(mape), 3),
                "R2": round(float(r2), 3),
                "best_params": str(grid.best_params_),
                "cv_best_RMSE": round(float(-grid.best_score_), 3),
            }
        )

        prediction_store[model_name] = preds
        fitted_models[model_name] = best_estimator
        hyperparameter_summary.append(
            {
                "model": model_name,
                "best_params": grid.best_params_,
                "cv_best_RMSE": round(float(-grid.best_score_), 3),
            }
        )

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]
    best_predictions = prediction_store[best_model_name]
    best_model = fitted_models[best_model_name]

    predictions_df = pd.DataFrame(
        {
            "timestamp": model_df[timestamp_col].iloc[split_index:].values,
            "actual": y_test.values,
            "predicted": best_predictions,
            "error": y_test.values - best_predictions,
        }
    )

    st.subheader("Model metrics table")
    st.dataframe(results_df, use_container_width=True)
    st.success(f"Best model by RMSE: {best_model_name}")

    st.subheader("Hyperparameter tuning summary")
    st.dataframe(pd.DataFrame(hyperparameter_summary), use_container_width=True)

    # Alternative weekly resampling robustness check.
    weekly_robustness_df = pd.DataFrame()
    try:
        weekly_df = (
            work_df[[timestamp_col, target_col]]
            .set_index(timestamp_col)
            .resample("W")[target_col]
            .mean()
            .reset_index()
            .dropna(subset=[target_col])
        )
        if len(weekly_df) > 60:
            weekly_features, weekly_X, weekly_y, weekly_cols = make_baseline_features(
                weekly_df,
                timestamp_col=timestamp_col,
                target_col=target_col,
                horizon=1,
            )
            if len(weekly_features) > 40:
                weekly_split = int(len(weekly_features) * 0.8)
                weekly_model = Ridge(alpha=1.0)
                weekly_model.fit(weekly_X.iloc[:weekly_split], weekly_y.iloc[:weekly_split])
                weekly_preds = weekly_model.predict(weekly_X.iloc[weekly_split:])
                weekly_rmse = float(np.sqrt(mean_squared_error(weekly_y.iloc[weekly_split:], weekly_preds)))
                weekly_mae = float(mean_absolute_error(weekly_y.iloc[weekly_split:], weekly_preds))
                weekly_robustness_df = pd.DataFrame(
                    [{
                        "frequency": "Weekly",
                        "model": "Ridge Regression",
                        "horizon": 1,
                        "MAE": round(weekly_mae, 3),
                        "RMSE": round(weekly_rmse, 3),
                        "rows": int(len(weekly_features)),
                    }]
                )
    except Exception:
        weekly_robustness_df = pd.DataFrame()

    if not weekly_robustness_df.empty:
        st.subheader("Alternative resampling robustness check")
        st.dataframe(weekly_robustness_df, use_container_width=True)
        st.write(
            "This check demonstrates an alternative weekly frequency. The main model remains daily "
            "because daily forecasting preserves more observations and gives a more detailed forecast."
        )

# -----------------------------
# STUDENT ADDITIONS: DASHBOARD
# -----------------------------
st.header("6. STUDENT ADDITIONS — Dashboard")

st.info("Dashboard evidence includes actual vs predicted values, residual diagnostics, model comparison, feature importance, and resampling robustness.")

dashboard_plot_count = 0

if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    with kpi_a:
        st.metric("Best model", best_model_name)
    with kpi_b:
        st.metric("Best RMSE", results_df.iloc[0]["RMSE"])
    with kpi_c:
        st.metric("Best MAE", results_df.iloc[0]["MAE"])
    with kpi_d:
        st.metric("Best R2", results_df.iloc[0]["R2"])

    st.subheader("Actual vs predicted consumption")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions_df["timestamp"], predictions_df["actual"], label="Actual")
    ax.plot(predictions_df["timestamp"], predictions_df["predicted"], label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily electricity consumption")
    ax.set_title(f"Actual vs Predicted — {best_model_name}")
    ax.legend()
    st.pyplot(fig)
    dashboard_plot_count += 1

    st.subheader("Prediction error over time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions_df["timestamp"], predictions_df["error"])
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction error")
    ax.set_title("Forecast Error Over Time")
    st.pyplot(fig)
    dashboard_plot_count += 1

    st.subheader("Residual distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(predictions_df["error"], bins=30)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    st.pyplot(fig)
    dashboard_plot_count += 1

    st.subheader("Model comparison by RMSE")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results_df["model"], results_df["RMSE"])
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison by RMSE")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    dashboard_plot_count += 1

    st.subheader("Feature importance / model interpretability")
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_,
            index=student_feature_cols,
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(importances.index, importances.values)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.set_title(f"Feature Importance — {best_model_name}")
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig)
        dashboard_plot_count += 1

        st.write(
            "This chart improves interpretability by showing which lag, rolling, "
            "and calendar features contributed most to the selected tree-based model."
        )
    elif hasattr(best_model, "coef_"):
        coefficients = pd.Series(
            best_model.coef_,
            index=student_feature_cols,
        ).sort_values(key=lambda s: s.abs(), ascending=False)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(coefficients.index, coefficients.values)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"Ridge Coefficients — {best_model_name}")
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig)
        dashboard_plot_count += 1

        st.write(
            "This chart improves interpretability by showing which features have the largest "
            "positive or negative coefficients in the selected Ridge model."
        )

    if "weekly_robustness_df" in globals() and isinstance(weekly_robustness_df, pd.DataFrame) and not weekly_robustness_df.empty:
        st.subheader("Weekly resampling robustness visual")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(weekly_robustness_df["frequency"], weekly_robustness_df["RMSE"])
        ax.set_xlabel("Alternative frequency")
        ax.set_ylabel("RMSE")
        ax.set_title("Alternative Weekly Resampling Check")
        st.pyplot(fig)
        dashboard_plot_count += 1

    st.subheader("Forecasting insights")
    st.write(
        f"""
        The model uses a time-based 80/20 split, so earlier dates are used for training and
        later dates are used for testing. The best model is **{best_model_name}** based on RMSE.

        Extra features were added beyond the starter baseline: day of week, quarter, lag 7,
        rolling mean 7, and rolling standard deviation 7. Hyperparameter tuning was performed
        using TimeSeriesSplit cross-validation, and the final models were evaluated on the
        held-out future test period.

        The residual chart and residual distribution help diagnose over-prediction and
        under-prediction. The feature-importance or coefficient chart supports model
        interpretability by showing which engineered features were most influential.
        """
    )
else:
    st.warning("Run the modeling section successfully before dashboard plots can be displayed.")

student_insights = st.text_area(
    "Student insights and interpretation",
    value="""This project uses a chronological 80/20 time-based split, so earlier observations are used for training and later observations are used for testing. This avoids data leakage because the model does not train on future observations.

The dataset was checked for missing values, invalid timestamps, invalid target values, and outliers. The timestamp column was parsed as datetime and the target column was converted to numeric. Invalid timestamp or target rows were removed before sorting by date.

The dataset is already daily electricity consumption, so daily frequency is the main modelling frequency. A weekly resampling robustness check is also included to demonstrate how results can change when the time frequency is aggregated.

Multiple models were compared using MAE, RMSE, MAPE, R2, and RMSE confidence intervals. Hyperparameter tuning was performed with TimeSeriesSplit cross-validation. RMSE is used to select the best model because it penalizes larger forecasting errors.

Additional features were created beyond the starter baseline, including day of week, quarter, lag 7, rolling mean 7, and rolling standard deviation 7. These features help capture weekly patterns, recent average demand, recent volatility, and calendar effects.

The dashboard includes actual vs predicted values, prediction error over time, residual distribution, model comparison, and feature importance or coefficient interpretation. These visuals explain both forecast accuracy and model behaviour.""",
    height=260,
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
    "time_based_split_used": True if isinstance(results_df, pd.DataFrame) else False,
    "models_used": [] if results_df is None else results_df["model"].tolist(),
    "hyperparameter_tuning_used": True if isinstance(results_df, pd.DataFrame) else False,
    "hyperparameter_summary": [] if "hyperparameter_summary" not in globals() else hyperparameter_summary,
    "dashboard_plot_count": int(dashboard_plot_count),
    "feature_importance_or_coefficients_shown": True if dashboard_plot_count >= 5 else False,
    "weekly_resampling_check": False if "weekly_robustness_df" not in globals() else bool(isinstance(weekly_robustness_df, pd.DataFrame) and not weekly_robustness_df.empty),
    "rmse_confidence_interval_included": True if isinstance(results_df, pd.DataFrame) and "RMSE_CI_low" in results_df.columns else False,
    "student_insights": student_insights,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}

submission_json = safe_json_dumps(evidence)

audit_summary = audit_df.to_csv(index=False)
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

