# app.py
import streamlit as st
import pandas as pd
import os
import numpy as np

# Import modules
from profiler.stats_report import analyze_dataset
from profiler.leakage_detector import detect_target_leakage, detect_high_correlation
from models.trainer import evaluate_models
from explainability.shap_engine import explain_model_with_shap
from explainability.error_analysis import find_error_clusters
from explainability.fairness_checker import check_fairness
from recommender.fix_generator import generate_suggestions
from visualizer.plots import plot_shap_summary, plot_confusion_matrix
from reports.report_generator import generate_markdown_report, generate_pdf_report
from utils.helpers import clean_column_names, safe_drop_target

# Page config
st.set_page_config(
    page_title="🧠 ExplainML++",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🧠 ExplainML++ – Intelligent AutoML with Failure Diagnosis")

st.markdown("""
Upload a **CSV file** to automatically:
- Analyze data quality
- Train and compare models
- Diagnose failures
- Suggest fixes
- Generate reports
""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = clean_column_names(df)

        st.success(f"✅ Loaded `{uploaded_file.name}` with `{len(df)} rows` and `{len(df.columns)} columns`.")

        # Show data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10))
            st.write(f"**Columns:** {', '.join(df.columns)}")

        # Target selection
        target_col = st.selectbox(
            "🎯 Select the target column (what you want to predict):",
            df.columns,
            index=None,
            placeholder="Choose a column..."
        )

        if target_col is None:
            st.info("👈 Please select a target column to begin analysis.")
            st.stop()

        # === VALIDATION: Check if target is valid ===
        if df[target_col].nunique() == len(df):
            st.error(f"""
            ❌ The column **`{target_col}`** has a unique value for every row — it looks like an **ID or name**, not a label.

            💡 **Tip**: Choose a column like:
            - `Class`, `Status`, `Category`, `Rating`, or `Outcome`
            - Something with **repeated values** (e.g., 'Hit/Flop', 'High/Medium/Low')
            """)
            st.stop()

        if df[target_col].nunique() < 2:
            st.error("❌ Target column must have at least 2 unique values to learn from.")
            st.stop()

        if len(df) < 10:
            st.warning("⚠️ Dataset is very small (<10 rows). Results may not be reliable.")

        # Button to start
        if st.button("🚀 Start AutoML Analysis", type="primary"):
            with st.spinner("🔍 Analyzing dataset and training models..."):
                # --- 1. Profiling ---
                try:
                    profile = analyze_dataset(df, target_col)
                except Exception as e:
                    st.error(f"❌ Failed to analyze dataset: {e}")
                    st.stop()

                st.subheader("🔍 Dataset Profile")
                st.json({
                    "rows": profile["rows"],
                    "columns": profile["columns"],
                    "target": profile["target"],
                    "task_type": profile["task_type"],
                    "missing_percentage": {k: f"{v:.1f}%" for k, v in profile["missing_percentage"].items() if v > 0},
                    "imbalance_ratio": round(profile["imbalance_ratio"], 2) if profile["task_type"] == "classification" else None,
                    "class_distribution": profile["class_distribution"] if profile["task_type"] == "classification" else "N/A"
                })

                # --- 2. Prepare Features ---
                X, y = safe_drop_target(df, target_col)
                X_num = X.select_dtypes(include=[np.number]).fillna(0)

                if X_num.empty:
                    st.error("""
                    ❌ No numeric features found. 
                    
                    💡 Add numeric columns (e.g., budget, score, age) or extend with feature engineering.
                    """)
                    st.stop()

                # --- 3. Leakage & Correlation ---
                try:
                    leaks = detect_target_leakage(X_num, y, threshold=0.8)
                    corrs = detect_high_correlation(X_num, threshold=0.9)

                    if leaks:
                        st.warning(f"⚠️ **Possible data leakage**: {leaks}")
                    if corrs:
                        st.warning(f"⚠️ **High correlation between features**: {corrs}")
                except Exception as e:
                    st.caption(f"🔍 Leakage/correlation check failed: {e}")

                # --- 4. Model Training ---
                try:
                    results_df, best_model = evaluate_models(X_num, y, cv=3)
                    task_type = results_df["task_type"].iloc[0]

                    st.subheader("🏆 Model Performance")
                    if task_type == "regression":
                        results_df_display = results_df[["model", "score_mean", "score_std"]].rename(
                            columns={"score_mean": "R² Mean", "score_std": "R² Std"}
                        )
                        st.dataframe(results_df_display.round(3))
                        best_score = results_df.iloc[0]["score_mean"]
                        st.success(f"✅ **Best Model**: `{results_df.iloc[0]['model']}` (R² = `{best_score:.3f}`)")
                    else:
                        results_df_display = results_df[["model", "score_mean", "score_std"]].rename(
                            columns={"score_mean": "F1 Mean", "score_std": "F1 Std"}
                        )
                        st.dataframe(results_df_display.round(3))
                        best_score = results_df.iloc[0]["score_mean"]
                        st.success(f"✅ **Best Model**: `{results_df.iloc[0]['model']}` (F1 = `{best_score:.3f}`)")

                    # Predictions for analysis
                    y_pred = best_model.predict(X_num)

                except Exception as e:
                    st.error("❌ Model training failed.")
                    st.exception(e)
                    st.stop()

                # --- 5. SHAP Explainability ---
                try:
                    shap_data = explain_model_with_shap(best_model, X_num, sample_size=min(200, len(X_num)))
                    st.subheader("🧠 Model Explainability (SHAP)")
                    fig = plot_shap_summary(shap_data)
                    st.pyplot(fig)
                    st.caption("Top features influencing predictions")
                except Exception as e:
                    st.warning(f"⚠️ SHAP explanation failed: {e}")

                # --- 6. Error Analysis (Classification only) ---
                if task_type == "classification" and 'shap_data' in locals():
                    try:
                        error_clusters = find_error_clusters(
                            X_num, y, y_pred, shap_data["shap_values"], X_num.columns
                        )
                        if error_clusters:
                            st.subheader("💥 Failure Patterns")
                            for cluster in error_clusters[:3]:
                                st.markdown(f"- 🔍 High errors in: `{cluster['condition']}` (size: {cluster['size']})")
                    except Exception as e:
                        st.caption(f"🔍 Error clustering failed: {e}")

                # --- 7. Fairness Check (if binary column exists) ---
                if task_type == "classification":
                    sensitive_cols = [col for col in X.columns if X[col].nunique() == 2]
                    if sensitive_cols:
                        st.subheader("⚖️ Fairness Check")
                        for col in sensitive_cols[:2]:
                            try:
                                fairness = check_fairness(y, y_pred, X[col])
                                st.write(f"**{col}**: {fairness}")
                            except Exception as e:
                                st.caption(f"Fairness check failed for {col}: {e}")

                # --- 8. Fix Suggestions ---
                issues = {
                    "imbalance_ratio": profile.get("imbalance_ratio"),
                    "missing_percentage": profile["missing_percentage"],
                    "numeric_skew": profile["numeric_skew"],
                    "target_leakage": leaks if 'leaks' in locals() else [],
                    "high_correlation": corrs if 'corrs' in locals() else [],
                    "error_clusters": error_clusters if 'error_clusters' in locals() else []
                }

                diag_data = {
                    "dataset": uploaded_file.name,
                    "target": target_col,
                    "best_model": results_df.iloc[0]["model"],
                    "f1_score": best_score,
                    "issues": issues
                }

                suggestions = generate_suggestions(diag_data)
                diag_data["suggestions"] = suggestions

                if suggestions:
                    st.subheader("🛠️ Suggested Improvements")
                    priority_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    for s in suggestions:
                        p = s["priority"].lower()
                        icon = priority_icons.get(p, "⚪")
                        st.markdown(f"{icon} **{p.upper()}**: {s['suggestion']}")

                # --- 9. Generate Reports ---
                st.subheader("📄 Auto-Generated Reports")

                os.makedirs("reports", exist_ok=True)
                try:
                    generate_markdown_report(diag_data, "reports/diagnostic_report.md")
                    generate_pdf_report(diag_data, "reports/diagnostic_report.pdf")

                    col1, col2 = st.columns(2)
                    with col1:
                        with open("reports/diagnostic_report.md", "r", encoding="utf-8") as f:
                            st.download_button(
                                "⬇️ Download Markdown Report",
                                f.read(),
                                "diagnostic_report.md",
                                "text/markdown"
                            )
                    with col2:
                        with open("reports/diagnostic_report.pdf", "rb") as f:
                            st.download_button(
                                "📄 Download PDF Report",
                                f.read(),
                                "diagnostic_report.pdf",
                                "application/pdf"
                            )
                except Exception as e:
                    st.error(f"📄 Report generation failed: {e}")

    except Exception as e:
        st.error("❌ Failed to process file. Please upload a valid CSV.")
        st.exception(e)