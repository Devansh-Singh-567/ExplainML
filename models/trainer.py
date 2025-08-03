# models/trainer.py
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_models(task_type: str):
    if task_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        }
    elif task_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(eval_metric='rmse', random_state=42)
        }
    return {}

def evaluate_models(X: pd.DataFrame, y: pd.Series, cv=3):
    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    if X_num.empty:
        raise ValueError("No numeric features available.")

    # Detect task type
    task_type = 'classification' if y.nunique() <= 20 else 'regression'

    # Encode y only if classification and not already numeric
    original_y = y.copy()
    if task_type == "classification":
        if y.dtype == 'object' or y.dtype == 'float64':
            le = LabelEncoder()
            y = le.fit_transform(y)
        scoring = 'f1_macro'
    else:
        scoring = 'r2'  # Use R² (better interpretation)

    models = get_models(task_type)
    results = []

    for name, model in models.items():
        try:
            # Scale only for linear models in regression
            X_used = X_num.copy()
            if task_type == "regression" and "Linear" in name:
                scaler = StandardScaler()
                X_used = scaler.fit_transform(X_used)
            else:
                X_used = X_used.values  # Just ensure array

            scores = cross_val_score(model, X_used, y, cv=cv, scoring=scoring)
            results.append({
                "model": name,
                "score_mean": scores.mean(),
                "score_std": scores.std(),
                "model_obj": model
            })
        except Exception as e:
            print(f"❌ Failed {name}: {e}")
            continue

    if not results:
        raise ValueError("No models were able to train successfully.")

    results_df = pd.DataFrame(results).sort_values("score_mean", ascending=False)
    best_model = results_df.iloc[0]["model_obj"]

    # Refit on full data (with scaling if needed)
    X_fit = X_num.copy()
    if task_type == "regression" and "Linear" in results_df.iloc[0]["model"]:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)
        # Wrap model with scaler if needed (advanced)
    best_model.fit(X_fit, y)

    results_df["task_type"] = task_type
    return results_df, best_model