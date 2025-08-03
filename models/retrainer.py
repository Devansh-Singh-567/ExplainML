# models/retrainer.py
import pandas as pd
import numpy as np

def apply_fixes_and_retrain(df: pd.DataFrame, target: str, suggestions: list):
    """Apply fixes and retrain best model."""
    df_clean = df.copy()

    for suggestion in suggestions:
        sugg_type = suggestion["type"]
        feature = suggestion["feature"]

        if sugg_type == "removal" and feature in df_clean.columns:
            df_clean = df_clean.drop(columns=[feature])
            print(f"❌ Removed: {feature}")

        elif sugg_type == "balancing":
            from imblearn.over_sampling import SMOTE
            X, y = df_clean.drop(columns=[target]), df_clean[target]
            X = X.select_dtypes(include=[float, int]).fillna(0)
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            df_clean = pd.concat([X_res, y_res], axis=1)
            print("♻️ Applied SMOTE")

        elif sugg_type == "transform" and feature in df_clean.columns:
            df_clean[feature] = np.log1p(df_clean[feature])
            print(f"📈 Log-transformed: {feature}")

    return df_clean