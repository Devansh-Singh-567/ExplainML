# explainability/fairness_checker.py
import pandas as pd

def check_fairness(y_true, y_pred, sensitive_col: pd.Series):
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "group": sensitive_col})
    group_stats = df.groupby("group").apply(
        lambda g: {"accuracy": (g['true'] == g['pred']).mean(), "pred_positive_rate": g['pred'].mean()}
    )
    return group_stats.to_dict()