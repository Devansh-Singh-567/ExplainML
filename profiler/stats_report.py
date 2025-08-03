# profiler/stats_report.py
import pandas as pd
import numpy as np
from scipy.stats import skew
from utils.helpers import detect_task_type

def analyze_dataset(df: pd.DataFrame, target_col: str):
    """Analyze dataset structure, missingness, imbalance, skew."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df.dropna(subset=[target_col])  # Ensure target is clean

    # Detect task type
    y = df[target_col]
    task_type = detect_task_type(y)

    # Convert y to Series if needed (e.g., after prior encoding)
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name=target_col)

    # Missing data
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    # Class imbalance or target stats
    if task_type == "classification":
        class_counts = y.value_counts()
        class_pcts = (class_counts / len(df)) * 100
        imbalance_ratio = class_pcts.max() / class_pcts.min()
    else:
        class_counts = None
        imbalance_ratio = None

    # Skewness
    numeric_skew = df.select_dtypes(include=[np.number]).apply(skew).to_dict()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "target": target_col,
        "task_type": task_type,
        "missing_data": missing.to_dict(),
        "missing_percentage": missing_pct.to_dict(),
        "numeric_skew": numeric_skew,
        "class_distribution": class_counts.to_dict() if class_counts is not None else None,
        "imbalance_ratio": float(imbalance_ratio) if imbalance_ratio else None,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }