# profiler/leakage_detector.py
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

def detect_target_leakage(X: pd.DataFrame, y: pd.Series, threshold=0.8):
    """
    Detect features highly correlated with target (possible leakage).
    Handles both pandas Series and numpy arrays.
    """
    X_num = X.select_dtypes(include=[np.number])
    if X_num.empty:
        return []

    # Convert y to pandas Series if it's a numpy array
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    leaks = []
    for col in X_num.columns:
        if y.nunique() <= 2:
            # Binary classification: use point biserial correlation
            try:
                corr = pointbiserialr(y, X_num[col])[0]
            except:
                corr = 0.0
        else:
            # Regression or multi-class: use Pearson
            corr = y.corr(X_num[col])
        if abs(corr) > threshold:
            leaks.append((col, round(corr, 3)))
    return leaks

def detect_high_correlation(X: pd.DataFrame, threshold=0.9):
    """Detect multicollinearity between numeric features."""
    X_num = X.select_dtypes(include=[np.number])
    if X_num.empty:
        return []
    
    corr_matrix = X_num.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    highly_corr = [
        (X_num.columns[i], X_num.columns[j]) 
        for i, j in zip(*np.where(upper > threshold)) 
        if i < j
    ]
    return highly_corr