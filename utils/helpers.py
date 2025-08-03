# utils/helpers.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# utils/helpers.py
def detect_task_type(y: pd.Series) -> str:
    # If numeric, treat as regression
    if y.dtype in ['int64', 'float64'] and y.nunique() > 20:
        return 'regression'
    # If categorical or low unique count
    if y.dtype == 'object' or y.nunique() <= 20:
        return 'classification'
    return 'regression'

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove spaces and special chars from column names."""
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    return df

def safe_drop_target(df: pd.DataFrame, target: str):
    """
    Safely drop target column and return X, y.
    Ensures y is always a pandas Series (even after encoding).
    """
    X = df.drop(columns=[target], errors='ignore')
    y = df[target].copy()

    # Encode if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target, index=df.index)
    else:
        y = pd.Series(y, name=target, index=df.index)
    
    return X, y