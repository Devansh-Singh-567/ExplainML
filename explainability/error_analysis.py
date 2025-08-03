# explainability/error_analysis.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def find_error_clusters(X_test, y_test, y_pred, shap_values, feature_names, n_clusters=3):
    """Find clusters of misclassified samples."""
    if len(y_test) != len(y_pred):
        return []
    
    # Get misclassified indices
    errors_mask = (y_test != y_pred)
    if not errors_mask.any():
        return []
    
    X_err = X_test[errors_mask]
    if len(X_err) < n_clusters:
        return []
    
    # Cluster top 2 most important features from SHAP
    try:
        err_shap = shap_values[errors_mask]
        mean_abs_shap = np.mean(np.abs(err_shap.values), axis=0)
        top_idx = np.argsort(mean_abs_shap)[-2:]
        cluster_data = X_err.iloc[:, top_idx].values

        kmeans = KMeans(n_clusters=min(n_clusters, len(cluster_data)), n_init=10, random_state=42)
        labels = kmeans.fit_predict(cluster_data)

        clusters = []
        for i in range(kmeans.n_clusters):
            group = X_err[labels == i]
            center = kmeans.cluster_centers_[i]
            desc = f"{feature_names[top_idx[0]]}≈{center[0]:.1f}, {feature_names[top_idx[1]]}≈{center[1]:.1f}"
            clusters.append({
                "condition": desc,
                "size": len(group),
                "features": [feature_names[j] for j in top_idx]
            })
        return clusters
    except Exception as e:
        print(f"Clustering failed: {e}")
        return []