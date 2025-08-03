# recommender/fix_generator.py
def generate_suggestions(diag_data: dict) -> list:
    suggestions = []
    issues = diag_data.get("issues", {})

    # Handle imbalance_ratio safely
    imbalance_ratio = issues.get("imbalance_ratio")
    if imbalance_ratio is not None and isinstance(imbalance_ratio, (int, float)) and imbalance_ratio > 2.0:
        suggestions.append({
            "type": "balancing",
            "feature": diag_data.get("target"),
            "issue": f"High class imbalance ({imbalance_ratio:.1f}x)",
            "suggestion": "Apply SMOTE or class weighting.",
            "priority": "high"
        })

    # Missing data
    for col, pct in issues.get("missing_percentage", {}).items():
        if pct > 50:
            suggestions.append({
                "type": "removal",
                "feature": col,
                "issue": f"{pct:.1f}% missing",
                "suggestion": f"Remove '{col}' due to excessive missing data.",
                "priority": "high"
            })
        elif pct > 20:
            suggestions.append({
                "type": "imputation",
                "feature": col,
                "issue": f"{pct:.1f}% missing",
                "suggestion": f"Impute '{col}' using median or mode.",
                "priority": "medium"
            })

    # Skewness
    for col, skew_val in issues.get("numeric_skew", {}).items():
        if abs(skew_val) > 2.0:
            suggestions.append({
                "type": "transformation",
                "feature": col,
                "issue": f"High skewness ({skew_val:.2f})",
                "suggestion": f"Apply log transform to '{col}'.",
                "priority": "high"
            })

    # Leakage
    for col, corr in issues.get("target_leakage", []):
        suggestions.append({
            "type": "leakage",
            "feature": col,
            "issue": f"High corr with target ({corr:.2f})",
            "suggestion": f"⚠️ Remove '{col}' (data leakage risk).",
            "priority": "critical"
        })

    # Correlation
    for col1, col2 in issues.get("high_correlation", []):
        suggestions.append({
            "type": "collinearity",
            "feature": f"{col1}, {col2}",
            "issue": "High correlation",
            "suggestion": f"Remove one of '{col1}' or '{col2}'.",
            "priority": "medium"
        })

    # Error clusters
    for cluster in issues.get("error_clusters", []):
        suggestions.append({
            "type": "error",
            "feature": ", ".join(cluster.get("features", [])),
            "issue": f"High errors in {cluster['condition']}",
            "suggestion": "Collect more data or create interaction feature.",
            "priority": "high"
        })

    return suggestions