# nlp/insight_summarizer.py
def generate_natural_insights(diag_data: dict) -> str:
    """Generate human-like insight summary."""
    target = diag_data["target"]
    best_model = diag_data["best_model"]
    f1 = diag_data["f1_score"]
    suggestions = {s["type"]: s for s in diag_data.get("suggestions", [])}

    insights = f"The {best_model} model achieves an F1 score of {f1:.2f} in predicting '{target}'. "

    if "balancing" in suggestions:
        insights += "Performance may be limited by class imbalance; resampling could help. "
    if "removal" in suggestions:
        insights += "Several features with high missing rates were dropped to improve reliability. "
    if "error" in suggestions:
        insights += "The model struggles with specific groups (e.g., older low-fare passengers), suggesting bias or data gaps. "
    if "leakage" in suggestions:
        insights += "Potential data leakage was detected and corrected. "

    insights += "Recommendations: " + ", ".join([s["suggestion"] for s in suggestions.values()][:2]) + "."
    return insights