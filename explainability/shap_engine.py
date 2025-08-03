# explainability/shap_engine.py
import shap
import matplotlib.pyplot as plt

def explain_model_with_shap(model, X, sample_size=200):
    """Generate SHAP values and return data + figure."""
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
    
    # Use TreeExplainer for tree models, otherwise default
    if hasattr(model, "tree_structure"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X)
    
    shap_values = explainer(X)
    
    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "data_sample": X
    }