# visualizer/plots.py
import matplotlib.pyplot as plt
import shap

def plot_shap_summary(shap_data):
    """Plot SHAP summary using matplotlib backend."""
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_data["shap_values"], shap_data["data_sample"], show=False)
    return fig

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    return fig