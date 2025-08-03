# models/tuner.py
import optuna
from sklearn.model_selection import cross_val_score
import numpy as np

def objective(trial, model_class, X, y):
    params = {}
    if model_class.__name__ == "RandomForestClassifier":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
        }
    elif model_class.__name__ == "XGBClassifier":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }
    model = model_class(**params, random_state=42)
    return np.mean(cross_val_score(model, X, y, cv=3, scoring="f1"))

def tune_model(model_class, X, y, n_trials=20):
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, model_class, X, y)
    study.optimize(func, n_trials=n_trials)
    return study.best_params