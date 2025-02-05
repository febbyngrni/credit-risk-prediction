import numpy as np
import pandas as pd
from src.utils.helper import load_joblib, dump_joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, fbeta_score


def hyperparam_process(model_path: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = load_joblib(path = model_path)
    
    PARAMS_RF = {
        'n_estimators' : [50, 100],
        'max_depth' : [10, 20],
        'min_samples_split' : [5, 10]
    }
    
    k_folds = KFold(n_splits = 5)
    
    best_rf_random = RandomizedSearchCV(
        estimator = model,
        param_distributions = PARAMS_RF,
        cv = k_folds,
        verbose = 3
    )
    
    best_rf_random.fit(X_train, y_train)
    
    return best_rf_random.best_params_


def best_model_train(X_train: pd.DataFrame, y_train: pd.Series):
    best_model = RandomForestClassifier(n_estimators=100,  min_samples_split=10, max_depth=20)
    
    best_model.fit(X_train, y_train)
    
    dump_joblib(best_model, "models/best_model.pkl")
    
    return best_model

def threshold_tuning(model, X_valid: pd.DataFrame, y_valid: pd.Series):
    thresholds = np.arange(0.0, 1.01, 0.01)
    f2_scores = []

    y_pred_proba = model.predict_proba(X_valid)[:, 1]

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f2 = fbeta_score(y_valid, y_pred, beta=2)
        f2_scores.append(f2)

    optimal_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f2 = f2_scores[optimal_idx]

    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Maximum F2-Score: {optimal_f2}")

    return optimal_threshold


def predict_best_model(model, X_test, y_test):
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_thres_test = (y_pred_proba_test >= 0.2).astype(int)
    
    print(f'Accuracy Test Set: {accuracy_score(y_test, y_pred_thres_test):.3f}')
    print(classification_report(y_test, y_pred_thres_test))