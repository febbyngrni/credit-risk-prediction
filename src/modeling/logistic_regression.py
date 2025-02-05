import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils.helper import dump_joblib
from sklearn.metrics import accuracy_score, classification_report

def modeling_logeg(X_train: pd.DataFrame, y_train: pd.Series):
    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)
    
    dump_joblib(logreg, "models/vanilla_logreg_model.pkl")
    
    return logreg

def predict_baseline(model, X_valid, y_valid):
    y_pred_logreg = model.predict(X_valid)
    
    print(f'Accuracy: {accuracy_score(y_valid, y_pred_logreg):.3f}')
    print(classification_report(y_valid, y_pred_logreg))