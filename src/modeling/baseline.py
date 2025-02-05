import pandas as pd
from sklearn.dummy import DummyClassifier
from src.utils.helper import dump_joblib
from sklearn.metrics import accuracy_score, classification_report

def modeling_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    dummy_clf = DummyClassifier(strategy = "most_frequent")

    dummy_clf.fit(X_train, y_train)
    
    dump_joblib(dummy_clf, "models/baseline_model.pkl")
    
    return dummy_clf

def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f'Accuracy: {accuracy_score(y_valid, y_pred_dummy):.3f}')
    print(classification_report(y_valid, y_pred_dummy))