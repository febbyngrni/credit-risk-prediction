import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.utils.helper import dump_joblib
from sklearn.metrics import accuracy_score, classification_report

def modeling_multiple(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dc_baseline = DecisionTreeClassifier()
    knn_baseline = KNeighborsClassifier()
    rf_baseline = RandomForestClassifier()
    svc_baseline = SVC()
    
    dc_baseline.fit(X_train, y_train)
    knn_baseline.fit(X_train, y_train)
    rf_baseline.fit(X_train, y_train)
    svc_baseline.fit(X_train, y_train)
    
    dump_joblib(dc_baseline, "models/dc_baseline.pkl")
    dump_joblib(knn_baseline, "models/knn_baseline.pkl")
    dump_joblib(rf_baseline, "models/rf_baseline.pkl")
    dump_joblib(svc_baseline, "models/svc_baseline.pkl")

    return dc_baseline, knn_baseline, rf_baseline, svc_baseline


# def modeling_logeg(X_train: pd.DataFrame, y_train: pd.Series):
#     logreg = LogisticRegression()

#     logreg.fit(X_train, y_train)
    
#     dump_joblib(logreg, "models/vanilla_logreg_model.pkl")
    
#     return logreg

# def predict_baseline(model, X_valid, y_valid):
#     y_pred_logreg = model.predict(X_valid)
    
#     print(f'Accuracy: {accuracy_score(y_valid, y_pred_logreg):.3f}')
#     print(classification_report(y_valid, y_pred_logreg))