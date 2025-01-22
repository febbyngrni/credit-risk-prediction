import utils
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV

def load_data(params):
    DATA_PROCESSED_PATH = params["dataset_dump_path"]["processed"]

    X_train_final = utils.load_joblib(path = DATA_PROCESSED_PATH + "X_train_final.pkl")
    y_train = utils.load_joblib(path = DATA_PROCESSED_PATH + "y_train_final.pkl")
    X_test_final = utils.load_joblib(path = DATA_PROCESSED_PATH + "X_test_final.pkl")
    y_test = utils.load_joblib(path = DATA_PROCESSED_PATH + "y_test_final.pkl")
    X_valid_final = utils.load_joblib(path = DATA_PROCESSED_PATH + "X_valid_final.pkl")
    y_valid = utils.load_joblib(path = DATA_PROCESSED_PATH + "y_valid_final.pkl")

    return X_train_final, y_train, X_valid_final, y_valid, X_test_final, y_test

if __name__ == '__main__':
    params = utils.load_params(params_dir="config/config.yaml")

    X_train_final, y_train, X_valid_final, y_valid, X_test_final, y_test = load_data(params)

    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(X_train_final, y_train)
    y_pred_dummy = dummy_clf.predict(X_valid_final)

    print('Classification Report Baseline Model')
    print(classification_report(y_true = y_valid, y_pred = y_pred_dummy))

    logreg = LogisticRegression()
    logreg.fit(X_train_final, y_train)
    y_pred_logreg = logreg.predict(X_valid_final)

    print('Classification Report Vanilla Model')
    print(classification_report(y_true = y_valid, y_pred = y_pred_logreg))

    dc_baseline = DecisionTreeClassifier()
    knn_baseline = KNeighborsClassifier()
    rf_baseline = RandomForestClassifier()
    svc_baseline = SVC()

    dc_baseline.fit(X_train_final, y_train)
    knn_baseline.fit(X_train_final, y_train)
    rf_baseline.fit(X_train_final, y_train)
    svc_baseline.fit(X_train_final, y_train)

    y_pred_dc = dc_baseline.predict(X_valid_final)
    y_pred_knn = knn_baseline.predict(X_valid_final)
    y_pred_rf = rf_baseline.predict(X_valid_final)
    y_pred_svc = svc_baseline.predict(X_valid_final)

    print('Classification Report Multiple Best Model')
    print(classification_report(y_true = y_valid, y_pred = y_pred_rf))

    utils.dump_joblib(rf_baseline, "models/rf_baseline.pkl")

    PARAMS_RF = {
        'n_estimators' : [50, 100],
        'max_depth' : [10, 20],
        'min_samples_split' : [5, 10]
    }
    k_folds = KFold(n_splits = 5)

    best_rf_random = RandomizedSearchCV(
        estimator = rf_baseline,
        param_distributions = PARAMS_RF,
        cv = k_folds,
        verbose = 3
    )
    best_rf_random.fit(X_train_final, y_train)

    best_rf_tune = RandomForestClassifier(n_estimators=100,  min_samples_split=5, max_depth=20)
    best_rf_tune.fit(X_train_final, y_train)
    y_pred_best = best_rf_tune.predict(X_valid_final)

    print('Classification Report Best Model')
    print(classification_report(y_true = y_valid, y_pred = y_pred_best))

    y_pred_final = best_rf_tune.predict(X_test_final)

    print('Classification Report Final Model')
    print(classification_report(y_true = y_valid, y_pred = y_pred_final))

    utils.dump_joblib(best_rf_tune, "models/rf_best_model_v2.pkl")