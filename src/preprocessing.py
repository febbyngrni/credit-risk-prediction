import utils
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def filter_outlier(data):
    data_copy = data.copy()

    data_copy = data_copy[
        (data_copy['person_emp_length'] <= 35) | 
        (data_copy['person_age'] <= 70)
    ]

    return data_copy

def split_input_output(data, column):
    data_copy = data.copy()

    y = data_copy[column]
    X = data_copy.drop(columns=column)

    return X, y

def split_num_cat(data: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get cat data
    data_cat = data[params["object_columns"]].copy()

    # get num data
    data_num = data[params["feature_num_columns"]].copy()

    return data_cat, data_num

def ohe_fit(params: dict):
    for col in params['ohe_columns']:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(np.array(params['value_' + col]).reshape(-1,1))
        utils.dump_joblib(ohe, params['dataset_dump_path']['processed'] + f'ohe_model_{col}.pkl')

def ohe_transform(data, params):
    data_copy = data.copy()

    for col in params['ohe_columns']:
        ohe = utils.load_joblib(params['dataset_dump_path']['processed'] + f'ohe_model_{col}.pkl')
        ohe_features = ohe.transform(np.array(data_copy[col].to_list()).reshape(-1,1))

        column_name = ohe.get_feature_names_out([col])
        ohe_features = pd.DataFrame(ohe_features, columns=column_name)

        ohe_features.set_index(data_copy.index, inplace=True)
        data_copy = pd.concat([ohe_features, data_copy], axis=1)
        data_copy.drop(columns=col, inplace=True)

    return data_copy

def custom_label_encoder(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    MAPPER_VALUE = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    }

    for col in params["label_encoder_columns"]:
        data[col] = data[col].replace(MAPPER_VALUE)

    return data

if __name__ == '__main__':
    params = utils.load_params(params_dir="config/config.yaml")

    DATA_INTERIM_PATH = params["dataset_dump_path"]["interim"]

    X_train = utils.load_joblib(DATA_INTERIM_PATH + "X_train.pkl")
    y_train = utils.load_joblib(DATA_INTERIM_PATH + "y_train.pkl")
    X_test = utils.load_joblib(DATA_INTERIM_PATH + "X_test.pkl")
    y_test = utils.load_joblib(DATA_INTERIM_PATH + "y_test.pkl")
    X_valid = utils.load_joblib(DATA_INTERIM_PATH + "X_valid.pkl")
    y_valid = utils.load_joblib(DATA_INTERIM_PATH + "y_valid.pkl")

    imp_med = SimpleImputer(strategy='median')

    # fit to train set
    columns_to_impute = ['person_emp_length', 'loan_int_rate']
    imp_med.fit(X_train[columns_to_impute])

    # transform
    X_train[columns_to_impute] = imp_med.transform(X_train[columns_to_impute])
    X_valid[columns_to_impute] = imp_med.transform(X_valid[columns_to_impute])
    X_test[columns_to_impute] = imp_med.transform(X_test[columns_to_impute])

    train_set = pd.concat([X_train, y_train], axis=1)
    valid_set = pd.concat([X_valid, y_valid], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    train_set = filter_outlier(data=train_set)
    valid_set = filter_outlier(data=valid_set)
    test_set = filter_outlier(data=test_set)

    X_train, y_train = split_input_output(data=train_set, column='loan_status')
    X_valid, y_valid = split_input_output(data=valid_set, column='loan_status')
    X_test, y_test = split_input_output(data=test_set, column='loan_status')

    X_train_cat, X_train_num = split_num_cat(data=X_train, params=params)
    X_valid_cat, X_valid_num = split_num_cat(data=X_valid, params=params)
    X_test_cat, X_test_num = split_num_cat(data=X_test, params=params)

    ohe_fit(params=params)

    X_train_cat = ohe_transform(data=X_train_cat, params=params)
    X_valid_cat = ohe_transform(data=X_valid_cat, params=params)
    X_test_cat = ohe_transform(data=X_test_cat, params=params)

    X_train_cat_le = custom_label_encoder(data = X_train_cat, params = params)
    X_test_cat_le = custom_label_encoder(data = X_test_cat, params = params)
    X_valid_cat_le = custom_label_encoder(data = X_valid_cat, params = params)

    X_train_final = pd.concat([X_train_cat_le, X_train_num], axis = 1)
    X_test_final = pd.concat([X_test_cat_le, X_test_num], axis = 1)
    X_valid_final = pd.concat([X_valid_cat_le, X_valid_num], axis = 1)

    DATA_PROCESSED_PATH = params["dataset_dump_path"]["processed"]

    utils.dump_joblib(X_train_final, DATA_PROCESSED_PATH + "X_train_final.pkl")
    utils.dump_joblib(y_train, DATA_PROCESSED_PATH + "y_train_final.pkl")

    utils.dump_joblib(X_test_final, DATA_PROCESSED_PATH + "X_test_final.pkl")
    utils.dump_joblib(y_test, DATA_PROCESSED_PATH + "y_test_final.pkl")

    utils.dump_joblib(X_valid_final, DATA_PROCESSED_PATH + "X_valid_final.pkl")
    utils.dump_joblib(y_valid, DATA_PROCESSED_PATH + "y_valid_final.pkl")