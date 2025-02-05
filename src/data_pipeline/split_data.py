import pandas as pd
from src.utils.helper import dump_joblib
from sklearn.model_selection import train_test_split                                                                                                                                                                                                                                                                       

def split_data(data: pd.DataFrame, params: dict) -> None:
    # set params
    data_dump_raw = params['dataset_dump_path']['raw']
    data_dump_interim = params['dataset_dump_path']['interim']
    target_col = params['target_col']

    # set target col
    y = data[target_col]
    X = data.drop(columns=target_col, axis=1)

    # validation
    print(f'Feature shape: {X.shape}')
    print(f'Target shape: {y.shape}')

    # save the X and y to pkl
    dump_joblib(X, data_dump_raw + "X.pkl")
    dump_joblib(y, data_dump_raw + "y.pkl")

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 42,
        stratify = y)
    
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test,
        y_test,
        test_size = 0.2,
        random_state = 42,
        stratify = y_test)
    
    # validation
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_valid shape  :', X_valid.shape)
    print('y_valid shape  :', y_valid.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)    
    
    # dump
    dump_joblib(X_train, data_dump_interim + "X_train.pkl")
    dump_joblib(y_train, data_dump_interim + "y_train.pkl")
    dump_joblib(X_valid, data_dump_interim + "X_valid.pkl")
    dump_joblib(y_valid, data_dump_interim + "y_valid.pkl")
    dump_joblib(X_test, data_dump_interim + "X_test.pkl")
    dump_joblib(y_test, data_dump_interim + "y_test.pkl")