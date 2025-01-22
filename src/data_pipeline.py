import utils
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

def init_engine():
    db_conn = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    return db_conn

def check_data(input_data, params):
    try:
        print("===== Start Data Defense Checker =====")
        # check data types
        assert input_data.select_dtypes('object').columns.to_list() == params['object_columns'], 'an error occurs in object column(s)'
        assert input_data.select_dtypes('float').columns.to_list() == params['float_columns'], 'an error occurs in float column(s)'
        assert input_data.select_dtypes('int64').columns.to_list() == params['int64_columns'], 'an error occurs in integer column(s)'

        # check values
        assert set(input_data[params['object_columns'][0]]).issubset(set(params['value_person_home_ownership'])), 'an error occurs in person home ownership column'
        assert set(input_data[params['object_columns'][1]]).issubset(set(params['value_loan_intent'])), 'an error occurs in loan intent column'
        assert set(input_data[params['object_columns'][2]]).issubset(set(params['value_loan_grade'])), 'an error occurs in loan grade column'
        assert set(input_data[params['object_columns'][3]]).issubset(set(params['value_cb_person_default_on_file'])), 'an error occurs in cb person default on file column'
    
    except Exception:
        raise Exception("Failed Data Defense Checker")
    
    finally:
        print("===== Finish Data Defense Checker =====")

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
    utils.dump_joblib(X, data_dump_raw + "X.pkl")
    utils.dump_joblib(y, data_dump_raw + "y.pkl")

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
    utils.dump_joblib(X_train, data_dump_interim + "X_train.pkl")
    utils.dump_joblib(y_train, data_dump_interim + "y_train.pkl")
    utils.dump_joblib(X_valid, data_dump_interim + "X_valid.pkl")
    utils.dump_joblib(y_valid, data_dump_interim + "y_valid.pkl")
    utils.dump_joblib(X_test, data_dump_interim + "X_test.pkl")
    utils.dump_joblib(y_test, data_dump_interim + "y_test.pkl")

if __name__ == '__main__':
    params = utils.load_params(params_dir="config/config.yaml")

    DB_HOST = "ep-autumn-bar-a1ubq400.ap-southeast-1.aws.neon.tech"
    DB_USER = "siswa_bfp"
    DB_PASS = "bfp_aksel_keren"
    DB_NAME = "credit_risk_db"
    DB_PORT = "5432"

    conn = init_engine()

    try:
        query = 'select * from credit_risk'
        df = pd.read_sql(sql = query, con = conn)

    except Exception as e:
        raise Exception(f"Error Message: {e}")

    finally:
        conn.dispose()

    utils.dump_joblib(data=df, path=params["dataset_dump_path"]["raw"] + "raw_data.pkl")

    df['person_age'] = df['person_age'].astype('int64')
    df['person_income'] = df['person_income'].astype('int64')

    df = df.drop_duplicates(keep='first')

    check_data(input_data=df, params=params)

    split_data(data=df, params=params)