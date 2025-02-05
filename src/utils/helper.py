import yaml
import joblib
import pandas as pd
from sqlalchemy import create_engine

def dump_joblib(data, path: str) -> None:
    joblib.dump(data, path)

def load_joblib(path: str):
    return joblib.load(path)

def load_params(param_dir):
    with open(param_dir, 'r') as file:
        params = yaml.safe_load(file)

    return params

def concat_data(data, data_2) -> pd.DataFrame:
    final_data = pd.concat([data, data_2], axis = 1)
    
    return final_data

def init_engine():
    DB_HOST = "ep-autumn-bar-a1ubq400.ap-southeast-1.aws.neon.tech"
    DB_USER = "siswa_bfp"
    DB_PASS = "bfp_aksel_keren"
    DB_NAME = "credit_risk_db"
    DB_PORT = "5432"

    conn = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    try:
        query = 'select * from credit_risk'
        df = pd.read_sql(sql = query, con = conn)
        return df

    except Exception as e:
        raise Exception(f"Error Message: {e}")

    finally:
        conn.dispose()

def split_input_output(data, column):
    data_copy = data.copy()

    y = data_copy[column]
    X = data_copy.drop(columns=column)

    return X, y

def split_num_cat(data: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get cat data
    data_cat = data[params['object_columns']].copy()

    # get num data
    data_num = data[params['feature_num_columns']].copy()

    return data_cat, data_num