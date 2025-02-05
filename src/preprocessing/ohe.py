import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.utils.helper import dump_joblib, load_joblib, concat_data

def ohe_fit(params: dict):
    for col in params['ohe_columns']:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(np.array(params['value_' + col]).reshape(-1,1))
        dump_joblib(ohe, params['dataset_dump_path']['processed'] + f'ohe_model_{col}.pkl')

def ohe_transform(data, params):
    data_copy = data.copy()

    for col in params['ohe_columns']:
        ohe = load_joblib(params['dataset_dump_path']['processed'] + f'ohe_model_{col}.pkl')
        ohe_features = ohe.transform(np.array(data_copy[col].to_list()).reshape(-1,1))

        column_name = ohe.get_feature_names_out([col])
        ohe_features = pd.DataFrame(ohe_features, columns=column_name)

        ohe_features.set_index(data_copy.index, inplace=True)
        data_copy = concat_data(ohe_features, data_copy)
        data_copy.drop(columns=col, inplace=True)

    return data_copy