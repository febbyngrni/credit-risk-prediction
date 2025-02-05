from src.utils.helper import load_params
from src.utils.helper import split_num_cat, concat_data
from src.preprocessing.ohe import ohe_transform
from src.preprocessing.label_encoding import custom_label_encoder
import pandas as pd

params = load_params(param_dir = "config/config.yaml")

def preprocess_process(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    cat_data, num_data = split_num_cat(data = data, params = params)
    
    cat_ohe_data = ohe_transform(data=cat_data, params=params)
    
    cat_final_data = custom_label_encoder(data = cat_ohe_data, params = params)
    
    final_data = concat_data(data=cat_final_data, data_2=num_data)
    
    return final_data