import pandas as pd

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