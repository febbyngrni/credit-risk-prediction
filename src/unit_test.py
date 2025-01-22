from unittest.mock import patch
import pandas as pd
import numpy as np
import utils
import preprocessing

def test_features_shape():
    # arrange
    X_train = utils.load_joblib("data/processed/X_train_final.pkl")
    X_valid = utils.load_joblib("data/processed/X_valid_final.pkl")
    X_test = utils.load_joblib("data/processed/X_test_final.pkl")

    # act
    N_COLS_THRESH = 11

    # assert
    assert X_train.shape[1] == N_COLS_THRESH, "Input Train columns not match"
    assert X_valid.shape[1] == N_COLS_THRESH, "Input Train columns not match"
    assert X_test.shape[1] == N_COLS_THRESH, "Input Train columns not match"

def test_ohe_transform():
    # arrange
    params = utils.load_params(params_dir="config/config.yaml")
    params['object_columns'] = ['person_home_ownership']

    mock_data = pd.DataFrame({'person_home_ownership' : ['RENT', 'MORTGAGE', 'OWN', 'RENT']})
    expected_columns = ['person_home_ownership_RENT']
    expected_values = pd.DataFrame({'person_home_ownership_RENT' : [1, 0, 0, 1]})

    # act
    preprocessing.ohe_fit(params)
    transformed_data = preprocessing.ohe_transform(mock_data, params)

    # assert
    for col in expected_columns:
        assert col in transformed_data, f'{col} not found in transformed data columns'

    pd.testing.assert_frame_equal(
        transformed_data[expected_values].reset_index(drop=True),
        expected_values,
        check_dtype=False
    )