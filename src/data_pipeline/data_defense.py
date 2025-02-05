import pandas as pd

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