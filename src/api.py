from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import utils
import data_pipeline
import preprocessing

params = utils.load_params('config/config.yaml')
best_model = utils.load_joblib(path=params['model_dump_path'] + 'rf_best_model_v2.pkl')

app = FastAPI()

class APIData(BaseModel):
    person_age : int
    person_income : int
    loan_amnt : int
    cb_person_cred_hist_length : int
    person_emp_length : float
    loan_int_rate : float
    loan_percent_income : float
    person_home_ownership : str
    loan_intent : str
    loan_grade : str
    cb_person_default_on_file : str

@app.get('/')
def root():
    return {
        'message' : 'Hello, FastAPI is up!',
        'status' : 'success'
    }

@app.post('/predict')
def predict(data: APIData):
    df_data = pd.DataFrame([data.dict()])

    try:
        data_pipeline.check_data(df_data, params)
    except AssertionError as ae:
        return {
            'res' : [],
            'error_msg' : str(ae),
            'status_code' : 400
        }
    
    try:
        df_data = preprocessing.ohe_transform(df_data, params)
        print('Transformed Input Data:', df_data)
    except Exception as e:
        return {
            'res' : [],
            'error_msg' : str(e),
            'status_code' : 500
        }
    
    try:
        df_data = preprocessing.custom_label_encoder(df_data, params)
    except Exception as e:
        return {
            'res' : [],
            'error_msg' : str(e),
            'status_code' : 600
        }
    
    y_pred = best_model.predict(df_data)

    if y_pred[0] is None:
        return {
            'res' : 'Failed API',
            'credit_risk_prediction' : None,
            'status_code' : 700,
            'error_msg' : 'Prediction returned None.'
        }
    
    return {
        "res": "Found API",
        "credit_risk_prediction": y_pred[0],
        "status_code": 200,
        "error_msg": ""
    }