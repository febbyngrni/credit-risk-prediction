from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.utils.helper import load_joblib, load_params
from src.data_pipeline.data_defense import check_data
from src.preprocessing.preprocess import preprocess_process


# init models and params
params = load_params(param_dir = "config/config.yaml")
best_model = load_joblib(path = "models/rf_best_model_v2.pkl")

# create FastAPI object
app = FastAPI()

# init base model to define the data type
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

# for root dir website, do this process
@app.get("/")
def root():
    return {
        "msg": "Hello, FastAPI is up!",
        "status": "success"
    }


# service for predict ML model based on input data from user
@app.post("/predict")
def predict(data: APIData):
    # Convert input data to DataFrame
    df_data = pd.DataFrame([data.dict()])
 
    # Validate using data checker
    try:
        check_data(input_data=df_data, params=params)
    except AssertionError as ae:
        return {
            "res": [],
            "error_msg": str(ae),
            "status_code": 400
        }
        
    # If valid, preprocess the data
    df_data = preprocess_process(data=df_data, params=params)
    
    # Predict the input data
    y_pred = best_model.predict(df_data)
    
    if y_pred[0] is None:
        return {
            "res": "Failed API",
            "credit_risk_predict": None,
            "status_code": 500,
            "error_msg": "Prediction returned None."
        }
        
    return {
        "res": "Found API",
        "credit_risk_predict": int(y_pred[0]),
        "status_code": 200,
        "error_msg": ""
    }