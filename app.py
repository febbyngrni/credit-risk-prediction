import streamlit as st
import requests

st.title('Credit Risk Prediction')
st.subheader("Enter the data below!")


with st.form(key = 'credit_risk_form'):
    person_age = st.number_input(
        label = "Enter the person's age:",
        min_value = 0,
        max_value = 80,
        help = 'Person age from 0 to 80'
    )

    person_income = st.number_input(
        label = "Enter the person's income:",
        min_value = 0,
        max_value = 10000000,
        help = 'Person age from 0 to 10.000.000'
    )

    loan_amnt = st.number_input(
        label = "Enter the number of loan amount:",
        min_value = 0,
        max_value = 100000,
        help = 'Person age from 0 to 100.000'
    )

    cb_person_cred_hist_length = st.number_input(
        label = "Enter the number of credit history length:",
        min_value = 0,
        max_value = 50,
        help = 'Person age from 0 to 50'
    )

    person_emp_length = st.number_input(
        label = "Enter the person's employee length:",
        min_value = 0.0,
        max_value = 50.0,
        help = 'Person age from 0 to 50'
    )

    loan_int_rate = st.number_input(
        label = "Enter the number of loan interest rate:",
        min_value = 0.0,
        max_value = 100.0,
        help = 'Person age from 0 to 100'
    )

    loan_percent_income = st.number_input(
        label = "Enter the number of loan percent income:",
        min_value = 0.0,
        max_value = 1.0,
        help = 'Person age from 0 to 1'
    )

    person_home_ownership = st.selectbox(
        label = "Enter the person's home ownership:",
        options = (
            "RENT", 
            "OWN", 
            "MORTGAGE", 
            "OTHER"
        ),
        help = 'The value are rent, own, mortgage or other'
    )

    loan_intent = st.selectbox(
        label = "Enter the status of loan intent:",
        options = (
            "PERSONAL", 
            "EDUCATION", 
            "MEDICAL", 
            "VENTURE", 
            "HOMEIMPROVEMENT", 
            "DEBTCONSOLIDATION"
        ),
        help = 'The value are personal, education, medical, venture, home improvement or debt consolidation'
    )

    loan_grade = st.selectbox(
        label = "Enter the status of loan grade:",
        options = ('A', 'B', 'C', 'D', 'E', 'F', 'G'),
        help = 'The value are A, B, C, D, E, F, or G'
    )

    cb_person_default_on_file = st.selectbox(
        label = "Enter the status of historical default:",
        options = ('Y', 'N'),
        help = 'The value are Yes or No'
    )

    submitted = st.form_submit_button("Predict")
    
    if submitted:
        raw_data = {
            'person_age' : person_age, 
            'person_income' : person_income,
            'loan_amnt' : loan_amnt,
            'cb_person_cred_hist_length' : cb_person_cred_hist_length,
            'person_emp_length' : person_emp_length,
            'loan_int_rate' : loan_int_rate,
            'loan_percent_income' : loan_percent_income,
            'person_home_ownership' : person_home_ownership,
            'loan_intent' : loan_intent,
            'loan_grade' : loan_grade,
            'cb_person_default_on_file' : cb_person_default_on_file
        }
        
        with st.spinner("Sending data to the API service..."):
            res = requests.post("http://127.0.0.1:8000/predict", json=raw_data).json()
            
        if res["error_msg"]:
            st.error(f"Error: {res['error_msg']}")
        else:
            if res["res"] == "Found API":
                st.success("Prediction Successful!")
                st.write(f"Predicted Credit Risk: {res['credit_risk_predict']}")
            else:
                st.error("Prediction Failed!")