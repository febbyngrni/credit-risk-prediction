def filter_outlier(data):
    data_copy = data.copy()

    data_copy = data_copy[
        (data_copy['person_emp_length'] <= 35) | 
        (data_copy['person_age'] <= 70)
    ]

    return data_copy