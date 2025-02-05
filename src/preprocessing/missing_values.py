from sklearn.impute import SimpleImputer

def fit_imputer(data, columns_to_impute):
    # fit to train set
    imp_med = SimpleImputer(strategy='median')
    imp_med.fit(data[columns_to_impute])

    return imp_med

def transform_imputer(imputer, data, columns_to_impute):
    data.loc[:, columns_to_impute] = imputer.transform(data[columns_to_impute])
    
    return data