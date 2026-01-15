import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder

# to stop auto convert of data type in fillna
pd.set_option('future.no_silent_downcasting', True)


def feature_Engineering(df, force_refit=False):
    
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    
    target = None
    if 'Loan_Status' in df.columns:
        target = df['Loan_Status']
        df = df.drop(columns=['Loan_Status'])
        
 
    df['Credit_History'] = df['Credit_History'].astype('object')
    df['ApplicantIncome'] = df['ApplicantIncome'].astype('float')


    categorical = list(df.select_dtypes(include='object').columns)
    numerical = list(df.select_dtypes(include='number').columns)
    
    
    for col in categorical:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    

    for col in numerical:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Path for OneHotEncoder model
    model_path = '/Users/amitroy/Code/ML_Projects/loan-approval-ml/models/one_hot_encoder.pkl'


    if os.path.exists(model_path) and (force_refit == False):
        ohe = joblib.load(model_path)
        ohe_categorical = pd.DataFrame(ohe.transform(df[categorical]))
    else:
        ohe = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore',
        )
        ohe_categorical = pd.DataFrame(ohe.fit_transform(df[categorical]))
        
        joblib.dump(ohe, model_path)

    ohe_categorical.index = df.index

    df = df.drop(columns=categorical, axis=1)

    df = pd.concat([df, ohe_categorical], axis=1)

    df.columns = df.columns.astype(str)
    
    # Add back target variable if it was present
    if target is not None:
        df['Loan_Status'] = target
    
    return df