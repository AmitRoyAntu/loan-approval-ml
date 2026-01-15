import pandas as pd
import joblib

from src.feature_engineering import feature_Engineering

model_path = '/Users/amitroy/Code/ML_Projects/loan-approval-ml/models/model.pkl'
model = joblib.load(model_path)

def predict_loan(input_data):
    
    df = pd.DataFrame([input_data])
    
    df = feature_Engineering(df)
    
    pred = model.predict(df)[0]
    pred_prob = model.predict_proba(df)[0][1]
    
    status = 'Approved' if pred == 1 else 'Rejected'
    approved_confidence = round(float(pred_prob), 2)
    
    return {
        'Loan_Status': status,
        "confidence": approved_confidence if status == 'Approved' else 1 - approved_confidence
    }

# if __name__ == "__main__":

    # sample = {
    #     "Gender": "Male",
    #     "Married": "No",
    #     "Dependents": "0",
    #     "Education": "Not Graduate",
    #     "Self_Employed": "No",
    #     "ApplicantIncome": 49.0,
    #     "CoapplicantIncome": 0.0,
    #     "LoanAmount": 146.41,
    #     "Loan_Amount_Term": 360.0,
    #     "Credit_History": 1.0,
    #     "Property_Area": "Urban"
    # }

    # print(predict_loan(sample))