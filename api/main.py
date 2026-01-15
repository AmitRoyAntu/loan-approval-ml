from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal


from src.predict import predict_loan

app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0"
)


class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Optional[Literal["0", "1", "2", "3+"]] = None
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Optional[Literal["Yes", "No"]] = None

    ApplicantIncome: float
    CoapplicantIncome: float = 0.0

    LoanAmount: Optional[float] = None
    Loan_Amount_Term: Optional[float] = None
    Credit_History: Optional[Literal[0, 1]] = None

    Property_Area: Literal["Urban", "Semiurban", "Rural"]


@app.post("/predict")
def predict(data: LoanApplication):
    try:
        result = predict_loan(data.model_dump())
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

