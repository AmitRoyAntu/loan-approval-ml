Loan Approval Prediction System (ML + FastAPI)

An end-to-end Machine Learning system for predicting loan approval status. This project follows industry-standard ML engineering practices, including data validation, feature engineering, model training, inference, and a production-ready REST API using FastAPI.

⸻

🚀 Features
	•	Clean and scalable project structure (training vs inference separation)
	•	Proper data validation & feature engineering pipeline
	•	Trained ML model (XGBoost-compatible)
	•	Robust handling of missing values
	•	FastAPI-based REST API (POST /predict)
	•	Swagger UI for easy testing and experimentation

⸻

📁 Project Structure

loan-approval-ml/
│
├── data/
│   ├── raw/                # Original Kaggle CSV (never modified)
│   └── processed/          # Cleaned and validated dataset
│
├── src/
│   ├── data_validation.py      # Dataset-level cleaning & checks
│   ├── feature_engineering.py  # Encoding, imputers, scaling
│   ├── train.py                # Model training & artifact saving
│   └── predict.py              # Inference logic (reused by API)
│
├── models/
│   ├── model.pkl               # Trained ML model
│   ├── one_hot_encoder.pkl     # Saved encoder
│
├── api/
│   └── main.py                 # FastAPI application
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── requirements.txt
├── README.md


⸻

📊 Dataset
	•	Source: Kaggle Loan Prediction Dataset
https://www.kaggle.com/competitions/playground-series-s4e10/data?select=train.csv
	•	Rows: 614
	•	Target Column: Loan_Status (Approved / Rejected)

Important Notes
	•	Raw data is never modified
	•	Cleaned and validated data is stored in data/processed/

⸻

🔄 ML Pipeline Flow

Raw Data
   ↓
Data Validation
   ↓
Cleaned Data
   ↓
Feature Engineering + Imputation + Scaling
   ↓
Train Model
   ↓
Save Artifacts (model, encoders, imputers)
   ↓
Prediction (predict.py)
   ↓
API (FastAPI)


⸻

🧪 How to Run the Project

1️⃣ Clone & Setup

git clone <your-repo-url>
cd loan-approval-ml

2️⃣ Install Dependencies

pip install -r requirements.txt


⸻

🏋️ Train the Model

python src/train.py

This will generate:
	•	models/model.pkl
	•	models/one_hot_encoder.pkl

⸻

🔍 Test Prediction Locally (Without API)

python src/predict.py

This uses the __main__ block inside predict.py with a sample input to verify inference logic.

⸻

🌐 Run the API

Important: Create an empty __init__.py file inside the src/ folder before running the API.

uvicorn api.main:app --reload

API will be available at:

http://127.0.0.1:8000


⸻

🧪 Test the API (Swagger UI)

Open in browser:

http://127.0.0.1:8000/docs

Sample Request

{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": null,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}

Sample Response

{
  "loan_status": "Approved",
  "confidence": 0.87
}


⸻

🔐 Design Principles Followed
	•	Separation of concerns (validation vs features vs inference)
	•	No data leakage (fit only on training data)
	•	Reusable and testable prediction logic (predict.py)
	•	API contains no ML logic, only orchestration

⸻

🛠 Tech Stack
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn / XGBoost
	•	FastAPI
	•	Uvicorn

⸻

📌 Future Improvements
	•	Dockerize the application
	•	Add /health endpoint
	•	Logging & monitoring
	•	Model explainability (SHAP)
	•	CI/CD pipeline

⸻

👨‍🎓 Author

Amit Roy Antu
University Student | ML & Backend Enthusiast