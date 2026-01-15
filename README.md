# 🏦 Loan Approval Prediction System (ML + FastAPI)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Production--Ready-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

An **end-to-end Machine Learning project** for predicting loan approval status.
Designed following **industry-grade ML engineering practices** with a clean pipeline, reusable inference logic, and a **production-ready FastAPI service**.

---

## 📑 Table of Contents

* [Features](#-features)
* [Project Structure](#-project-structure)
* [Dataset](#-dataset)
* [ML Pipeline](#-ml-pipeline)
* [How to Run](#-how-to-run-the-project)
* [Train the Model](#-train-the-model)
* [Run the API](#-run-the-api)
* [API Testing](#-test-the-api-swagger-ui)
* [Design Principles](#-design-principles-followed)
* [Tech Stack](#-tech-stack)
* [Future Improvements](#-future-improvements)
* [Author](#-author)

---

## 🚀 Features

* ✅ Clean and scalable project structure
* ✅ Separation of training and inference logic
* ✅ Proper data validation & feature engineering
* ✅ Missing value handling (robust & consistent)
* ✅ Trained ML model (XGBoost compatible)
* ✅ REST API using FastAPI (`POST /predict`)
* ✅ Interactive Swagger UI for testing

---

## 📁 Project Structure

```text
loan-approval-ml/
│
├── data/
│   ├── raw/                # Original Kaggle CSV (never modified)
│   └── processed/          # Cleaned & validated data
│
├── src/
│   ├── data_validation.py      # Dataset-level checks & cleaning
│   ├── feature_engineering.py  # Encoding, imputers, scaling
│   ├── train.py                # Model training & saving artifacts
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
```

---

## 📊 Dataset

* **Source:** Kaggle Loan Prediction Dataset
  [https://www.kaggle.com/competitions/playground-series-s4e10/data?select=train.csv](https://www.kaggle.com/competitions/playground-series-s4e10/data?select=train.csv)
* **Rows:** 614
* **Target Column:** `Loan_Status` (Approved / Rejected)

### 🔒 Data Handling Rules

* Raw data is **never modified**
* Cleaned data is saved in `data/processed/`
* No information leakage from test to train

---

## 🔄 ML Pipeline

```text
Raw Data
   ↓
Data Validation
   ↓
Cleaned Data
   ↓
Feature Engineering
(Imputation + Encoding + Scaling)
   ↓
Model Training
   ↓
Save Artifacts
   ↓
Prediction Logic (predict.py)
   ↓
FastAPI Service
```

---

## 🧪 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd loan-approval-ml
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python src/train.py
```

### 📦 Generated Artifacts

* `models/model.pkl`
* `models/one_hot_encoder.pkl`

---

## 🔍 Test Prediction Locally (Without API)

```bash
python src/predict.py
```

Uses the `__main__` block with a sample input for quick testing.

---

## 🌐 Run the API

> ⚠️ **Important:** Create an empty `__init__.py` file inside the `src/` directory.

```bash
uvicorn api.main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

## 🧪 Test the API (Swagger UI)

Open in browser:

```
http://127.0.0.1:8000/docs
```

### 📥 Sample Request

```json
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
```

### 📤 Sample Response

```json
{
  "loan_status": "Approved",
  "confidence": 0.87
}
```

---

## 🔐 Design Principles Followed

* Separation of concerns (data, features, inference, API)
* No data leakage (fit only on training data)
* Reusable & testable prediction logic
* API layer contains **zero ML logic**

---

## 🛠 Tech Stack

* **Language:** Python
* **Data:** Pandas, NumPy
* **ML:** Scikit-learn, XGBoost
* **API:** FastAPI
* **Server:** Uvicorn

---

## 📌 Future Improvements

* Dockerize the application
* Add `/health` endpoint
* Centralized logging & monitoring
* Model explainability using SHAP
* CI/CD pipeline integration

---

## 👨‍🎓 Author

**Amit Roy Antu**
University Student | ML & Backend Enthusiast