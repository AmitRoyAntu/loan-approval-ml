import pandas as pd
import joblib 

from src.feature_engineering import feature_Engineering

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report



df = pd.read_csv('/Users/amitroy/Code/ML_Projects/loan-approval-ml/data/processed/loan_cleaned.csv',)

# Use force_refit=True to retrain the encoder with new data
df = feature_Engineering(df=df, force_refit=True)

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status'].map({'N': 0, 'Y': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = XGBClassifier(
    n_estimators= 200,
    max_depth= 3,
    objective='binary:logistic',
    eval_metric='logloss',
    subsample=.7,
    colsample_bytree=.7,
    random_state=0
)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(classification_report(y_test,y_pred))

model.fit(X,y)
joblib.dump(model, '/Users/amitroy/Code/ML_Projects/loan-approval-ml/models/model.pkl')