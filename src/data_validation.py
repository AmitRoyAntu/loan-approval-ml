import pandas as pd

Raw_Path = "/Users/amitroy/Code/ML_Projects/loan-approval-ml/data/raw/loan.csv"
Processed_Path = "/Users/amitroy/Code/ML_Projects/loan-approval-ml/data/processed/loan_cleaned.csv"

def validate_and_clean():
    
    df = pd.read_csv(Raw_Path)

    df = df.drop_duplicates()

    df = df.dropna(subset=['Loan_Status'])
    
    if not df['Loan_Status'].isin(['Y', 'N']).all():
        raise ValueError("Invalid Loan_Status values found")
    
    numeric = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    
    for income_col in numeric:
        if (df[income_col] < 0).any():
            raise ValueError(f"Negative value found in {income_col} column")

    df.to_csv(Processed_Path, index=False)
    
    print("✅ Data validation & cleaning completed")
    

if __name__ == "__main__":
    validate_and_clean()

