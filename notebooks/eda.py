#!/usr/bin/env python
# coding: utf-8

# Missing values analysis
# Target imbalance check
# Income vs Loan status
# Credit history impact
# 
# Don't overdo visualizations

# In[630]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[632]:


df = pd.read_csv('/Users/amitroy/Code/ML_Projects/loan-approval-ml/data/raw/loan.csv')


# In[633]:


df


# In[634]:


df.isna().sum() 


# In[635]:


def missing_report(df):
    plt.figure(figsize=(8, 4))
    missing_pct = df.isna().sum() * 100 / len(df)
    ax = missing_pct.plot(kind='barh')
    plt.xlabel('Percentage Missing')

    # Add values on bars
    for i, v in enumerate(missing_pct):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')

    plt.tight_layout()


# In[636]:


missing_report(df)


# In[637]:


df = df.dropna(subset=['Loan_Status'])


# In[638]:


df.columns


# In[639]:


df.drop(columns=['Loan_ID'], inplace=True)


# In[640]:


df['Credit_History'] = df['Credit_History'].astype('object')


# In[641]:


df.columns


# In[642]:


missing_report(df.select_dtypes(include='object'))


# In[643]:


df['Gender'].mode()[0]


# In[644]:


# df['Gender'].fillna(df['Gender'].mode()[0]).isna().sum()
# help(df.fillna)


# In[645]:


df['Dependents'].value_counts()


# In[646]:


# For categorical values
df = df.fillna(value={
    'Property_Area': df['Property_Area'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'Education': df['Education'].mode()[0],
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],

    'Credit_History': df['Credit_History'].mode()[0],
    'Loan_Amount_Term': df['Loan_Amount_Term'].value_counts().mode()[0],
})
df.isna().sum()


# In[647]:


df['Credit_History'] = df['Credit_History'].astype('object')
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')


# In[648]:


missing_report(df.select_dtypes(include='object'));


# In[649]:


missing_report(df.select_dtypes(include='number'))


# In[650]:


df = df.fillna(value={
    'LoanAmount': df['LoanAmount'].mean(),
    'CoapplicantIncome': df['CoapplicantIncome'].mean(),
    'ApplicantIncome': df['ApplicantIncome'].mean()
})


# In[651]:


missing_report(df)


# In[652]:


df.isna().sum()


# In[653]:


df.describe()


# In[654]:


df.select_dtypes(include='object').describe()


# In[655]:


sns.countplot(df, x='Loan_Status', hue='Credit_History')


# In[665]:


sns.boxplot(df, x='Loan_Status', y='LoanAmount')


# In[667]:


sns.countplot(df,x='Loan_Status', hue='Dependents')


# In[674]:


sns.heatmap(df.select_dtypes(include='number').corr(), annot=True,)


# In[676]:


df.to_csv('../data/processed/cleaned_loan.csv')

