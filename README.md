# AI Powered Fraud Monitoring System

## Project Overview

This project identifies potentially fraudulent credit card transactions using custom fraud detection rules and machine learning.

## Fraud Detection Rules

### Amount Risk

- Amount > 500 → +40 Risk Points
- Amount > 100 → +20 Risk Points
- Amount > 50 → +10 Risk Points

### Time Risk

Transactions occurring between:

- 10 PM to 3 AM → +30 Risk Points

### Category Risk

- shopping_net → +20 Risk Points
- misc_net → +15 Risk Points

## Risk Classification

- Score >= 60 → High Risk
- Score >= 30 → Medium Risk
- Score < 30 → Low Risk

## Features

- Dynamic Fraud Risk Scoring
- Fraud Category Analysis
- Fraud Hour Analysis
- Fraud State Analysis
- Fraud Merchant Analysis
- Interactive Streamlit Dashboard
- Machine Learning Fraud Prediction
- CSV Upload Support

## Machine Learning
Model Used:
- Random Forest Classifier

Features:
- Amount
- Hour
- Risk Score
- Transaction Category

Target:
- is_fraud

## Tech Stack
- Python
- Pandas
- SQL
- Streamlit
- Scikit-Learn
- Joblib

# Streamlit LINK
https://ai-fraud-monotoring-system.streamlit.app/
To see the output must dowmload the sample_fraud.csv file

## Author
Nagalla Venkata Krishna
