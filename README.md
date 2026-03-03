📈 Walmart Department Sales Forecasting

🔍 Overview

This project builds an end-to-end department-level retail sales forecasting system using time-series feature engineering and XGBoost regression.

The model predicts the next 4 weeks of weekly sales for a selected Store–Department combination using recursive multi-step forecasting.

A live Streamlit app is deployed for interactive forecasting.


🎯 Problem Statement

Retail demand forecasting is critical for:
Inventory planning
Workforce scheduling
Supply chain optimization
Revenue planning

The objective is to predict future weekly department sales using historical sales data and external economic indicators.


📊 Dataset

Walmart Retail Dataset containing:

Store
Department
Weekly_Sales
IsHoliday
Temperature
Fuel_Price
CPI
Unemployment
MarkDown features
Store Type & Size

Time Range:
Feb 2010 – Oct 2012


🧠 Feature Engineering
1️⃣ Calendar Features

Year
Month
Week number
Day of week

2️⃣ Lag Features

lag_1 (last week)
lag_4 (last month)
lag_12 (last quarter)

3️⃣ Rolling Features

4-week rolling mean
12-week rolling mean

These capture momentum, seasonality, and short-term trends.


🤖 Model

XGBoost Regressor
Hyperparameters:

n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8

Time-based train-test split used (no data leakage).


📈 Model Performance

Metric     XGBoost	Baseline (Lag_1)
MAE	      1437	           1651
RMSE	      3106                    3776

Model improves over naive baseline by:

~13% MAE reduction
~18% RMSE reduction


🔎 Feature Importance

Top Predictors:

lag_1 (48%)
rolling_mean_4 (23%)
lag_4 (12%)

Indicates strong autoregressive sales behavior and short-term momentum dominance.


🔄 Forecasting Strategy

Recursive Multi-Step Forecasting:

Each predicted week feeds into future lag features to generate a 4-week forecast horizon.

This simulates real-world forward prediction without future data access.


🚀 Deployment

Live Streamlit App:
Select Store
Select Department
Generate 4-week forecast
View forecast table and visualization

The system dynamically:

Rebuilds lag features
Aligns model feature schema
Predicts future sales


🛠 Tech Stack

Python
Pandas
NumPy
XGBoost
Scikit-learn
Streamlit
Matplotlib


📁 Repository Structure
Walmart-Sales-Forecasting/
│
├── model/
│   ├── xgb_sales_forecast_model.pkl
│   ├── forecast_feature_columns.pkl
│
├── processed_sales_small.csv
├── app.py
├── requirements.txt
└── README.md


🏁 Key Takeaways

Implemented proper time-series validation
Built autoregressive forecasting pipeline
Compared against naive baseline
Deployed live forecasting application
Avoided data leakage in lag construction


📌 Future Improvements

Hyperparameter tuning with time-series CV
Add weighted MAE (holiday emphasis)
Extend to multi-store batch forecasting
Incorporate Prophet or LSTM comparison
