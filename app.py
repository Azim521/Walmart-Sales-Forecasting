import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide")

st.title("📈 Walmart Department Sales Forecasting")
st.write("4-week recursive demand forecasting using XGBoost.")

# -----------------------------
# Load Data & Model
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("processed_sales_small.csv", parse_dates=["Date"])

@st.cache_resource
def load_model():
    model = joblib.load("model/xgb_sales_forecast_model.pkl")
    feature_cols = joblib.load("model/forecast_feature_columns.pkl")
    return model, feature_cols

df = load_data()
model, feature_columns = load_model()

# -----------------------------
# User Selection
# -----------------------------
store_list = sorted(df["Store"].unique())
dept_list = sorted(df["Dept"].unique())

store_id = st.selectbox("Select Store", store_list)
dept_id = st.selectbox("Select Department", dept_list)

# -----------------------------
# Forecast Logic
# -----------------------------
def forecast_next_4_weeks(store_id, dept_id):

    history = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
    history = history.sort_values("Date").tail(30)

    predictions = []

    for i in range(4):

        last_row = history.iloc[-1].copy()
        new_row = last_row.copy()

        new_row["Date"] = new_row["Date"] + pd.Timedelta(days=7)

        new_row["Year"] = new_row["Date"].year
        new_row["Month"] = new_row["Date"].month
        new_row["Week"] = new_row["Date"].week
        new_row["DayOfWeek"] = new_row["Date"].dayofweek

        new_row["lag_1"] = history.iloc[-1]["Weekly_Sales"]
        new_row["lag_4"] = history.iloc[-4]["Weekly_Sales"]
        new_row["lag_12"] = history.iloc[-12]["Weekly_Sales"]

        new_row["rolling_mean_4"] = history.tail(4)["Weekly_Sales"].mean()
        new_row["rolling_mean_12"] = history.tail(12)["Weekly_Sales"].mean()

        X_input = pd.DataFrame([new_row])
        X_input = X_input.drop(columns=["Weekly_Sales"])

        X_input = pd.get_dummies(X_input, columns=["Type"], drop_first=True)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict(X_input)[0]

        new_row["Weekly_Sales"] = pred
        predictions.append(new_row)

        history = pd.concat([history, pd.DataFrame([new_row])])

    return pd.DataFrame(predictions)[["Date", "Weekly_Sales"]]

# -----------------------------
# Run Forecast
# -----------------------------
if st.button("Generate 4-Week Forecast"):

    forecast_df = forecast_next_4_weeks(store_id, dept_id)

    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    # Plot
    history = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
    history = history.sort_values("Date").tail(12)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(history["Date"], history["Weekly_Sales"], label="Historical (Last 12 Weeks)")
    ax.plot(forecast_df["Date"], forecast_df["Weekly_Sales"], label="Forecast (Next 4 Weeks)", linestyle="--")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.set_title("Sales Forecast")

    st.pyplot(fig)
