import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide", page_icon="📈")

# ─────────────────────────────────────────
# Dark Theme CSS
# ─────────────────────────────────────────
st.markdown("""
<style>

.stApp {
    background-color:#0e1117;
    color:white;
}

.metric-card {
    background:#161b22;
    border-radius:10px;
    padding:16px 20px;
    text-align:center;
}

.metric-label {
    font-size:13px;
    color:#9ca3af;
}

.metric-value {
    font-size:26px;
    font-weight:700;
    color:#00C2FF;
}

.metric-sub {
    font-size:12px;
    color:#6b7280;
}

.section-header {
    font-size:20px;
    font-weight:700;
    border-left:4px solid #00C2FF;
    padding-left:10px;
    margin:20px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)

st.title("📈 Walmart Department Sales Forecasting")
st.caption("Retail demand forecasting using XGBoost · What-If Scenarios · Department Rankings")

# ─────────────────────────────────────────
# Load Data & Model
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("processed_sales_small.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_model():
    model = joblib.load("model/xgb_sales_forecast_model.pkl")
    feature_cols = joblib.load("model/forecast_feature_columns.pkl")
    return model, feature_cols

df = load_data()
model, feature_columns = load_model()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:

    st.header("🎛️ Forecast Controls")

    store_list = sorted(df["Store"].unique())
    store_id = st.selectbox("Select Store", store_list)

    dept_list = sorted(df[df["Store"] == store_id]["Dept"].unique())
    dept_id = st.selectbox("Select Department", dept_list)

    n_weeks = st.radio("Forecast Horizon", [4,8,12], horizontal=True)

    st.markdown("---")

    st.subheader("🎉 Holiday Weeks")

    holiday_flags = []
    for w in range(1,n_weeks+1):
        flag = st.checkbox(f"Week {w} is Holiday", value=False)
        holiday_flags.append(int(flag))

    st.markdown("---")
    st.subheader("🔧 What-If Scenario")

    hist_store = df[(df["Store"]==store_id) & (df["Dept"]==dept_id)]

    def safe_mean(col, default=0):
        if col in hist_store.columns:
            val = hist_store[col].dropna()
            return float(val.mean()) if len(val)>0 else default
        return default

    markdown_override = st.slider(
        "Total Markdown ($)",
        0,50000,
        int(safe_mean("total_markdown",5000)),
        step=500
    )

    temp_override = st.slider(
        "Temperature (°F)",
        0,110,
        int(safe_mean("Temperature",60))
    )

    fuel_override = st.slider(
        "Fuel Price ($/gal)",
        2.0,6.0,
        round(safe_mean("Fuel_Price",3.5),2),
        step=0.05
    )

    cpi_override = st.slider(
        "CPI",
        120.0,230.0,
        round(safe_mean("CPI",180.0),1),
        step=0.5
    )

    run_forecast = st.button("🚀 Generate Forecast", use_container_width=True)

# ─────────────────────────────────────────
# Forecast Function
# ─────────────────────────────────────────
def forecast_n_weeks(store_id, dept_id, n_weeks, holiday_flags,
                     markdown_val,temp_val,fuel_val,cpi_val):

    history = df[(df["Store"]==store_id) & (df["Dept"]==dept_id)].copy()
    history = history.sort_values("Date").tail(30)

    if len(history)<12:
        return None,"Not enough historical data."

    predictions=[]

    for i in range(n_weeks):

        last_row = history.iloc[-1].copy()
        new_row = last_row.copy()

        new_row["Date"] = new_row["Date"] + pd.Timedelta(days=7)

        new_row["Year"] = new_row["Date"].year
        new_row["Month"] = new_row["Date"].month
        new_row["Week"] = new_row["Date"].isocalendar()[1]
        new_row["DayOfWeek"] = new_row["Date"].dayofweek

        new_row["lag_1"] = history.iloc[-1]["Weekly_Sales"]
        new_row["lag_4"] = history.iloc[-4]["Weekly_Sales"]
        new_row["lag_12"] = history.iloc[-12]["Weekly_Sales"]

        new_row["rolling_mean_4"] = history.tail(4)["Weekly_Sales"].mean()
        new_row["rolling_mean_12"] = history.tail(12)["Weekly_Sales"].mean()

        new_row["IsHoliday"] = holiday_flags[i]

        new_row["total_markdown"] = markdown_val
        new_row["Temperature"] = temp_val
        new_row["Fuel_Price"] = fuel_val
        new_row["CPI"] = cpi_val

        X_input = pd.DataFrame([new_row]).drop(columns=["Weekly_Sales"],errors="ignore")
        X_input = pd.get_dummies(X_input, columns=["Type"], drop_first=True)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

        pred = max(0,float(model.predict(X_input)[0]))

        new_row["Weekly_Sales"] = pred

        predictions.append(new_row)

        history = pd.concat([history,pd.DataFrame([new_row])])

    result = pd.DataFrame(predictions)[["Date","Weekly_Sales","IsHoliday"]]

    return result,None


# ─────────────────────────────────────────
# Main Output
# ─────────────────────────────────────────
if run_forecast:

    forecast_df,error = forecast_n_weeks(
        store_id,dept_id,n_weeks,holiday_flags,
        markdown_override,temp_override,fuel_override,cpi_override
    )

    if error:
        st.error(error)

    else:

        history = df[(df["Store"]==store_id) & (df["Dept"]==dept_id)]
        history = history.sort_values("Date").tail(16)

        rolling_std = history["Weekly_Sales"].std()

        upper = forecast_df["Weekly_Sales"] + rolling_std
        lower = (forecast_df["Weekly_Sales"] - rolling_std).clip(lower=0)

        st.markdown(
        f'<div class="section-header">Store {store_id} · Dept {dept_id} Forecast</div>',
        unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history["Date"],
            y=history["Weekly_Sales"],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(color="#00C2FF", width=3),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255,176,0,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="Confidence Band"
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Weekly_Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#FFB000", width=3, dash="dash"),
            marker=dict(size=7)
        ))

        fig.update_layout(

            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",

            font=dict(color="white"),

            xaxis_title="Date",
            yaxis_title="Weekly Sales ($)",

            hovermode="x unified",
            height=450
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="white"
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            tickprefix="$",
            linecolor="white"
        )

        st.plotly_chart(fig,use_container_width=True)

else:

    st.info("👈 Configure settings and click Generate Forecast")

    hist_preview = df[(df["Store"]==1)&(df["Dept"]==1)].sort_values("Date").tail(16)

    fig0 = px.line(hist_preview,
                   x="Date",
                   y="Weekly_Sales",
                   title="Historical Sales Preview")

    fig0.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white")
    )

    fig0.update_traces(line_color="#00C2FF")

    st.plotly_chart(fig0,use_container_width=True)
