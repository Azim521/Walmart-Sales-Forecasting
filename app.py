import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide", page_icon="📈")

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #555; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1f77b4; }
    .metric-sub   { font-size: 12px; color: #888; margin-top: 2px; }
    .section-header {
        font-size: 20px; font-weight: 700;
        border-left: 4px solid #1f77b4;
        padding-left: 10px; margin: 20px 0 10px 0;
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
# Sidebar — Controls
# ─────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Forecast Controls")

    store_list = sorted(df["Store"].unique())
    store_id = st.selectbox("Select Store", store_list)

    dept_list = sorted(df[df["Store"] == store_id]["Dept"].unique())
    dept_id = st.selectbox("Select Department", dept_list)

    n_weeks = st.radio("Forecast Horizon", [4, 8, 12], horizontal=True)

    st.markdown("---")
    st.subheader("🎉 Holiday Weeks")
    st.caption("Mark upcoming weeks as holiday to see the impact.")
    holiday_flags = []
    for w in range(1, n_weeks + 1):
        flag = st.checkbox(f"Week {w} is a Holiday", value=False)
        holiday_flags.append(int(flag))

    st.markdown("---")
    st.subheader("🔧 What-If Scenario")
    st.caption("Adjust these to simulate promotions or external changes.")

    # Get historical averages for defaults
    hist_store = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)]

    def safe_mean(col, default=0.0):
        if col in hist_store.columns:
            val = hist_store[col].dropna()
            return float(val.mean()) if len(val) > 0 else default
        return default

    markdown_override = st.slider(
        "Total Markdown ($)",
        min_value=0, max_value=50000,
        value=int(safe_mean("total_markdown", 5000)),
        step=500
    )
    temp_override = st.slider(
        "Temperature (°F)",
        min_value=0, max_value=110,
        value=int(safe_mean("Temperature", 60))
    )
    fuel_override = st.slider(
        "Fuel Price ($/gal)",
        min_value=2.0, max_value=6.0,
        value=round(safe_mean("Fuel_Price", 3.5), 2),
        step=0.05
    )
    cpi_override = st.slider(
        "CPI",
        min_value=120.0, max_value=230.0,
        value=round(safe_mean("CPI", 180.0), 1),
        step=0.5
    )

    run_forecast = st.button("🚀 Generate Forecast", use_container_width=True)

# ─────────────────────────────────────────
# Forecast Logic
# ─────────────────────────────────────────
def forecast_n_weeks(store_id, dept_id, n_weeks, holiday_flags,
                     markdown_val, temp_val, fuel_val, cpi_val):

    history = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
    history = history.sort_values("Date").tail(30)

    if len(history) < 12:
        return None, "Not enough historical data for this Store-Department combination."

    predictions = []

    for i in range(n_weeks):
        last_row = history.iloc[-1].copy()
        new_row = last_row.copy()

        new_row["Date"] = new_row["Date"] + pd.Timedelta(days=7)
        new_row["Year"]      = new_row["Date"].year
        new_row["Month"]     = new_row["Date"].month
        new_row["Week"]      = new_row["Date"].isocalendar()[1]
        new_row["DayOfWeek"] = new_row["Date"].dayofweek

        # Lags
        new_row["lag_1"] = history.iloc[-1]["Weekly_Sales"]
        new_row["lag_4"] = history.iloc[-4]["Weekly_Sales"] if len(history) >= 4  else history["Weekly_Sales"].mean()
        new_row["lag_12"] = history.iloc[-12]["Weekly_Sales"] if len(history) >= 12 else history["Weekly_Sales"].mean()

        # Rolling means
        new_row["rolling_mean_4"]  = history.tail(4)["Weekly_Sales"].mean()
        new_row["rolling_mean_12"] = history.tail(12)["Weekly_Sales"].mean()

        # Holiday & what-if overrides
        new_row["IsHoliday"] = holiday_flags[i]
        if "holiday_sales_boost" in feature_columns:
            new_row["holiday_sales_boost"] = holiday_flags[i] * new_row["lag_1"]
        if "total_markdown" in feature_columns:
            new_row["total_markdown"] = markdown_val
        if "Temperature" in new_row.index:
            new_row["Temperature"] = temp_val
        if "Fuel_Price" in new_row.index:
            new_row["Fuel_Price"] = fuel_val
        if "CPI" in new_row.index:
            new_row["CPI"] = cpi_val

        X_input = pd.DataFrame([new_row]).drop(columns=["Weekly_Sales"], errors="ignore")
        X_input = pd.get_dummies(X_input, columns=["Type"], drop_first=True)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

        pred = max(0.0, float(model.predict(X_input)[0]))

        new_row["Weekly_Sales"] = round(pred, 2)
        new_row["IsHoliday_Week"] = bool(holiday_flags[i])
        predictions.append(new_row)

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    result = pd.DataFrame(predictions)[["Date", "Weekly_Sales", "IsHoliday_Week"]].reset_index(drop=True)
    result["Week #"] = [f"Week {i+1}" for i in range(n_weeks)]
    return result, None


def get_model_metrics():
    """Compute metrics on the test slice of the loaded data."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    test = df.dropna(subset=["lag_1", "Weekly_Sales"]).copy()
    if len(test) == 0:
        return None
    X = pd.get_dummies(test.drop(columns=["Weekly_Sales", "Date"], errors="ignore"),
                       columns=["Type"], drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    y = test["Weekly_Sales"]
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mask = y != 0
    mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
    avg_sales = y.mean()
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Avg Sales": avg_sales}


# ─────────────────────────────────────────
# Model Metrics Row (always visible)
# ─────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

metrics = get_model_metrics()
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">${metrics['MAE']:,.0f}</div>
            <div class="metric-sub">Mean Absolute Error</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">${metrics['RMSE']:,.0f}</div>
            <div class="metric-sub">Root Mean Sq. Error</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">MAPE</div>
            <div class="metric-value">{metrics['MAPE']:.1f}%</div>
            <div class="metric-sub">Mean Abs % Error</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Weekly Sales</div>
            <div class="metric-value">${metrics['Avg Sales']:,.0f}</div>
            <div class="metric-sub">Dataset baseline</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────
# Main Forecast Output
# ─────────────────────────────────────────
if run_forecast:
    forecast_df, error = forecast_n_weeks(
        store_id, dept_id, n_weeks, holiday_flags,
        markdown_override, temp_override, fuel_override, cpi_override
    )

    if error:
        st.error(error)
    else:
        # ── Historical data ──
        history = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
        history = history.sort_values("Date").tail(16)

        rolling_std = history["Weekly_Sales"].std()
        upper = forecast_df["Weekly_Sales"] + rolling_std
        lower = (forecast_df["Weekly_Sales"] - rolling_std).clip(lower=0)

        # ── Forecast Chart ──
        st.markdown(f'<div class="section-header">📉 Store {store_id} · Dept {dept_id} · {n_weeks}-Week Forecast</div>',
                    unsafe_allow_html=True)

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=history["Date"], y=history["Weekly_Sales"],
            mode="lines+markers", name="Historical Sales",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5)
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="Confidence Band"
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Weekly_Sales"],
            mode="lines+markers", name="Forecast",
            line=dict(color="orange", width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond")
        ))

        # Holiday markers
        holiday_dates = forecast_df[forecast_df["IsHoliday_Week"] == True]["Date"]
        for hdate in holiday_dates:
            fig.add_vline(x=hdate, line_dash="dot", line_color="red",
                          annotation_text="🎉 Holiday", annotation_position="top right")

        fig.update_layout(
            xaxis_title="Date", yaxis_title="Weekly Sales ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified", height=420,
            margin=dict(t=40, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee")
        fig.update_yaxes(showgrid=True, gridcolor="#eee", tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

        # ── Forecast Table + Download ──
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown('<div class="section-header">📋 Forecast Table</div>', unsafe_allow_html=True)
            display_df = forecast_df[["Week #", "Date", "Weekly_Sales", "IsHoliday_Week"]].copy()
            display_df.columns = ["Week", "Date", "Predicted Sales ($)", "Holiday?"]
            display_df["Predicted Sales ($)"] = display_df["Predicted Sales ($)"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv_data = forecast_df[["Week #", "Date", "Weekly_Sales", "IsHoliday_Week"]].to_csv(index=False)
            st.download_button(
                "⬇️ Download Forecast CSV",
                data=csv_data,
                file_name=f"forecast_store{store_id}_dept{dept_id}_{n_weeks}w.csv",
                mime="text/csv"
            )

        with col_right:
            st.markdown('<div class="section-header">💰 Revenue Summary</div>', unsafe_allow_html=True)
            total_forecast = forecast_df["Weekly_Sales"].sum()
            avg_forecast   = forecast_df["Weekly_Sales"].mean()
            hist_avg       = history["Weekly_Sales"].mean()
            delta_pct      = ((avg_forecast - hist_avg) / hist_avg * 100) if hist_avg != 0 else 0

            st.metric("Total Projected Revenue", f"${total_forecast:,.0f}")
            st.metric("Avg Weekly Sales (Forecast)", f"${avg_forecast:,.0f}")
            st.metric("vs Historical Avg", f"${hist_avg:,.0f}", delta=f"{delta_pct:+.1f}%")

        # ── Department Rankings ──
        st.markdown(f'<div class="section-header">🏬 All Departments — Store {store_id} Projected Sales Ranking</div>',
                    unsafe_allow_html=True)
        st.caption("4-week projected total sales across all departments in this store.")

        dept_forecasts = []
        all_depts = sorted(df[df["Store"] == store_id]["Dept"].unique())

        prog = st.progress(0, text="Computing department rankings...")
        for idx, d in enumerate(all_depts):
            try:
                f, _ = forecast_n_weeks(
                    store_id, d, 4,
                    [0, 0, 0, 0],  # no holiday override for ranking
                    markdown_override, temp_override, fuel_override, cpi_override
                )
                if f is not None:
                    dept_forecasts.append({
                        "Department": int(d),
                        "4-Week Forecast ($)": round(f["Weekly_Sales"].sum(), 0),
                        "Avg/Week ($)": round(f["Weekly_Sales"].mean(), 0)
                    })
            except Exception:
                pass
            prog.progress((idx + 1) / len(all_depts), text=f"Computing dept {d}...")

        prog.empty()

        if dept_forecasts:
            rank_df = pd.DataFrame(dept_forecasts).sort_values("4-Week Forecast ($)", ascending=False).reset_index(drop=True)
            rank_df.index += 1

            # Highlight selected dept
            colors = ["orange" if int(d) == int(dept_id) else "#1f77b4"
                      for d in rank_df["Department"]]

            fig2 = go.Figure(go.Bar(
                x=rank_df["Department"].astype(str),
                y=rank_df["4-Week Forecast ($)"],
                marker_color=colors,
                text=rank_df["4-Week Forecast ($)"].apply(lambda x: f"${x:,.0f}"),
                textposition="outside"
            ))
            fig2.update_layout(
                xaxis_title="Department", yaxis_title="4-Week Projected Sales ($)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=20, b=40),
                yaxis=dict(tickprefix="$", showgrid=True, gridcolor="#eee")
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("📄 View Full Rankings Table"):
                st.dataframe(rank_df, use_container_width=True)

else:
    # ── Placeholder before forecast runs ──
    st.info("👈 Configure your forecast settings in the sidebar and click **Generate Forecast** to begin.")

    hist_preview = df[(df["Store"] == store_list[0]) & (df["Dept"] == sorted(df["Dept"].unique())[0])].sort_values("Date").tail(16)
    if len(hist_preview) > 0:
        fig0 = px.line(hist_preview, x="Date", y="Weekly_Sales",
                       title="Historical Sales Preview (Store 1, Dept 1)",
                       labels={"Weekly_Sales": "Weekly Sales ($)"})
        fig0.update_traces(line_color="#1f77b4")
        fig0.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           yaxis=dict(tickprefix="$"))
        st.plotly_chart(fig0, use_container_width=True)
