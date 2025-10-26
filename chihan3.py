import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="AI Demand Forecasting - Cihan Food", layout="wide")

# Title
st.title("📊 Intelligent Demand Forecasting System – Food Sector (Cihan Food)")
st.markdown("""
An interactive model demonstrating how **Artificial Intelligence** can help **Cihan Food Group**
accurately forecast **daily / weekly / monthly** demand for food products.
""")

# Load data
df = pd.read_csv("chihan10.csv")
df['ddate'] = pd.to_datetime(df['ddate'])
df = df.sort_values('ddate')

# Product selection
product = st.selectbox("🔍 Choose a product:", df['product'].unique())

# Forecast period
forecast_weeks = st.slider("⏱️ Select number of future periods:", 4, 52, 12, step=4)

# 🔹 Forecast type (Daily / Weekly / Monthly)
freq_option = st.radio(
    "📅 Choose forecast frequency:",
    options=["Daily", "Weekly", "Monthly"],
    horizontal=True
)

# Define frequency value
if freq_option == "Daily":
    freq_value = 'D'
    freq_label = "Daily"
elif freq_option == "Weekly":
    freq_value = 'W'
    freq_label = "Weekly"
else:
    freq_value = 'M'
    freq_label = "Monthly"

# Filter data up to today
today = df['ddate'].max()
product_df = df[df['product'] == product]
prophet_df = product_df[['ddate', 'sells']].rename(columns={'ddate': 'ds', 'sells': 'y'})

# Train the model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=(freq_value == 'D'))
model.fit(prophet_df)

# Create future periods
future = model.make_future_dataframe(periods=forecast_weeks, freq=freq_value)
forecast = model.predict(future)

# Prepare data
forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
    columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'}
)
actual_df = prophet_df.rename(columns={'ds': 'Date', 'y': 'Actual_Demand'})
merged_df = pd.merge(forecast_df, actual_df, on='Date', how='left')

# Future only
future_only = merged_df[merged_df['Date'] > today]

# 🎨 Chart
fig = go.Figure()

# Forecast zone
fig.add_vrect(
    x0=today,
    x1=future_only['Date'].max(),
    fillcolor="rgba(255, 165, 0, 0.1)",
    layer="below",
    line_width=0,
    annotation_text=f"Forecast Area ({freq_label})",
    annotation_position="top left",
    annotation_font_size=14,
    annotation_font_color="orange"
)

# Confidence interval
fig.add_trace(go.Scatter(
    x=merged_df['Date'],
    y=merged_df['Upper_Bound'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=merged_df['Date'],
    y=merged_df['Lower_Bound'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255,165,0,0.2)',
    line=dict(width=0),
    name='Confidence Interval'
))

# Actual demand
fig.add_trace(go.Scatter(
    x=merged_df['Date'],
    y=merged_df['Actual_Demand'],
    mode='lines+markers',
    name='Actual Demand',
    line=dict(color='royalblue', width=3)
))

# Forecast demand
fig.add_trace(go.Scatter(
    x=merged_df['Date'],
    y=merged_df['Forecast'],
    mode='lines',
    name='Forecasted Demand',
    line=dict(color='darkorange', width=3, dash='dot')
))

fig.update_layout(
    title=f"📈 {freq_label} Demand Forecast starting from {today.date()} – {product}",
    template='plotly_white',
    hovermode="x unified",
    xaxis_title=f"🗓️ Date ({freq_label})",
    yaxis_title="📦 Units Sold",
    plot_bgcolor="#fafafa",
    paper_bgcolor="#ffffff",
    margin=dict(l=20, r=20, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)

# 🔹 Future forecast table
table_df = future_only.copy()
table_df['Change_%'] = table_df['Forecast'].pct_change() * 100

def get_direction(change):
    if pd.isna(change):
        return "—"
    elif change > 3:
        return "⬆️ Increase"
    elif change < -3:
        return "⬇️ Decrease"
    else:
        return "➖ Stable"

def get_recommendation(change):
    if pd.isna(change):
        return "—"
    elif change > 3:
        return "Increase stock or promotions"
    elif change < -3:
        return "Reduce production or purchases"
    else:
        return "Maintain current level"

table_df['Trend'] = table_df['Change_%'].apply(get_direction)
table_df['Recommendation'] = table_df['Change_%'].apply(get_recommendation)

def color_trend(row):
    if "⬆️" in row['Trend']:
        color = "green"
    elif "⬇️" in row['Trend']:
        color = "red"
    else:
        color = "gray"
    return f"<span style='color:{color}; font-weight:bold;'>{row['Trend']}</span>"

table_df['Trend'] = table_df.apply(color_trend, axis=1)
table_df['Change_%'] = table_df['Change_%'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

st.subheader(f"📊 Future Forecasts ({freq_label}) After Today")
st.markdown(
    table_df[['Date', 'Forecast', 'Change_%', 'Trend', 'Recommendation']].to_html(escape=False, index=False),
    unsafe_allow_html=True
)
