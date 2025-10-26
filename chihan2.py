import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# إعداد الصفحة
st.set_page_config(page_title="AI Demand Forecasting - Cihan Food", layout="wide")

# العنوان
st.title("📊 نظام التنبؤ الذكي بالطلب – قطاع الأغذية (Cihan Food)")
st.markdown("""
نموذج تفاعلي يوضح كيف يمكن للذكاء الاصطناعي مساعدة **مجموعة جيهان فود** 
في التنبؤ بالطلب على المنتجات الغذائية على مستوى **يومي / أسبوعي / شهري** بدقة عالية.
""")

# تحميل البيانات
df = pd.read_csv("chihan10.csv")
df['ddate'] = pd.to_datetime(df['ddate'])
df = df.sort_values('ddate')

# اختيار المنتج
product = st.selectbox("🔍 اختر المنتج:", df['product'].unique())

# اختيار فترة التنبؤ
forecast_weeks = st.slider("⏱️ اختر عدد الفترات المستقبلية:", 4, 52, 12, step=4)

# 🔹 اختيار نوع التنبؤ (يومي / أسبوعي / شهري)
freq_option = st.radio(
    "📅 اختر نوع التنبؤ:",
    options=["يومي", "أسبوعي", "شهري"],
    horizontal=True
)

# تحديد قيمة freq حسب الاختيار
if freq_option == "يومي":
    freq_value = 'D'
    freq_label = "يومي"
elif freq_option == "أسبوعي":
    freq_value = 'W'
    freq_label = "أسبوعي"
else:
    freq_value = 'M'
    freq_label = "شهري"

# تصفية البيانات حتى اليوم فقط
today = df['ddate'].max()
product_df = df[df['product'] == product]
prophet_df = product_df[['ddate', 'sells']].rename(columns={'ddate': 'ds', 'sells': 'y'})

# تدريب النموذج
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=(freq_value == 'D'))
model.fit(prophet_df)

# إنشاء الفترات المستقبلية
future = model.make_future_dataframe(periods=forecast_weeks, freq=freq_value)
forecast = model.predict(future)

# تجهيز البيانات
forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
    columns={'ds': 'التاريخ', 'yhat': 'التوقع', 'yhat_lower': 'الحد_الأدنى', 'yhat_upper': 'الحد_الأعلى'}
)
actual_df = prophet_df.rename(columns={'ds': 'التاريخ', 'y': 'الطلب_الفعلي'})
merged_df = pd.merge(forecast_df, actual_df, on='التاريخ', how='left')

# فصل المستقبل فقط
future_only = merged_df[merged_df['التاريخ'] > today]

# 🎨 الرسم البياني
fig = go.Figure()

# منطقة التنبؤ
fig.add_vrect(
    x0=today,
    x1=future_only['التاريخ'].max(),
    fillcolor="rgba(255, 165, 0, 0.1)",
    layer="below",
    line_width=0,
    annotation_text=f"منطقة التنبؤ ({freq_label})",
    annotation_position="top left",
    annotation_font_size=14,
    annotation_font_color="orange"
)

# نطاق الثقة
fig.add_trace(go.Scatter(
    x=merged_df['التاريخ'],
    y=merged_df['الحد_الأعلى'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=merged_df['التاريخ'],
    y=merged_df['الحد_الأدنى'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255,165,0,0.2)',
    line=dict(width=0),
    name='نطاق الثقة'
))

# الطلب الفعلي
fig.add_trace(go.Scatter(
    x=merged_df['التاريخ'],
    y=merged_df['الطلب_الفعلي'],
    mode='lines+markers',
    name='الطلب الفعلي',
    line=dict(color='royalblue', width=3)
))

# التنبؤ
fig.add_trace(go.Scatter(
    x=merged_df['التاريخ'],
    y=merged_df['التوقع'],
    mode='lines',
    name='التنبؤ بالطلب',
    line=dict(color='darkorange', width=3, dash='dot')
))

fig.update_layout(
    title=f"📈 التنبؤ {freq_label} بالطلب ابتداءً من {today.date()} – {product}",
    template='plotly_white',
    hovermode="x unified",
    xaxis_title=f"🗓️ التاريخ ({freq_label})",
    yaxis_title="📦 الكمية المباعة (وحدة)",
    plot_bgcolor="#fafafa",
    paper_bgcolor="#ffffff",
    margin=dict(l=20, r=20, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)

# 🔹 جدول التوقعات المستقبلية
table_df = future_only.copy()
table_df['التغير_٪'] = table_df['التوقع'].pct_change() * 100

def get_direction(change):
    if pd.isna(change):
        return "—"
    elif change > 3:
        return "⬆️ زيادة"
    elif change < -3:
        return "⬇️ نقصان"
    else:
        return "➖ ثابت"

def get_recommendation(change):
    if pd.isna(change):
        return "—"
    elif change > 3:
        return "زيادة الكمية أو العروض"
    elif change < -3:
        return "تقليل الإنتاج أو المشتريات"
    else:
        return "الحفاظ على المستوى الحالي"

table_df['الاتجاه'] = table_df['التغير_٪'].apply(get_direction)
table_df['التوصية'] = table_df['التغير_٪'].apply(get_recommendation)

def color_trend(row):
    if "⬆️" in row['الاتجاه']:
        color = "green"
    elif "⬇️" in row['الاتجاه']:
        color = "red"
    else:
        color = "gray"
    return f"<span style='color:{color}; font-weight:bold;'>{row['الاتجاه']}</span>"

table_df['الاتجاه'] = table_df.apply(color_trend, axis=1)
table_df['التغير_٪'] = table_df['التغير_٪'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

st.subheader(f"📊 التوقعات المستقبلية ({freq_label}) بعد تاريخ اليوم")
st.markdown(
    table_df[['التاريخ', 'التوقع', 'التغير_٪', 'الاتجاه', 'التوصية']].to_html(escape=False, index=False),
    unsafe_allow_html=True
)
