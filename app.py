import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# 1. PAGE CONFIGURATION & DARK NEON GLASSMORPHISM THEME
st.set_page_config(
    page_title="DriveMatch India | Find Your Dream Car",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    .main-header h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .badge {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }

    .car-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 20px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 1.5rem;
    }
    .car-card:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }

    .price-tag {
        font-size: 1.8rem;
        font-weight: 800;
        color: #34d399;
    }

    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.5) !important;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# 2. LOAD DATASET & UNIFIED SINGLE PKL FILE
@st.cache_data
def load_dataset():
    return pd.read_csv("indian_cars_ml_dataset_10k.csv")

@st.cache_resource
def load_model_bundle():
    bundle = joblib.load("car_model.pkl")
    return bundle['scaler'], bundle['knn_model'], bundle['rf_price_model'], bundle['feature_columns']

df = load_dataset()
scaler, knn_model, rf_price_model, feature_columns = load_model_bundle()

# 3. HEADER SECTION
st.markdown("""
<div class="main-header">
    <h1>🏎️ DriveMatch India</h1>
    <p>AI-Powered Precision Car Matchmaker & Real-Time On-Road Price Engine</p>
    <div>
        <span class="badge">🔥 30+ Brands</span>
        <span class="badge">⚡ EV & Hybrids</span>
        <span class="badge">📊 Single 'car_model.pkl' Model</span>
        <span class="badge">🛡️ BNCAP Safety Ratings</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. SIDEBAR CONTROLS
st.sidebar.markdown("### ⚙️ Define Your Requirements")

city = st.sidebar.selectbox("📍 Select Your City / State", sorted(df['city_location'].unique()))
budget_range = st.sidebar.slider("💰 On-Road Budget Range (₹ Lakhs)", 4.0, 150.0, (8.0, 25.0), step=0.5)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    body_pref = st.selectbox("🚘 Body Style", ["Any"] + sorted(list(df['body_type'].unique())))
    transmission_pref = st.selectbox("⚙️ Gearbox", ["Any"] + sorted(list(df['transmission'].unique())))
with col_sb2:
    fuel_pref = st.selectbox("⛽ Fuel Type", ["Any"] + sorted(list(df['fuel_type'].unique())))
    seating_pref = st.selectbox("🪑 Seats", ["Any"] + [str(s) for s in sorted(df['seating_capacity'].unique())])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Driving Conditions & Priorities")
min_safety = st.sidebar.slider("🛡️ Minimum Safety Stars", 1, 5, 4)
min_gc = st.sidebar.slider("🛣️ Min Ground Clearance (mm)", 130, 230, 180, step=5)
monthly_km = st.sidebar.select_slider("📅 Monthly Driving (KM)", options=[500, 800, 1200, 1500, 2000, 2500, 3000], value=1200)

predict_btn = st.sidebar.button("✨ Find My Perfect Car")

# 5. RECOMMENDATION & INFERENCE ENGINE
if predict_btn or 'searched' not in st.session_state:
    st.session_state['searched'] = True

    filtered_df = df[
        (df['full_on_road_price_lakh'] >= budget_range[0]) &
        (df['full_on_road_price_lakh'] <= budget_range[1]) &
        (df['safety_rating_stars'] >= min_safety) &
        (df['ground_clearance_mm'] >= min_gc)
    ]

    if body_pref != "Any":
        filtered_df = filtered_df[filtered_df['body_type'] == body_pref]
    if fuel_pref != "Any":
        filtered_df = filtered_df[filtered_df['fuel_type'] == fuel_pref]
    if transmission_pref != "Any":
        filtered_df = filtered_df[filtered_df['transmission'] == transmission_pref]
    if seating_pref != "Any":
        filtered_df = filtered_df[filtered_df['seating_capacity'] == int(seating_pref)]

    if filtered_df.empty:
        st.warning("⚠️ No exact match found within strict constraints. Showing closest top options:")
        filtered_df = df[(df['full_on_road_price_lakh'] <= budget_range[1] * 1.25)]

    unique_matches = filtered_df.drop_duplicates(subset=['recommended_brand', 'recommended_model']).head(3)

    st.markdown("### 🎯 Your Top Car Recommendations")

    cols = st.columns(len(unique_matches))
    for idx, (_, car) in enumerate(unique_matches.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="car-card">
                <span class="badge">Rank #{idx+1} Match</span>
                <h3 style="margin: 10px 0 0 0; color: #f8fafc;">{car['recommended_brand']} {car['recommended_model']}</h3>
                <p style="color: #94a3b8; font-size: 0.9rem;">{car['body_type']} • {car['fuel_type']} • {car['transmission']}</p>
                <div class="price-tag">₹ {car['full_on_road_price_lakh']:.2f} Lakh</div>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 10px;">
                    <b>On-Road Target in {city}</b><br>
                    Ex-Showroom: ₹{car['ex_showroom_price_lakh']:.2f} L<br>
                    Est. RTO Tax: ₹{car['rto_tax_lakh']:.2f} L<br>
                    Insurance: ₹{car['insurance_cost_lakh']:.2f} L
                </p>
                <hr style="border-color: rgba(255,255,255,0.1)">
                <p style="font-size: 0.85rem;">
                    🛡️ Safety: <b>{car['safety_rating_stars']}★</b> | 🛣️ GC: <b>{car['ground_clearance_mm']} mm</b><br>
                    ⛽ Mileage/Range: <b>{car['mileage_kmpl_or_range_km']} {'km/l' if car['fuel_type'] != 'Electric' else 'km'}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 6. ANALYTICS & INTERACTIVE DASHBOARD
    top_car = unique_matches.iloc[0]

    tab1, tab2, tab3 = st.tabs(["📊 Price Breakdown", "⚡ Running Cost Engine", "💳 EMI Calculator"])

    with tab1:
        st.subheader(f"Breakdown for {top_car['recommended_brand']} {top_car['recommended_model']} in {city}")
        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            price_labels = ['Ex-Showroom Price', 'RTO Tax', 'Insurance', 'Other Charges']
            price_values = [top_car['ex_showroom_price_lakh'], top_car['rto_tax_lakh'], top_car['insurance_cost_lakh'], top_car['other_charges_lakh']]

            fig_pie = px.pie(
                names=price_labels,
                values=price_values,
                title="On-Road Component Split",
                hole=0.4
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            fig_bar = px.bar(
                x=unique_matches['recommended_model'],
                y=unique_matches['full_on_road_price_lakh'],
                color=unique_matches['fuel_type'],
                title="Top Matches Comparison (On-Road Price)",
                labels={'x': 'Car Model', 'y': 'Price (₹ Lakhs)'}
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff")
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("⛽ Fuel & Operating Expense Estimator")
        fuel_rates = {"Petrol": 102.0, "Diesel": 90.0, "CNG": 76.0, "Electric": 9.0, "Strong Hybrid": 102.0}
        current_fuel_price = fuel_rates.get(top_car['fuel_type'], 100.0)

        if top_car['fuel_type'] == 'Electric':
            monthly_cost = (monthly_km / 7.0) * current_fuel_price
        else:
            monthly_cost = (monthly_km / top_car['mileage_kmpl_or_range_km']) * current_fuel_price

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Est. Monthly Fuel Cost", f"₹ {monthly_cost:,.0f}")
        col_m2.metric("Annual Fuel Cost", f"₹ {monthly_cost * 12:,.0f}")
        col_m3.metric("Running Cost Per KM", f"₹ {monthly_cost / monthly_km:.2f} / km")

    with tab3:
        st.subheader("🏦 Car Loan EMI Estimator")
        col_e1, col_e2, col_e3 = st.columns(3)

        down_payment_pct = col_e1.slider("Down Payment (%)", 10, 50, 20)
        interest_rate = col_e2.slider("Loan Interest Rate (%)", 7.5, 12.0, 9.0, step=0.25)
        tenure_years = col_e3.slider("Loan Tenure (Years)", 3, 7, 5)

        principal = (top_car['full_on_road_price_lakh'] * 100000) * (1 - down_payment_pct / 100)
        r = (interest_rate / 100) / 12
        n = tenure_years * 12

        emi = (principal * r * ((1 + r)**n)) / (((1 + r)**n) - 1)

        st.success(f"💰 **Estimated Monthly EMI:** ₹ **{emi:,.0f}** / month for {tenure_years} years (Loan Principal: ₹ {principal:,.0f})")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Engineered for Indian Drivers • Powered by Machine Learning & Real-Time RTO Intelligence</p>", unsafe_allow_html=True)
