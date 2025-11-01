# TrueVaL — AI-Powered Car Price Prediction App

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("car_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")

data = pd.read_csv("Cardetails.csv")

st.set_page_config(page_title="TrueVaL — Car Price Predictor", layout="centered")

st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">🚗 TrueVaL</h1>
    <h4 style="text-align:center; color:#555;">AI-Powered Car Price Prediction Platform</h4>
    <p style="text-align:center; color:#888; font-size:15px;">
        Enter your car details below to predict its <b>fair selling price</b> instantly.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

brands = sorted(data["name"].apply(lambda x: str(x).split()[0]).unique())
selected_brand = st.selectbox("Select Brand", brands)
models = sorted(data[data["name"].str.startswith(selected_brand)]["name"].unique())
selected_model = st.selectbox("Select Model", models)

# step sizes set for intuitive increments
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2018, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000, step=1000)
fuel = st.selectbox("Fuel Type", sorted(data["fuel"].unique()))
seller_type = st.selectbox("Seller Type", sorted(data["seller_type"].unique()))
transmission = st.selectbox("Transmission Type", sorted(data["transmission"].unique()))
owner = st.selectbox("Number of Owners", sorted(data["owner"].unique()))
mileage = st.number_input("Mileage (km/l)", min_value=1.0, value=18.0, step=1.0, format="%.1f")
engine = st.number_input("Engine (CC)", min_value=50.0, value=1200.0, step=50.0, format="%.0f")
max_power = st.number_input("Max Power (bhp)", min_value=1.0, value=80.0, step=5.0, format="%.1f")
torque = st.number_input("Torque (Nm)", min_value=0.0, value=100.0, step=10.0, format="%.1f")
seats = st.number_input("Seats", min_value=1, max_value=12, value=5, step=1)

input_dict = {
    "name": selected_model,
    "year": year,
    "selling_price": 0,
    "km_driven": km_driven,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "torque": torque,
    "seats": seats
}

for col in input_dict:
    if col in label_encoders:
        le = label_encoders[col]
        val = input_dict[col]
        if val not in le.classes_:
            val = le.classes_[0]
        input_dict[col] = le.transform([val])[0]

input_df = pd.DataFrame([input_dict]).drop(columns=["selling_price"], errors="ignore")
input_scaled = scaler.transform(input_df)
input_poly = poly.transform(input_scaled)

if st.button("🔍 Predict Price"):
    price = model.predict(input_poly)[0]
    st.success(f"💰 **Predicted Selling Price:** ₹ {price:,.2f}")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 10px; color: #666;">
        <p style="font-size:15px;"><b>TrueVaL</b> — AI-Powered Second-Hand Price Prediction Platform</p>
        <p style="font-size:13px;">
            Made with ❤️ by <b>Nirmalya Pradhan</b><br>
            © 2025 TrueVaL | All Rights Reserved
        </p>
        <p style="font-size:12px; color:#888;">
            🚧 This application is currently under active development 🚧
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
