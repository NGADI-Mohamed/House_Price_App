import streamlit as st 
import joblib 
import numpy as np

model = joblib.load('House_Price_Prediction_Model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter house details:")

square_feet = st.number_input("Square Feet ", min_value=100, step=50)
num_rooms = st.number_input("Number of Rooms", min_value=1, step=1)
age = st.number_input("House Age (years)", min_value=0, step=1)
distance = st.number_input("Distance to City (km)", min_value=0.0, step=0.5)

if square_feet < 100:
    st.warning("⚠️ Square feet seems too low")

if num_rooms > square_feet / 100:
    st.warning("⚠️ Too many rooms for the given size")

if st.button("Predict Price"):
    input_data = np.array([[square_feet, num_rooms, age, distance]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Estimated House Price: {prediction[0]:,.2f} $")

st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <small>Made by NGADI MOHAMED</small>
    </div>
    """,
    unsafe_allow_html=True
)

# to run the app open terminal in the folder and type : streamlit run app.py

