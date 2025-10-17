import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Predictor 🚢", layout="centered")

st.title("Titanic Survival Predictor")
st.write("Predict whether a passenger would have survived the Titanic disaster.")


try:
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error(" Model files not found! Make sure logistic_model.pkl and scaler.pkl are in the same folder as app.py.")
    st.stop()


pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Ticket Fare (€)", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])


sex_encoded = 1 if sex == "Male" else 0
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success(f" Survived! (Probability: {probability:.2%})")
    else:
        st.error(f" Did not survive. (Probability: {probability:.2%})")
