import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏥 Disease Prediction System")

st.write("""
This system predicts diabetes risk using Machine Learning.
""")

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.header("Enter Patient Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider(
    "Age",
    1,
    100,
    30
)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.sidebar.selectbox(
    "Smoking History",
    [
        "never",
        "former",
        "current",
        "not current"
    ]
)

bmi = st.sidebar.slider(
    "BMI",
    10.0,
    50.0,
    25.0
)

hba1c = st.sidebar.slider(
    "HbA1c Level",
    3.0,
    10.0,
    5.5
)

glucose = st.sidebar.slider(
    "Blood Glucose Level",
    50,
    300,
    100
)

predict = st.sidebar.button(
    "Predict Disease"
)

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

gender_value = 1 if gender == "Male" else 0

smoking_map = {
    "never": 0,
    "former": 1,
    "current": 2,
    "not current": 3
}

smoking_value = smoking_map[
    smoking_history
]

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict:

    input_data = np.array([
        [
            gender_value,
            age,
            hypertension,
            heart_disease,
            smoking_value,
            bmi,
            hba1c,
            glucose
        ]
    ])

    scaled_input = scaler.transform(
        input_data
    )

    prediction = model.predict(
        scaled_input
    )[0]

    probability = model.predict_proba(
        scaled_input
    )[0][1]

    # ---------------------------------------------------
    # DISPLAY DETAILS
    # ---------------------------------------------------

    st.subheader("📌 Patient Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Gender", gender)

    with col2:
        st.metric("Age", age)

    with col3:
        st.metric("BMI", bmi)

    with col4:
        st.metric(
            "Glucose",
            glucose
        )

    # ---------------------------------------------------
    # RESULT
    # ---------------------------------------------------

    st.subheader("🎯 Prediction Result")

    if prediction == 1:

        st.error(
            "High Risk of Diabetes Detected"
        )

    else:

        st.success(
            "Low Risk of Diabetes"
        )

    st.info(
        f"Prediction Confidence: "
        f"{probability * 100:.2f}%"
    )

    # ---------------------------------------------------
    # RISK CHART
    # ---------------------------------------------------

    st.subheader("📊 Risk Analysis")

    fig, ax = plt.subplots(figsize=(6,4))

    labels = [
        "Low Risk",
        "High Risk"
    ]

    values = [
        1 - probability,
        probability
    ]

    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title("Disease Risk Probability")

    st.pyplot(fig)

    # ---------------------------------------------------
    # HEALTH INSIGHTS
    # ---------------------------------------------------

    st.subheader("🩺 Health Insights")

    if glucose > 180:
        st.warning(
            "High blood glucose detected."
        )

    if bmi > 30:
        st.warning(
            "BMI indicates obesity risk."
        )

    if hypertension == 1:
        st.warning(
            "Patient has hypertension."
        )

    if heart_disease == 1:
        st.warning(
            "Heart disease risk present."
        )

