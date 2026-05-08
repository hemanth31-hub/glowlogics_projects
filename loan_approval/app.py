import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- STEP 1: MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Loan Intelligence Dashboard", layout="wide")

# --- STEP 2: LOAD MODEL ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('loan_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

assets = load_assets()

# Check if model exists before proceeding
if assets is None:
    st.error("⚠️ 'loan_model.pkl' not found. Please run 'python train_model.py' in your terminal first!")
    st.stop()

model = assets['model']

# --- STEP 3: UI HEADER ---
st.title("🏦 Loan Approval & Risk Assessment System")
st.markdown("---")

# --- STEP 4: SIDEBAR (DATA INPUT MODULE) ---
st.sidebar.header("Applicant Profile")
dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
education = st.sidebar.radio("Education Status", ["Graduate", "Not Graduate"])
employed = st.sidebar.radio("Self Employed?", ["Yes", "No"])
income = st.sidebar.number_input("Annual Income (₹)", min_value=10000, value=500000, step=10000)
loan_amt = st.sidebar.number_input("Requested Loan Amount (₹)", min_value=10000, value=200000, step=10000)
term = st.sidebar.slider("Loan Term (Years)", 2, 20, 10)
cibil = st.sidebar.slider("CIBIL Score", 300, 900, 700)

st.sidebar.subheader("Asset Valuation (₹)")
res_asset = st.sidebar.number_input("Residential Assets", value=100000)
com_asset = st.sidebar.number_input("Commercial Assets", value=50000)
lux_asset = st.sidebar.number_input("Luxury Assets", value=20000)
bank_asset = st.sidebar.number_input("Bank Assets", value=150000)

# --- STEP 5: PREDICTION LOGIC ---
if st.button("Run Risk Analysis"):
    # Encode inputs to match the training data
    edu_val = 0 if education == "Graduate" else 1
    emp_val = 1 if employed == "Yes" else 0
    
    input_data = np.array([[
        dependents, edu_val, emp_val, income, loan_amt, term, cibil, 
        res_asset, com_asset, lux_asset, bank_asset
    ]])
    
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0] 
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Decision Output")
        if prediction[0] == 0: # Approved
            st.success("🎯 Prediction: **LOAN APPROVED**")
            score = probabilities[0] * 100
        else: # Rejected
            st.error("⚠️ Prediction: **LOAN REJECTED**")
            score = probabilities[0] * 100

        rejection_risk = probabilities[1] * 100
        st.write(f"**Rejection Probability:** {rejection_risk:.1f}%")
        
        if rejection_risk < 25:
            st.info("Risk Level: 🟢 **LOW RISK**")
        elif 25 <= rejection_risk < 60:
            st.warning("Risk Level: 🟡 **MEDIUM RISK**")
        else:
            st.error("Risk Level: 🔴 **HIGH RISK**")

    with col2:
        st.subheader("Graphical Insights")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Approval Confidence (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71" if prediction[0] == 0 else "#e74c3c"},
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 100], 'color': 'gray'}]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Asset Chart
    st.markdown("---")
    st.subheader("Asset Portfolio Analysis")
    asset_data = pd.DataFrame({
        'Asset Type': ['Residential', 'Commercial', 'Luxury', 'Bank'],
        'Value (₹)': [res_asset, com_asset, lux_asset, bank_asset]
    })
    fig_assets = px.pie(asset_data, values='Value (₹)', names='Asset Type', hole=.3)
    st.plotly_chart(fig_assets, use_container_width=True)

st.markdown("---")
st.caption("Powered by Machine Learning | Project: Loan Approval Prediction System")