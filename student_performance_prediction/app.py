import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="EduPredict AI | Dashboard",
    page_icon="🎓",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    </style>
    """, unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    try:
        with open('student_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file 'student_model.pkl' not found! Run `train.py` first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

assets = load_model()
model = assets['model']
mappings = assets['mappings']
features = assets.get('features', ['Gender', 'Study_Hours_per_Week', 'Attendance_Rate', 
                                   'Past_Exam_Scores', 'Parental_Education_Level', 
                                   'Internet_Access_at_Home', 'Extracurricular_Activities'])

# ===================== SIDEBAR =====================
st.sidebar.title("🎛️ Student Profile")

with st.sidebar.expander("👤 Personal Information", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    parent_edu = st.selectbox("Parent Education", ["High School", "Bachelors", "Masters", "PhD"])

with st.sidebar.expander("📚 Academic Details", expanded=True):
    study_hours = st.slider("Weekly Study Hours", 5, 50, 25)
    attendance = st.slider("Attendance Rate (%)", 40, 100, 85)
    past_scores = st.slider("Previous Exam Scores", 40, 100, 75)

with st.sidebar.expander("🌐 Other Factors"):
    internet = st.radio("Internet Access at Home", ["Yes", "No"], horizontal=True)
    extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

# ===================== MAIN AREA =====================
st.title("🎓 EduPredict AI - Student Performance Analytics")
st.write("**Real-time Academic Risk Assessment using Random Forest**")
st.markdown("---")

# ===================== GENERATE REPORT =====================
if st.sidebar.button("✨ GENERATE PERFORMANCE REPORT", use_container_width=True, type="primary"):
    
    try:
        # Prepare input data
        input_dict = {
            'Gender': mappings['Gender'][gender],
            'Study_Hours_per_Week': study_hours,
            'Attendance_Rate': attendance,
            'Past_Exam_Scores': past_scores,
            'Parental_Education_Level': mappings['Parental_Education_Level'][parent_edu],
            'Internet_Access_at_Home': mappings['Internet_Access_at_Home'][internet],
            'Extracurricular_Activities': mappings['Extracurricular_Activities'][extracurricular]
        }
        
        # Create DataFrame with exact feature order
        input_df = pd.DataFrame([input_dict])[features]
        
        # Make Prediction
        prediction_prob = model.predict_proba(input_df)[0]
        pass_prob = prediction_prob[1] * 100
        status = "PASS" if pass_prob >= 50 else "FAIL"
        
        # ===================== METRICS =====================
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Predicted Status", status, 
                   delta="High Risk" if status == "FAIL" else "Good Standing")
        col2.metric("Pass Probability", f"{pass_prob:.1f}%")
        col3.metric("Attendance Rate", f"{attendance}%")
        col4.metric("Study Hours", f"{study_hours} hrs/week")

        st.markdown("---")
        
        # ===================== INSIGHTS =====================
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("🎯 Risk Assessment")
            if pass_prob > 80:
                st.success("### Excellent Performance\nLow risk of failure.")
            elif pass_prob > 60:
                st.warning("### Average Performance\nMonitor closely.")
            else:
                st.error("### At Risk\nImmediate intervention recommended.")
            
            # Feature Importance
            st.subheader("💡 Key Factors")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("📊 Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pass_prob,
                title={'text': "Pass Likelihood"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 50], 'color': '#ef4444'},
                        {'range': [50, 75], 'color': '#f59e0b'},
                        {'range': [75, 100], 'color': '#22c55e'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=380)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
        st.info("Check that column names in train.py match the input features.")

else:
    st.info("👈 Adjust parameters in the sidebar and click **GENERATE PERFORMANCE REPORT**")

st.caption("EduPredict AI | Powered by Random Forest")