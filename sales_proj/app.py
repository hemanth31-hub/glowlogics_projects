import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from utils import load_and_preprocess, create_sample_data

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title(" Sales Forecasting System")
st.markdown("**Multi-Model Time Series Forecasting**")

# ===================== SIDEBAR =====================
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your sales data (CSV)", type=["csv"])

if uploaded_file is None:
    st.sidebar.info(" Using sample dataset")
    df = create_sample_data()
else:
    df = pd.read_csv(uploaded_file)

df = load_and_preprocess(df)
df['Day'] = (df['Date'] - df['Date'].min()).dt.days   # Required for some models

st.sidebar.success(f"Records: {len(df)} | {df['Date'].min().date()} to {df['Date'].max().date()}")

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs([" EDA", " Analysis", " Forecasting", " Model Comparison"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df, x='Date', y='Units_Sold', title="Daily Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df['Month'] = df['Date'].dt.strftime('%B')
        monthly = df.groupby('Month')['Units_Sold'].mean().reset_index()
        fig2 = px.bar(monthly, x='Month', y='Units_Sold', title="Seasonality - Monthly Average")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader(" Time Series Analysis")
    
    analysis_model = st.selectbox(
        "Select Analysis Model",
        ["Prophet", "Moving Average", "Linear Regression Trend", "Random Forest Feature Importance"]
    )
    
    if st.button("Show Analysis", type="primary"):
        with st.spinner(f"Analyzing with {analysis_model}..."):
            
            if analysis_model == "Prophet":
                prophet_df = df[['Date', 'Units_Sold']].rename(columns={'Date': 'ds', 'Units_Sold': 'y'})
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=0)
                forecast = m.predict(future)
                st.pyplot(m.plot_components(forecast))
                st.success("Prophet Decomposition: Trend + Seasonality + Holidays")
                
            elif analysis_model == "Moving Average":
                window = st.slider("Moving Average Window (days)", 3, 30, 7)
                df['MA'] = df['Units_Sold'].rolling(window=window).mean()
                
                fig = px.line(df, x='Date', y=['Units_Sold', 'MA'], 
                             title=f"Moving Average Smoothing (Window = {window} days)",
                             labels={'value': 'Sales'})
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_model == "Linear Regression Trend":
                X = df[['Day']]
                y = df['Units_Sold']
                model = LinearRegression().fit(X, y)
                df['Trend'] = model.predict(X)
                
                fig = px.line(df, x='Date', y=['Units_Sold', 'Trend'],
                             title="Linear Regression - Overall Trend")
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_model == "Random Forest Feature Importance":
                df_temp = df.copy()
                df_temp['Month'] = df_temp['Date'].dt.month
                df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek
                df_temp['Lag_1'] = df_temp['Units_Sold'].shift(1)
                df_temp = df_temp.dropna()
                
                X = df_temp[['Day', 'Month', 'DayOfWeek', 'Lag_1']]
                y = df_temp['Units_Sold']
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                            title="Random Forest - Feature Importance")
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔮 Forecasting")
    
    model_choice = st.selectbox(
        "Select Forecasting Model",
        ["Prophet", "Linear Regression", "Moving Average", "Random Forest", "LSTM"]
    )
    
    forecast_days = st.slider("Forecast Next Days", 7, 90, 30)
    
    if st.button(" Generate Forecast", type="primary"):
        with st.spinner(f"Training {model_choice} model..."):
            data = df.copy()
            forecast_dates = pd.date_range(start=data['Date'].max() + timedelta(days=1), periods=forecast_days)
            
            if model_choice == "Prophet":
                prophet_df = data[['Date', 'Units_Sold']].rename(columns={'Date': 'ds', 'Units_Sold': 'y'})
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                pred = forecast['yhat'].tail(forecast_days).values
                
            elif model_choice == "Linear Regression":
                X = data[['Day']]
                y = data['Units_Sold']
                model = LinearRegression()
                model.fit(X, y)
                future_days = np.array(range(data['Day'].max()+1, data['Day'].max()+forecast_days+1)).reshape(-1, 1)
                pred = model.predict(future_days)
                
            elif model_choice == "Moving Average":
                window = st.slider("Moving Average Window (days)", 3, 30, 7, key="ma_forecast")
                ma_value = data['Units_Sold'].rolling(window=window).mean().iloc[-1]
                pred = [ma_value] * forecast_days
                
            elif model_choice == "Random Forest":
                data['Month'] = data['Date'].dt.month
                data['DayOfWeek'] = data['Date'].dt.dayofweek
                X = data[['Day', 'Month', 'DayOfWeek']]
                y = data['Units_Sold']
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                future_df = pd.DataFrame({
                    'Day': range(data['Day'].max()+1, data['Day'].max()+forecast_days+1),
                    'Month': forecast_dates.month,
                    'DayOfWeek': forecast_dates.dayofweek
                })
                pred = model.predict(future_df)
                
            elif model_choice == "LSTM":
                st.info("Training LSTM (this may take a few seconds)...")
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data['Units_Sold'].values.reshape(-1, 1))
                
                def create_sequences(data, seq_length=14):
                    X, y = [], []
                    for i in range(len(data) - seq_length):
                        X.append(data[i:i+seq_length])
                        y.append(data[i+seq_length])
                    return np.array(X), np.array(y)
                
                seq_length = 14
                X, y = create_sequences(scaled_data, seq_length)
                
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense
                
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
                model_lstm.add(Dense(1))
                model_lstm.compile(optimizer='adam', loss='mse')
                model_lstm.fit(X, y, epochs=20, batch_size=16, verbose=0)
                
                last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
                pred_scaled = []
                for _ in range(forecast_days):
                    pred_val = model_lstm.predict(last_seq, verbose=0)
                    pred_scaled.append(pred_val[0, 0])
                    last_seq = np.append(last_seq[:, 1:, :], pred_val.reshape(1, 1, 1), axis=1)
                
                pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted_Sales': np.round(pred, 2)
            })
            
            st.success(f" {model_choice} Forecast Generated!")
            st.dataframe(forecast_df, use_container_width=True)
            
            fig = px.line(forecast_df, x='Date', y='Predicted_Sales', 
                         title=f"Future Sales Forecast using {model_choice}")
            st.plotly_chart(fig, use_container_width=True)
            
            csv = forecast_df.to_csv(index=False)
            st.download_button(" Download Forecast", csv, f"{model_choice}_forecast.csv", "text/csv")

with tab4:
    st.subheader(" Model Comparison (Test Set Performance)")
    # (Same as previous version - Comparison remains unchanged)
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()
    
    results = {}
    
    # Linear Regression
    X_train = train[['Day']]
    y_train = train['Units_Sold']
    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(test[['Day']])
    results["Linear Regression"] = {
        "MAE": round(mean_absolute_error(test['Units_Sold'], pred_lr), 2),
        "RMSE": round(np.sqrt(mean_squared_error(test['Units_Sold'], pred_lr)), 2)
    }
    
    # Random Forest
    train_rf = train.copy()
    test_rf = test.copy()
    train_rf['Month'] = train_rf['Date'].dt.month
    test_rf['Month'] = test_rf['Date'].dt.month
    train_rf['DayOfWeek'] = train_rf['Date'].dt.dayofweek
    test_rf['DayOfWeek'] = test_rf['Date'].dt.dayofweek
    
    X_train_rf = train_rf[['Day', 'Month', 'DayOfWeek']]
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_rf, train_rf['Units_Sold'])
    pred_rf = rf.predict(test_rf[['Day', 'Month', 'DayOfWeek']])
    results["Random Forest"] = {
        "MAE": round(mean_absolute_error(test['Units_Sold'], pred_rf), 2),
        "RMSE": round(np.sqrt(mean_squared_error(test['Units_Sold'], pred_rf)), 2)
    }
    
    # Prophet
    p_train = train[['Date', 'Units_Sold']].rename(columns={'Date': 'ds', 'Units_Sold': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(p_train)
    future = m.make_future_dataframe(periods=len(test))
    forecast = m.predict(future)
    pred_prophet = forecast['yhat'].tail(len(test)).values
    results["Prophet"] = {
        "MAE": round(mean_absolute_error(test['Units_Sold'], pred_prophet), 2),
        "RMSE": round(np.sqrt(mean_squared_error(test['Units_Sold'], pred_prophet)), 2)
    }
    
    comparison_df = pd.DataFrame(results).T
    st.dataframe(comparison_df, use_container_width=True)

st.caption("Sales Forecasting System | Multiple Models for Forecasting & Analysis")