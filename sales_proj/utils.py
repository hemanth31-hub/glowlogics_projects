import pandas as pd
from datetime import timedelta
import numpy as np

def load_and_preprocess(df):
    """Clean and prepare the data - Fixed date parsing"""
    # FIXED: Handle DD-MM-YYYY format using dayfirst=True
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Check for any failed conversions
    if df['Date'].isna().any():
        st.error("Some dates could not be parsed. Please check your Date column format.")
        st.stop()
    
    df = df.sort_values('Date')
    
    # Aggregate daily sales
    df = df.groupby('Date').agg({
        'Units_Sold': 'sum',
        'Revenue': 'sum',
        'Promotion': 'mean',
        'Holiday': 'max'
    }).reset_index()
    
    return df


def create_sample_data():
    """Create sample dataset"""
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    np.random.seed(42)
    
    data = {
        'Date': dates,
        'Units_Sold': np.random.poisson(lam=200, size=len(dates)) + \
                      np.sin(np.arange(len(dates))/30)*50 + \
                      np.random.normal(0, 20, len(dates)),
        'Revenue': np.random.poisson(lam=5000, size=len(dates)) * 1.2,
        'Promotion': np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3]),
        'Holiday': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
    }
    return pd.DataFrame(data)