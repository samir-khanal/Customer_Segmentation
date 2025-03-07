import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

import os
current_dir = os.path.dirname(__file__)
scaler_path = os.path.join(current_dir, 'models', 'scaler.joblib')
scaler = joblib.load(scaler_path)
print("Current directory:", current_dir)
print("Scaler path:", scaler_path)


def preprocess_input(df):
    features = ['Recency', 'Frequency', 'Monetary', 'Total_Returns']
    df = df[features]
    df_scaled = scaler.transform(df)
    return df_scaled

def load_model():
    try:
        model_path = os.path.join(current_dir, 'models', 'kmeans_model.joblib')
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

def predict_segment(model, data):
    return model.predict(data)[0]
# model.predict(data) returns an array (e.g., [3] instead of just 3)
# [0] extracts the actual predicted cluster number (e.g., 3)
