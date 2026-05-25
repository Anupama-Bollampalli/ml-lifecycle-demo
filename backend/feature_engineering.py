from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
import joblib, os

def load_and_prepare():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'saved_models/scaler.joblib')
    return X_scaled, y, feature_names
