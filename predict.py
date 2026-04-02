from src.anfis import ANFIS

import pickle
import numpy as np

# Load model
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
model_path = os.path.join(BASE_DIR, "model", "anfis_full_model.pkl")

with open(model_path, "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler_X = data["scaler_X"]
scaler_y = data["scaler_y"]

def predict_aqi(input_data):
    input_data = np.array(input_data).reshape(1, -1)

    X_scaled = scaler_X.transform(input_data)

    y_pred_norm, _, _ = model.forward(X_scaled)

    y_pred = scaler_y.inverse_transform(
        y_pred_norm.reshape(-1, 1)
    ).flatten()

    return y_pred[0]


# Test
if __name__ == "__main__":
    sample = [50, 80, 30, 20, 10, 60]
    print("Predicted AQI:", predict_aqi(sample))