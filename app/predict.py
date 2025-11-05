from fastapi import FastAPI
import numpy as np
from pickel import load_model

app = FastAPI()

# Load at startup
weights, scaler, feature_names = load_model()

@app.post("/predict")
def predict(data: dict):
    try:
        # Validate input
        X = np.array([data[f] for f in feature_names]).reshape(1, -1)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = X_scaled @ weights
        
        return {"prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}