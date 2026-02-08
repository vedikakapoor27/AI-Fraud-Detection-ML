from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")


class InputData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "AI Fraud Detection API is running 🚀"}


@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)

    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][prediction])
    else:
        probability = 0.0

    return {
        "prediction": prediction,
        "probability": probability,
    }
