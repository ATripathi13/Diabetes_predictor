from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model/diabetes_model.pkl")

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}

@app.get("/features")
def feature_importance():
    importances = model.feature_importances_
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
    return dict(zip(feature_names, importances.tolist()))
