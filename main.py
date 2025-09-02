from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi.responses import JSONResponse
from pytorch_tabnet.tab_model import TabNetClassifier
import os
import uvicorn

model = tf.keras.models.load_model('ncd_models/hypertension/deep-model1.keras')
scaler = joblib.load('ncd_models/hypertension/deep-scaler.joblib')
csv_name = "ncd_models/hypertension/c2.csv"

model2 = tf.keras.models.load_model('ncd_models/arthritis/deep-model1.keras')
scaler2 = joblib.load('ncd_models/arthritis/deep-scaler.joblib')
csv_name2 = "ncd_models/arthritis/selected.csv"

model3 = tf.keras.models.load_model('ncd_models/lung_cancer/deep-model1.keras')
scaler3 = joblib.load('ncd_models/lung_cancer/deep-scaler.joblib')
csv_name3 = "ncd_models/lung_cancer/selected.csv"

model4 = TabNetClassifier()
model4.load_model('ncd_models/asthma/tabnet_asthma_model.zip')
scaler4 = joblib.load('ncd_models/asthma/tabnet_scaler.joblib')
csv_name4 = "ncd_models/asthma/selected.csv"

model5 = TabNetClassifier()
model5.load_model('ncd_models/diabetes/tabnet-model1.keras.zip')

scaler5 = joblib.load('ncd_models/diabetes/deep-scaler.joblib')
csv_name5 = "ncd_models/diabetes/selected.csv"


app = FastAPI()
# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with your frontend's origin (e.g. "http://localhost:8100")
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)
# Define the input data model
class PatientData(BaseModel):
    Sex: int
    PhysicalHealthDays: float
    MentalHealthDays: float
    LastCheckupTime: int
    PhysicalActivities: int
    SleepHours: float
    RemovedTeeth: int
    HadAngina: int
    HadStroke: int
    DeafOrHardOfHearing: int
    BlindOrVisionDifficulty: int
    DifficultyConcentrating: int
    DifficultyWalking: int
    DifficultyDressingBathing: int
    DifficultyErrands: int
    SmokerStatus: int
    ECigaretteUsage: int
    ChestScan: int
    RaceEthnicityCategory: int
    AgeCategory: int
    HeightInMeters: float
    WeightInKilograms: float
    BMI: float
    AlcoholDrinkers: int
    HIVTesting: int
    FluVaxLast12: int
    PneumoVaxEver: int
    TetanusLast10Tdap: int
    HighRiskLastYear: int
    CovidPos: int


@app.options("/predict")    
async def options_handler():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
    }
    return JSONResponse(content={}, headers=headers)


@app.post("/predict")
def predict_asthma(data: PatientData):
    # Convert incoming data to DataFrame (or array) matching model input
    input_df = pd.DataFrame([data.dict()])

    # You might need to do the same encoding/mapping/scaling you used in training:
    # For example, map string columns to ints (you can write a helper function)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability
    #proba = model.predict_proba(input_scaled)[:, 1][0]
    proba = model.predict(input_scaled).flatten()[0]

    # Predict class
    pred = model.predict(input_scaled)[0]

    return {
        "asthma_probability": float(proba),
        "asthma_prediction": int(pred)
    }

'''
if __name__ == "__main__":
    # Get port from environment (Render sets $PORT automatically)
    port = int(os.environ.get("PORT", 8080))  # fallback to 8000 for local dev

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
'''
