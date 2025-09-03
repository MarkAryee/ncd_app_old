from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(title="My FastAPI Demo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with your frontend's origin (e.g. "http://localhost:8100")
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Root route → show a simple HTML page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>FastAPI Demo</title>
        </head>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h1>🚀 FastAPI is running!</h1>
            <p>Welcome to my demo app.</p>
            <p>Try these routes:</p>
            <ul style="list-style:none;">
                <li><a href="/">🏠 Home</a></li>
                <li><a href="/hello?name=Mark">👋 /hello</a></li>
                <li><a href="/routes">📜 /routes</a></li>
                <li><a href="/docs">📖 /docs (Swagger UI)</a></li>
            </ul>
        </body>
    </html>
    """


# Example route with query parameter
@app.get("/hello")
def say_hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}


# Route to show all registered endpoints
@app.get("/routes")
def list_routes():
    routes_info = []
    for route in app.routes:
        routes_info.append({"path": route.path, "name": route.name, "methods": list(route.methods)})
    return {"available_routes": routes_info}









import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

model4 = TabNetClassifier()
model4.load_model('ncd_models/asthma/tabnet_asthma_model.zip')
scaler4 = joblib.load('ncd_models/asthma/tabnet_scaler.joblib')
csv_name4 = "ncd_models/asthma/selected.csv"

model5 = TabNetClassifier()
model5.load_model('ncd_models/diabetes/tabnet-model1.keras.zip')

scaler5 = joblib.load('ncd_models/diabetes/deep-scaler.joblib')
csv_name5 = "ncd_models/diabetes/selected.csv"




import pandas as pd
from pydantic import BaseModel
#---------------------------------TabNet_Models--------------------------------------------------








class PatientData4(BaseModel):# Defining the input data model
    Sex: int
    PhysicalHealthDays: float
    MentalHealthDays: float
    LastCheckupTime: int
    PhysicalActivities: int
    SleepHours: float
    RemovedTeeth: int
    HadHeartAttack: int
    HadAngina: int
    HadStroke: int
    HadSkinCancer: int
    HadCOPD: int
    HadDepressiveDisorder: int
    HadKidneyDisease: int
    HadArthritis: int
    HadDiabetes: int
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
 
from helpers_explain_utils import explain_prediction_tabnet
@app.post("/predict/asthma")
def predict_asthma(data: PatientData4):
    
    df = pd.read_csv(csv_name4)
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(['HadAsthma'], axis=1)
    
    return explain_prediction_tabnet(model4, scaler4, input_df, X, "Asthma", 0.3477, 0.5095)



#---------------------------------Model5--------------------------------------------------

class PatientData5(BaseModel):# Defining the input data model
    gender: float
    age: float
    #hypertension: int
    #heart_disease: int
    smoking_history: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    #diabetes: int

@app.post("/predict/diabetes")
def predict_diabetes(data: PatientData5):
    
    df = pd.read_csv(csv_name5)
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(['hypertension', 'heart_disease','diabetes'], axis=1)
    
    #return explain_prediction(model5, scaler5, input_df, X, "Diabetes", 0.5495, 0.5495)
    return explain_prediction_tabnet(model5, scaler5, input_df, X, "Diabetes", 0.5495, 0.5495)

  



