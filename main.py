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
import tensorflow as tf

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







from helpers_explain_utils import explain_prediction

#---------------------------------Model--------------------------------------------------
from pydantic import BaseModel
class PatientData(BaseModel):# Define the input data model
    age: int
    sex: int
    is_smoking: int
    cigsPerDay: float
    BPMeds: float
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float
    

@app.post("/predict/hypertension")
def predict_asthma(data: PatientData):
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    
    df = pd.read_csv(csv_name)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    X = df.drop(['prevalentHyp', 'prevalentStroke', 'diabetes', 'education'], axis=1)

    return explain_prediction(model, scaler, input_df, X, "Hypertension", 0.0114, 0.4006)









#---------------------------------Model2--------------------------------------------------

class PatientData2(BaseModel):# Defining the input data model
    Checkup: int
    Exercise: int
    #Heart_Disease: int
    Skin_Cancer: int
    Other_Cancer: int
   #Depression: int
    Diabetes: int
    #Arthritis: int
    Sex: int
    Age_Category: int
    Height_cm: float
    Weight_kg: float
    BMI: float
    Smoking_History: int
    Alcohol_Consumption: float
    Fruit_Consumption: float
    Green_Vegetables_Consumption: float
    FriedPotato_Consumption: float

@app.post("/predict/arthritis")
def predict_arthritis(data: PatientData2):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    
    df = pd.read_csv(csv_name2)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    X = df.drop(['Heart_Disease', 'Arthritis','Depression'], axis=1) 

    return explain_prediction(model2, scaler2, input_df, X, "Arthritis", 0.1969, 0.4359)
    
    '''
    input_scaled = scaler2.transform(input_df)    # Scale features

    # proba = model2.predict_proba(input_scaled)[:, 1][0] # Predict probability
    proba = model2.predict(input_scaled).flatten()[0]
    
    pred = model2.predict(input_scaled)[0]    # Predict class

    return {
        "ncd_probability": float(proba),
        "ncd_prediction": int(pred)
    }
    '''



#---------------------------------Model3--------------------------------------------------

class PatientData3(BaseModel):# Defining the input data model
    Age: int
    Gender: int
    Smoking_Status: int
    Second_Hand_Smoke: int
    Air_Pollution_Exposure: int
    Occupation_Exposure: int
    Rural_or_Urban: int
    Socioeconomic_Status: int
    Healthcare_Access: int
    Screening_Availability: int
    Family_History: int
    Indoor_Smoke_Exposure: int
    Tobacco_Marketing_Exposure: int

@app.post("/predict/lung_cancer")
def predict_lung_cancer(data: PatientData3):
    
    input_df = pd.DataFrame([data.dict()]) # Convert incoming data to DataFrame (or array) matching model input
    
     
    df = pd.read_csv(csv_name3)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])


    X = df.drop(['Final_Prediction'], axis=1)
    
    return explain_prediction(model3, scaler3, input_df, X, "Lung Cancer",0.4323, 0.4817)
    
    
#---------------------------------Model4--------------------------------------------------

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
   
#---------------------------------TabNet_Models--------------------------------------------------

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

  



