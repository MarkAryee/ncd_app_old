from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Create FastAPI app
app = FastAPI(title="My FastAPI Demo", version="1.0.0")


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

