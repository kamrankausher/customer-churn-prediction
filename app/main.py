from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Resolve project root dynamically (Render-safe)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Customer Churn Prediction API")

# --------------------------------------------------
# Input schema
# --------------------------------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "Customer Churn API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_churn(customer: CustomerData):
    input_df = pd.DataFrame([customer.dict()])
    processed = preprocessor.transform(input_df)

    prediction = int(model.predict(processed)[0])
    probability = float(model.predict_proba(processed)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4)
    }
