from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Resolve absolute paths (works locally + Render)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using ML",
    version="1.0"
)

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
    # Convert input to DataFrame
    input_df = pd.DataFrame([customer.dict()])

    # Apply preprocessing
    processed_data = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
