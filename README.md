# 📉 Customer Churn Prediction System

An end-to-end, production-style Machine Learning project that predicts whether a customer is likely to churn based on telecom customer data.

This project demonstrates how a real-world ML solution is built — from data analysis and model training to backend deployment and a user-facing web interface.

---

## 🚀 Project Overview

Problem:
Customer churn is a critical business problem where customers stop using a company’s services.

Solution:
This system predicts customer churn using a trained Machine Learning model, exposed via a FastAPI backend and accessed through an interactive Streamlit web application.

---

## 🧠 System Architecture

User
 ↓
Streamlit Web UI
 ↓
FastAPI Backend (REST API)
 ↓
Preprocessing Pipeline
 ↓
Machine Learning Model
 ↓
Churn Prediction + Probability

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Joblib
- Git & GitHub

---

## 📊 Key Features

- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing pipeline
- Multiple ML models with comparison
- Best model selection
- Production-ready FastAPI backend
- Interactive Streamlit frontend
- Consistent inference pipeline
- Clean and reproducible project structure

---

## ▶️ How to Run the Project (Step-by-Step)

1. Clone the repository

git clone https://github.com/kamrankausher/customer-churn-prediction.git  
cd customer-churn-prediction

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment

venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

---

## 🚀 Run the Applications

Start FastAPI backend

uvicorn app.main:app --reload

Swagger API documentation:
http://127.0.0.1:8000/docs

---

Start Streamlit frontend (new terminal)

streamlit run app/streamlit_app.py

The Streamlit UI opens automatically in your browser.

---

## 🔍 API Endpoint

POST /predict

Response Example:

{
  "churn_prediction": 1,
  "churn_probability": 0.7992
}

---

## 🎯 Project Output

- Predicts whether a customer is likely to churn
- Displays churn probability
- Accepts real-time user input through a web UI

---

## 🌍 How Others Can Use This Project

1. Clone the GitHub repository
2. Install dependencies
3. Run FastAPI backend
4. Run Streamlit frontend
5. Enter customer details and get predictions

---

## 📌 Deployment Status

- The project currently runs locally
- Can be deployed on cloud platforms (Render, Streamlit Cloud) for public access

---

## 👨‍💻 Author

Kamran Kausher  
Final Year B.Tech (Computer Science Engineering)
