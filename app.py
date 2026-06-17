import os
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

app = FastAPI(title="Loan Approval Predictor")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model.pkl relative to this file so it loads correctly no matter
# what directory Vercel's serverless runtime executes from.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


class LoanInput(BaseModel):
    age: float
    income: float
    credit_score: float
    loan_amount: float
    employment_years: float
    num_accounts: float
    num_loans: float
    monthly_expenses: float
    savings: float
    debt_ratio: float


@app.get("/health")
def health():
    return {
        "status": "Loan Approval API is running",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: LoanInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array([[
            data.age, data.income, data.credit_score, data.loan_amount,
            data.employment_years, data.num_accounts, data.num_loans,
            data.monthly_expenses, data.savings, data.debt_ratio,
        ]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return {
            "approved": int(prediction),
            "result": "Approved" if prediction == 1 else "Rejected",
            "confidence": round(float(max(probability)) * 100, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
