from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="fraud detection api")

# load once at startup
model_path = os.environ.get("MODEL_PATH", "models/production_model.joblib")
pipeline_path = os.environ.get("PIPELINE_PATH", "models/pipeline.joblib")

model = joblib.load(model_path)
pipeline = joblib.load(pipeline_path)

FRAUD_THRESHOLD = 0.5

# original column order before pipeline reorders them
RAW_FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    @field_validator("Amount")
    @classmethod
    def amount_not_negative(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

    @field_validator("Time")
    @classmethod
    def time_not_negative(cls, v):
        if v < 0:
            raise ValueError("Time cannot be negative")
        return v


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_path}


@app.post("/predict")
def predict(transaction: Transaction):
    # build a single-row dataframe in the original column order
    data = {f: [getattr(transaction, f)] for f in RAW_FEATURE_ORDER}
    df = pd.DataFrame(data)

    # pipeline applies the same scaling as training
    processed = pipeline.transform(df)

    prob = float(model.predict_proba(processed)[0][1])
    label = "fraud" if prob >= FRAUD_THRESHOLD else "legit"

    return {
        "fraud_probability": round(prob, 4),
        "label": label,
        "threshold_used": FRAUD_THRESHOLD
    }
