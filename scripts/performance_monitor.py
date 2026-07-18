import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from promote import THRESHOLDS

# rolling window monitor: checks model performance on recent predictions
# flags if precision or recall drop below the stage 4 thresholds
# in production this would run on live prediction logs with true labels coming in later
# here we simulate rolling windows using slices of the test set

from sklearn.metrics import precision_score, recall_score

model = joblib.load("models/production_model.joblib")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

WINDOW_SIZE = 5000

print(f"rolling window monitor — window size: {WINDOW_SIZE} transactions")
print(f"thresholds: recall >= {THRESHOLDS['recall']}, precision >= {THRESHOLDS['precision']}\n")

alerts = []
n = len(X_test)

for start in range(0, n, WINDOW_SIZE):
    end = min(start + WINDOW_SIZE, n)
    X_window = X_test.iloc[start:end]
    y_window = y_test.iloc[start:end]

    y_pred = model.predict(X_window.values)

    # skip windows with no fraud at all — can't compute meaningful recall
    if y_window.sum() == 0:
        continue

    prec = precision_score(y_window, y_pred, zero_division=0)
    rec = recall_score(y_window, y_pred, zero_division=0)
    fraud_in_window = y_window.sum()

    status = "ok"
    reasons = []
    if rec < THRESHOLDS["recall"]:
        reasons.append(f"recall {rec:.3f} < {THRESHOLDS['recall']}")
        status = "ALERT"
    if prec < THRESHOLDS["precision"]:
        reasons.append(f"precision {prec:.3f} < {THRESHOLDS['precision']}")
        status = "ALERT"

    flag = " ⚠" if status == "ALERT" else ""
    print(f"window {start}-{end} | fraud cases: {fraud_in_window} | recall: {rec:.3f} | precision: {prec:.3f} | {status}{flag}")
    if reasons:
        for r in reasons:
            print(f"  → {r}")
        alerts.append({"window": f"{start}-{end}", "reasons": reasons})

print(f"\n{len(alerts)} alert(s) triggered across {n // WINDOW_SIZE} windows")
