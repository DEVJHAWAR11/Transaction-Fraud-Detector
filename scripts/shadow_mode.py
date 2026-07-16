import pandas as pd
import numpy as np
import joblib
import os

# shadow mode: champion makes the real decision
# challenger scores every transaction too but its output is just logged, not used
# this is how you safely test a new model on real traffic without risk

champion = joblib.load("models/xgboost_baseline.joblib")
challenger = joblib.load("models/logistic_regression_challenger.joblib")
pipeline = joblib.load("models/pipeline.joblib")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

THRESHOLD = 0.5

champ_probs = champion.predict_proba(X_test.values)[:, 1]
chall_probs = challenger.predict_proba(X_test.values)[:, 1]

log = pd.DataFrame({
    "true_label": y_test.values,
    "champion_prob": champ_probs,
    "champion_pred": (champ_probs >= THRESHOLD).astype(int),
    "challenger_prob": chall_probs,
    "challenger_pred": (chall_probs >= THRESHOLD).astype(int),
    "decision": "champion"  # challenger is shadow — never used for actual decision
})

os.makedirs("data/processed", exist_ok=True)
log.to_csv("data/processed/shadow_log.csv", index=False)

print(f"shadow mode complete — {len(log)} transactions scored")
print(f"champion flagged fraud: {log['champion_pred'].sum()}")
print(f"challenger flagged fraud (shadow, not used): {log['challenger_pred'].sum()}")
print("saved to data/processed/shadow_log.csv")
