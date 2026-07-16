import pandas as pd
import numpy as np
import joblib
import os

# canary: route 5% of traffic to challenger, 95% to champion
# challenger's decision is used for that 5% — unlike shadow mode
# lets you test impact on a small slice of real traffic before full rollout

CANARY_PCT = 0.05

champion = joblib.load("models/xgboost_baseline.joblib")
challenger = joblib.load("models/logistic_regression_challenger.joblib")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

np.random.seed(42)
n = len(X_test)
# assign each transaction to either canary or champion group
is_canary = np.random.rand(n) < CANARY_PCT

champ_probs = champion.predict_proba(X_test.values)[:, 1]
chall_probs = challenger.predict_proba(X_test.values)[:, 1]

THRESHOLD = 0.5

# final decision: challenger decides for canary slice, champion decides for the rest
final_pred = np.where(
    is_canary,
    (chall_probs >= THRESHOLD).astype(int),
    (champ_probs >= THRESHOLD).astype(int)
)

log = pd.DataFrame({
    "true_label": y_test.values,
    "model_used": np.where(is_canary, "challenger", "champion"),
    "prob": np.where(is_canary, chall_probs, champ_probs),
    "pred": final_pred
})

os.makedirs("data/processed", exist_ok=True)
log.to_csv("data/processed/canary_log.csv", index=False)

champ_slice = log[log["model_used"] == "champion"]
canary_slice = log[log["model_used"] == "challenger"]

print(f"canary simulation complete — {n} transactions")
print(f"champion slice: {len(champ_slice)} transactions ({len(champ_slice)/n*100:.1f}%)")
print(f"canary slice:   {len(canary_slice)} transactions ({len(canary_slice)/n*100:.1f}%)")
print(f"champion fraud flags: {champ_slice['pred'].sum()}")
print(f"canary fraud flags:   {canary_slice['pred'].sum()}")
print("saved to data/processed/canary_log.csv")
