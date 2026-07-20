"""
retraining pipeline — full loop:
  1. check drift (evidently wasserstein distance on key features)
  2. check performance degradation (rolling window recall)
  3. if either triggers → retrain on latest data
  4. validate new model against thresholds (stage 4)
  5. if passes → log to mlflow as new challenger, ready for canary rollout

in production this script would be called by airflow, prefect, or a
github actions cron job. the logic here is identical to what that would run.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from promote import check_thresholds, get_metrics, THRESHOLDS
from sklearn.metrics import recall_score, precision_score
from xgboost import XGBClassifier

# ── step 1: check for data drift ─────────────────────────────────────────────

def check_drift(threshold=0.15):
    """
    computes wasserstein distance manually on key features
    avoids spinning up the full evidently report just for a trigger check
    returns True if drift detected (any feature above threshold)
    """
    from scipy.stats import wasserstein_distance

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    # simulate the same drift from stage 8 — this is our "new incoming batch"
    np.random.seed(42)
    current = X_test.sample(2000, random_state=99).reset_index(drop=True)
    current["Amount"] = current["Amount"] + 50 + np.random.normal(0, 10, len(current))
    current["V1"] = current["V1"] + np.random.normal(0.5, 0.3, len(current))
    current["V17"] = current["V17"] + np.random.normal(0.3, 0.2, len(current))

    reference = X_train.sample(5000, random_state=42).reset_index(drop=True)

    drifted = []
    key_features = ["Amount", "V1", "V17", "V14", "V12"]
    for col in key_features:
        score = wasserstein_distance(reference[col], current[col])
        # normalize by std of reference so scores are comparable across features
        score_normed = score / (reference[col].std() + 1e-8)
        if score_normed > threshold:
            drifted.append((col, round(score_normed, 4)))

    if drifted:
        print(f"drift detected in: {drifted}")
        return True

    print("no significant drift detected")
    return False


# ── step 2: check for performance degradation ─────────────────────────────────

def check_performance_degradation(window_size=5000, alert_threshold=0.5):
    """
    checks rolling windows — if more than alert_threshold fraction of windows
    are in ALERT state, flag as degraded
    """
    model = joblib.load("models/production_model.joblib")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    alert_count = 0
    window_count = 0

    for start in range(0, len(X_test), window_size):
        end = min(start + window_size, len(X_test))
        X_w = X_test.iloc[start:end]
        y_w = y_test.iloc[start:end]

        if y_w.sum() == 0:
            continue

        y_pred = model.predict(X_w.values)
        rec = recall_score(y_w, y_pred, zero_division=0)
        prec = precision_score(y_w, y_pred, zero_division=0)

        window_count += 1
        if rec < THRESHOLDS["recall"] or prec < THRESHOLDS["precision"]:
            alert_count += 1

    if window_count == 0:
        return False

    degraded_fraction = alert_count / window_count
    print(f"performance check: {alert_count}/{window_count} windows in alert ({degraded_fraction:.1%})")

    if degraded_fraction > alert_threshold:
        print("performance degradation detected — too many windows below threshold")
        return True

    print("performance within acceptable range")
    return False


# ── step 3: retrain ───────────────────────────────────────────────────────────

def retrain():
    print("\nretraining model on current processed data...")

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    # slightly tuned params vs baseline to simulate a real retrain attempt
    params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.08,
        "scale_pos_weight": spw,
        "eval_metric": "aucpr",
        "random_state": 99  # different seed = slightly different model
    }

    model = XGBClassifier(**params, verbosity=0)
    model.fit(X_train, y_train)

    path = "models/retrained_challenger.joblib"
    joblib.dump(model, path)
    print(f"retrained model saved to {path}")
    return model, path, params


# ── step 4: validate and log ──────────────────────────────────────────────────

def validate_and_log(model, model_path, params):
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    metrics = get_metrics(model, X_test.values, y_test)

    print("\nretrained model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    failures = check_thresholds(metrics)
    if failures:
        print("\n✗ retrained model failed validation — not logging as challenger")
        for f in failures:
            print(f)
        return False

    print("\n✓ retrained model passed validation")

    # log to mlflow as a new challenger run
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="retrained-challenger"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")
        mlflow.set_tag("status", "challenger-ready-for-canary")

    print("✓ logged to mlflow as 'retrained-challenger' — ready for canary rollout")
    return True


# ── main pipeline loop ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("retraining pipeline starting")
    print("=" * 55)

    drift_triggered = check_drift()
    perf_triggered = check_performance_degradation()

    if not drift_triggered and not perf_triggered:
        print("\nno retraining needed — model is healthy")
        sys.exit(0)

    print(f"\ntrigger reason: {'drift' if drift_triggered else ''} {'performance' if perf_triggered else ''}".strip())
    print("proceeding to retrain...")

    model, path, params = retrain()
    success = validate_and_log(model, path, params)

    if success:
        print("\npipeline complete — new challenger ready for canary")
    else:
        print("\npipeline complete — retraining did not produce a valid model, current champion unchanged")

    sys.exit(0 if success else 1)
