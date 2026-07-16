import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import shutil
import os

# minimum thresholds a model must clear to be promoted
THRESHOLDS = {
    "recall": 0.75,
    "precision": 0.10,
    "f1": 0.20,
    "roc_auc": 0.85
}

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

def check_thresholds(metrics):
    failures = []
    for metric, min_val in THRESHOLDS.items():
        if metrics[metric] < min_val:
            failures.append(f"  {metric}: got {metrics[metric]:.4f}, need >= {min_val}")
    return failures

def get_champion_metrics():
    import mlflow
    # pull best run from mlflow by f1 score
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fraud-detection")
    if experiment is None:
        return None, None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1
    )
    if not runs:
        return None, None

    best_run = runs[0]
    metrics = {
        "precision": best_run.data.metrics.get("precision"),
        "recall": best_run.data.metrics.get("recall"),
        "f1": best_run.data.metrics.get("f1"),
        "roc_auc": best_run.data.metrics.get("roc_auc")
    }
    return metrics, best_run.info.run_id

def promote(model_path, model_name="challenger"):
    print(f"\n=== validating: {model_name} ===")

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    model = joblib.load(model_path)
    metrics = get_metrics(model, X_test, y_test)

    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # threshold check
    failures = check_thresholds(metrics)
    if failures:
        print(f"\n✗ model rejected — failed thresholds:")
        for f in failures:
            print(f)
        return False

    print("\n✓ passed threshold checks")

    # compare against current champion in mlflow
    champion_metrics, champion_run_id = get_champion_metrics()
    if champion_metrics and champion_metrics["f1"] is not None:
        print(f"\nchampion f1 (run {champion_run_id[:8]}): {champion_metrics['f1']:.4f}")
        print(f"challenger f1: {metrics['f1']:.4f}")

        if metrics["f1"] <= champion_metrics["f1"]:
            print(f"\n✗ model rejected — challenger f1 ({metrics['f1']:.4f}) does not beat champion ({champion_metrics['f1']:.4f})")
            return False

        print("\n✓ challenger beats champion")

    # save as production model
    os.makedirs("models", exist_ok=True)
    dest = "models/production_model.joblib"
    shutil.copy(model_path, dest)
    print(f"\n✓ model promoted → saved to {dest}")
    return True


if __name__ == "__main__":
    # run against xgboost baseline as a demo
    result = promote("models/xgboost_baseline.joblib", model_name="xgboost-baseline")
    print(f"\npromotion result: {'approved' if result else 'rejected'}")
