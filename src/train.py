import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# fraud is rare so accuracy is useless — a model predicting all-legit gets 99.8% accuracy
# we care about: did we catch the fraud (recall), were our fraud flags correct (precision)
def get_metrics(y_true, y_pred, y_prob):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

mlflow.set_experiment("fraud-detection")

# scale_pos_weight tells xgboost how much rarer fraud is
# so it penalizes missing a fraud case more than a false alarm
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos
print(f"scale_pos_weight = {spw:.1f}")

# --- run 1: xgboost ---
with mlflow.start_run(run_name="xgboost-baseline"):
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "scale_pos_weight": spw,
        "eval_metric": "aucpr",
        "random_state": 42
    }
    model = XGBClassifier(**params, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_pred, y_prob)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "model")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_baseline.joblib")

    print("\n--- XGBoost ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# --- run 2: logistic regression as challenger ---
with mlflow.start_run(run_name="logistic-regression-challenger"):
    params_lr = {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",  # equivalent to scale_pos_weight for sklearn
        "random_state": 42
    }
    lr = LogisticRegression(**params_lr)
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    metrics_lr = get_metrics(y_test, y_pred_lr, y_prob_lr)

    mlflow.log_params(params_lr)
    mlflow.log_metrics(metrics_lr)
    mlflow.sklearn.log_model(lr, "model")

    joblib.dump(lr, "models/logistic_regression_challenger.joblib")

    print("\n--- Logistic Regression ---")
    for k, v in metrics_lr.items():
        print(f"  {k}: {v:.4f}")
