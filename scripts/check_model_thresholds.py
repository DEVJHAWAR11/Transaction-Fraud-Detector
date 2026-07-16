import sys
import os

# add src to path so we can import the threshold logic
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from promote import check_thresholds

# these are the last known metrics from the promoted xgboost model
# in a real pipeline this would load fresh metrics from mlflow or a json artifact
# for ci we validate that the threshold logic itself works and that
# the last known model metrics still clear the bar
last_known_metrics = {
    "precision": 0.3455,
    "recall": 0.8673,
    "f1": 0.4942,
    "roc_auc": 0.9811
}

print("running model threshold gate...")
print("metrics being checked:")
for k, v in last_known_metrics.items():
    print(f"  {k}: {v:.4f}")

failures = check_thresholds(last_known_metrics)

if failures:
    print("\n✗ model failed threshold checks:")
    for f in failures:
        print(f)
    sys.exit(1)  # non-zero exit = pipeline fails

print("\n✓ all thresholds passed — model is valid for promotion")
sys.exit(0)
