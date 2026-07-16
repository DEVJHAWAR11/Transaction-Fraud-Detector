import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from promote import check_thresholds, THRESHOLDS

def test_passes_all_thresholds():
    metrics = {"recall": 0.90, "precision": 0.40, "f1": 0.55, "roc_auc": 0.97}
    failures = check_thresholds(metrics)
    assert failures == []

def test_fails_recall():
    metrics = {"recall": 0.50, "precision": 0.40, "f1": 0.55, "roc_auc": 0.97}
    failures = check_thresholds(metrics)
    assert any("recall" in f for f in failures)

def test_fails_precision():
    metrics = {"recall": 0.90, "precision": 0.05, "f1": 0.55, "roc_auc": 0.97}
    failures = check_thresholds(metrics)
    assert any("precision" in f for f in failures)

def test_fails_f1():
    metrics = {"recall": 0.90, "precision": 0.40, "f1": 0.10, "roc_auc": 0.97}
    failures = check_thresholds(metrics)
    assert any("f1" in f for f in failures)

def test_fails_roc_auc():
    metrics = {"recall": 0.90, "precision": 0.40, "f1": 0.55, "roc_auc": 0.70}
    failures = check_thresholds(metrics)
    assert any("roc_auc" in f for f in failures)

def test_fails_multiple_thresholds():
    # a bad model fails more than one threshold
    metrics = {"recall": 0.40, "precision": 0.02, "f1": 0.05, "roc_auc": 0.60}
    failures = check_thresholds(metrics)
    assert len(failures) == 4

def test_exact_threshold_boundary():
    # exactly at threshold should pass (>= not >)
    metrics = {
        "recall": THRESHOLDS["recall"],
        "precision": THRESHOLDS["precision"],
        "f1": THRESHOLDS["f1"],
        "roc_auc": THRESHOLDS["roc_auc"]
    }
    failures = check_thresholds(metrics)
    assert failures == []
