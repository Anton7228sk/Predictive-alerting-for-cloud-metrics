import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_false_positive_rate,
    compute_lead_time,
    compute_precision_at_threshold,
    compute_recall_at_threshold,
    find_threshold_for_recall,
    full_evaluation_report,
)


def _make_test_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = pd.Series([False] * n, dtype=bool)
    y_true.iloc[50:60] = True
    y_true.iloc[120:135] = True

    scores = rng.uniform(0, 0.5, n)
    scores[45:62] = rng.uniform(0.7, 1.0, 17)
    scores[115:137] = rng.uniform(0.6, 0.95, 22)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    return y_true, scores, timestamps


@pytest.fixture
def test_data():
    return _make_test_data()


def test_recall_perfect(test_data):
    y_true, scores, _ = test_data
    recall = compute_recall_at_threshold(y_true, scores, threshold=0.1)
    assert recall == 1.0


def test_recall_zero(test_data):
    y_true, scores, _ = test_data
    recall = compute_recall_at_threshold(y_true, scores, threshold=0.99)
    assert recall == 0.0


def test_recall_no_incidents():
    y_true = pd.Series([False] * 50, dtype=bool)
    scores = np.random.uniform(0, 1, 50)
    recall = compute_recall_at_threshold(y_true, scores, threshold=0.5)
    assert recall == 0.0


def test_recall_partial(test_data):
    y_true, scores, _ = test_data
    recall = compute_recall_at_threshold(y_true, scores, threshold=0.65)
    assert 0.0 <= recall <= 1.0


def test_precision_at_threshold(test_data):
    y_true, scores, _ = test_data
    precision = compute_precision_at_threshold(y_true, scores, threshold=0.5)
    assert 0.0 <= precision <= 1.0


def test_precision_no_positives_predicted():
    y_true = pd.Series([True] * 10 + [False] * 90, dtype=bool)
    scores = np.zeros(100)
    precision = compute_precision_at_threshold(y_true, scores, threshold=0.5)
    assert precision == 0.0


def test_false_positive_rate(test_data):
    y_true, scores, _ = test_data
    fpr = compute_false_positive_rate(y_true, scores, threshold=0.5)
    assert 0.0 <= fpr <= 1.0


def test_false_positive_rate_no_negatives():
    y_true = pd.Series([True] * 50, dtype=bool)
    scores = np.ones(50)
    fpr = compute_false_positive_rate(y_true, scores, threshold=0.5)
    assert fpr == 0.0


def test_lead_time_returns_dict(test_data):
    y_true, scores, timestamps = test_data
    result = compute_lead_time(timestamps, y_true, scores, threshold=0.5)
    assert "mean_lead_time" in result
    assert "median_lead_time" in result
    assert "n_incidents" in result


def test_lead_time_positive(test_data):
    y_true, scores, timestamps = test_data
    result = compute_lead_time(timestamps, y_true, scores, threshold=0.5)
    if result["mean_lead_time"] is not None:
        assert result["mean_lead_time"] >= 0


def test_lead_time_no_pre_alerts():
    n = 100
    y_true = pd.Series([False] * 30 + [True] * 10 + [False] * 60, dtype=bool)
    scores = np.zeros(n)
    scores[70:80] = 0.9  # alerts fire well after the incident + lookahead window
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    result = compute_lead_time(timestamps, y_true, scores, threshold=0.5)
    assert result["n_with_lead_time"] == 0


def test_find_threshold_for_recall(test_data):
    y_true, scores, _ = test_data
    threshold = find_threshold_for_recall(y_true, scores, target_recall=0.8)
    assert 0.0 <= threshold <= 1.0
    recall = compute_recall_at_threshold(y_true, scores, threshold)
    assert recall >= 0.7


def test_find_threshold_no_incidents():
    y_true = pd.Series([False] * 100, dtype=bool)
    scores = np.random.uniform(0, 1, 100)
    threshold = find_threshold_for_recall(y_true, scores, target_recall=0.8)
    assert threshold == 0.5


def test_full_evaluation_report_keys(test_data):
    y_true, scores, timestamps = test_data
    report = full_evaluation_report(y_true, scores, timestamps)
    assert "roc_auc" in report
    assert "threshold_metrics" in report
    assert "at_optimal_threshold" in report
    assert "dataset_info" in report


def test_full_evaluation_report_dataset_info(test_data):
    y_true, scores, timestamps = test_data
    report = full_evaluation_report(y_true, scores, timestamps)
    info = report["dataset_info"]
    assert info["n_samples"] == len(y_true)
    assert info["n_incidents"] == 2
    assert 0.0 <= info["positive_rate"] <= 1.0


def test_full_evaluation_report_roc_auc(test_data):
    y_true, scores, timestamps = test_data
    report = full_evaluation_report(y_true, scores, timestamps)
    assert report["roc_auc"] is not None
    assert 0.0 <= report["roc_auc"] <= 1.0
