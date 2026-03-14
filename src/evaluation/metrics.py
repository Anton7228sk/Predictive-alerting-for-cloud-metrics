import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)


def _get_incident_intervals(y_true: pd.Series) -> list[tuple[int, int]]:
    intervals = []
    in_incident = False
    start = 0
    arr = y_true.values.astype(bool)
    for i, val in enumerate(arr):
        if val and not in_incident:
            start = i
            in_incident = True
        elif not val and in_incident:
            intervals.append((start, i - 1))
            in_incident = False
    if in_incident:
        intervals.append((start, len(arr) - 1))
    return intervals


def compute_recall_at_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> float:
    intervals = _get_incident_intervals(y_true)
    if not intervals:
        return 0.0

    alerts = scores >= threshold
    detected = sum(1 for start, end in intervals if alerts[start : end + 1].any())
    return detected / len(intervals)


def compute_precision_at_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> float:
    predicted_positive = (scores >= threshold).sum()
    if predicted_positive == 0:
        return 0.0
    true_positive = ((scores >= threshold) & y_true.values.astype(bool)).sum()
    return true_positive / predicted_positive


def compute_lead_time(
    timestamps: pd.DatetimeIndex,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    intervals = _get_incident_intervals(y_true)
    alerts = scores >= threshold
    lead_times = []

    for start, end in intervals:
        alert_indices = np.where(alerts[:start])[0]
        if len(alert_indices) == 0:
            continue
        last_pre_alert_idx = alert_indices[-1]
        lead_td = timestamps[start] - timestamps[last_pre_alert_idx]
        lead_minutes = lead_td.total_seconds() / 60.0
        lead_times.append(lead_minutes)

    if not lead_times:
        return {
            "mean_lead_time": None,
            "median_lead_time": None,
            "distribution": [],
            "n_with_lead_time": 0,
            "n_incidents": len(intervals),
        }

    return {
        "mean_lead_time": float(np.mean(lead_times)),
        "median_lead_time": float(np.median(lead_times)),
        "distribution": lead_times,
        "n_with_lead_time": len(lead_times),
        "n_incidents": len(intervals),
    }


def compute_false_positive_rate(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> float:
    y_arr = y_true.values.astype(bool)
    negatives = (~y_arr).sum()
    if negatives == 0:
        return 0.0
    false_positives = ((scores >= threshold) & ~y_arr).sum()
    return false_positives / negatives


def find_threshold_for_recall(
    y_true: pd.Series,
    scores: np.ndarray,
    target_recall: float = 0.8,
) -> float:
    intervals = _get_incident_intervals(y_true)
    if not intervals:
        return 0.5

    best_threshold = 0.5
    best_diff = float("inf")

    for threshold in np.linspace(0.01, 0.99, 200):
        recall = compute_recall_at_threshold(y_true, scores, threshold)
        diff = abs(recall - target_recall)
        if diff < best_diff or (diff == best_diff and threshold > best_threshold):
            best_diff = diff
            best_threshold = threshold
            if diff == 0:
                break

    return float(best_threshold)


def full_evaluation_report(
    y_true: pd.Series,
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]

    report: dict[str, Any] = {}

    try:
        report["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        report["roc_auc"] = None

    opt_threshold = find_threshold_for_recall(y_true, scores, target_recall=0.8)
    report["optimal_threshold_for_recall_0.8"] = opt_threshold

    threshold_metrics = {}
    for t in thresholds:
        threshold_metrics[str(t)] = {
            "recall": compute_recall_at_threshold(y_true, scores, t),
            "precision": compute_precision_at_threshold(y_true, scores, t),
            "fpr": compute_false_positive_rate(y_true, scores, t),
            "lead_time": compute_lead_time(timestamps, y_true, scores, t),
        }
    report["threshold_metrics"] = threshold_metrics

    opt_recall = compute_recall_at_threshold(y_true, scores, opt_threshold)
    opt_fpr = compute_false_positive_rate(y_true, scores, opt_threshold)
    report["at_optimal_threshold"] = {
        "threshold": opt_threshold,
        "recall": opt_recall,
        "fpr": opt_fpr,
        "lead_time": compute_lead_time(timestamps, y_true, scores, opt_threshold),
    }

    n_incidents = len(_get_incident_intervals(y_true))
    report["dataset_info"] = {
        "n_samples": len(y_true),
        "n_incidents": n_incidents,
        "positive_rate": float(y_true.mean()),
    }

    return report
