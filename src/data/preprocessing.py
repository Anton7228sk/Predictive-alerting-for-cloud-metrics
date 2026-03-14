import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METRIC_COLS = [
    "cpu_utilization",
    "memory_usage",
    "request_latency",
    "error_rate",
    "network_in",
    "network_out",
]

DEFAULT_WINDOWS = [5, 15, 30, 60]
DEFAULT_LAGS = [1, 5, 15, 30]


def _rolling_features(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    frames = []
    for w in windows:
        roll = df[cols].rolling(window=w, min_periods=1)
        mean = roll.mean().add_suffix(f"_mean_{w}m")
        std = roll.std().add_suffix(f"_std_{w}m")
        mn = roll.min().add_suffix(f"_min_{w}m")
        mx = roll.max().add_suffix(f"_max_{w}m")
        frames.extend([mean, std, mn, mx])
    return pd.concat(frames, axis=1)


def _lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    frames = []
    for lag in lags:
        lagged = df[cols].shift(lag).add_suffix(f"_lag_{lag}m")
        frames.append(lagged)
    return pd.concat(frames, axis=1)


def _rate_of_change_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    roc_1 = df[cols].diff(1).add_suffix("_roc_1m")
    roc_5 = df[cols].diff(5).add_suffix("_roc_5m")
    pct_1 = df[cols].pct_change(1).replace([np.inf, -np.inf], np.nan).add_suffix("_pct_1m")
    return pd.concat([roc_1, roc_5, pct_1], axis=1)


def _percentile_features(df: pd.DataFrame, cols: list[str], window: int = 60) -> pd.DataFrame:
    frames = []
    for col in cols:
        roll = df[col].rolling(window=window, min_periods=1)
        q25 = roll.quantile(0.25).rename(f"{col}_q25_{window}m")
        q75 = roll.quantile(0.75).rename(f"{col}_q75_{window}m")
        q95 = roll.quantile(0.95).rename(f"{col}_q95_{window}m")
        iqr = (q75 - q25).rename(f"{col}_iqr_{window}m")
        frames.extend([q25, q75, q95, iqr])
    return pd.concat(frames, axis=1)


def _time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=index,
    )


def create_features(
    df: pd.DataFrame,
    windows: list[int] = DEFAULT_WINDOWS,
    lags: list[int] = DEFAULT_LAGS,
    metric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = [c for c in METRIC_COLS if c in df.columns]

    if not metric_cols:
        raise ValueError("No metric columns found in DataFrame")

    parts = [
        df[metric_cols].copy(),
        _rolling_features(df, metric_cols, windows),
        _lag_features(df, metric_cols, lags),
        _rate_of_change_features(df, metric_cols),
        _percentile_features(df, metric_cols, window=max(windows)),
        _time_features(df.index),
    ]

    features = pd.concat(parts, axis=1)
    features = features.ffill().bfill()

    n_before = len(features)
    features = features.dropna()
    if len(features) < n_before:
        logger.debug("Dropped %d rows with remaining NaN values", n_before - len(features))

    return features


def create_targets(
    df: pd.DataFrame,
    lookahead_minutes: int = 15,
) -> pd.Series:
    if "incident" not in df.columns:
        raise ValueError("DataFrame must contain 'incident' column")

    incident = df["incident"].astype(bool)
    target = pd.Series(False, index=df.index, name="target")

    for i in range(len(df) - lookahead_minutes):
        if incident.iloc[i : i + lookahead_minutes + 1].any():
            target.iloc[i] = True

    return target


def split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
