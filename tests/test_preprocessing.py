import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_dataset
from src.data.preprocessing import (
    create_features,
    create_targets,
    split_temporal,
)


@pytest.fixture(scope="module")
def sample_df():
    return generate_dataset(n_days=5, freq_minutes=1, seed=0)


def test_generate_dataset_shape(sample_df):
    assert len(sample_df) == 5 * 24 * 60
    assert "incident" in sample_df.columns
    assert sample_df.index.name == "timestamp"


def test_generate_dataset_has_incidents(sample_df):
    assert sample_df["incident"].any(), "Expected at least one incident in 5-day dataset"


def test_create_features_returns_dataframe(sample_df):
    features = create_features(sample_df, windows=[5, 15], lags=[1, 5])
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0


def test_create_features_no_nans(sample_df):
    features = create_features(sample_df, windows=[5, 15], lags=[1, 5])
    assert not features.isnull().any().any(), "Features should not contain NaN values"


def test_create_features_has_rolling_columns(sample_df):
    features = create_features(sample_df, windows=[5, 15], lags=[1])
    rolling_cols = [c for c in features.columns if "_mean_" in c or "_std_" in c]
    assert len(rolling_cols) > 0


def test_create_features_has_lag_columns(sample_df):
    features = create_features(sample_df, windows=[5], lags=[1, 5])
    lag_cols = [c for c in features.columns if "_lag_" in c]
    assert len(lag_cols) > 0


def test_create_features_has_time_cols(sample_df):
    features = create_features(sample_df, windows=[5], lags=[1])
    assert "hour_sin" in features.columns
    assert "hour_cos" in features.columns
    assert "dow_sin" in features.columns


def test_create_features_missing_incident_col(sample_df):
    df_no_incident = sample_df.drop(columns=["incident"])
    features = create_features(df_no_incident, windows=[5], lags=[1])
    assert len(features) > 0


def test_create_targets_length(sample_df):
    targets = create_targets(sample_df, lookahead_minutes=15)
    assert len(targets) == len(sample_df)


def test_create_targets_is_boolean(sample_df):
    targets = create_targets(sample_df, lookahead_minutes=15)
    assert targets.dtype == bool or set(targets.unique()).issubset({True, False})


def test_create_targets_lookahead_propagates(sample_df):
    targets_short = create_targets(sample_df, lookahead_minutes=5)
    targets_long = create_targets(sample_df, lookahead_minutes=30)
    assert targets_long.sum() >= targets_short.sum()


def test_create_targets_missing_incident_raises(sample_df):
    df_no_incident = sample_df.drop(columns=["incident"])
    with pytest.raises(ValueError, match="incident"):
        create_targets(df_no_incident, lookahead_minutes=15)


def test_split_temporal_sizes(sample_df):
    train, test = split_temporal(sample_df, test_ratio=0.2)
    total = len(sample_df)
    assert len(train) == int(total * 0.8)
    assert len(test) == total - int(total * 0.8)


def test_split_temporal_no_overlap(sample_df):
    train, test = split_temporal(sample_df, test_ratio=0.2)
    assert train.index.max() < test.index.min()


def test_split_temporal_default_ratio(sample_df):
    train, test = split_temporal(sample_df)
    total = len(sample_df)
    assert len(train) + len(test) == total


def test_split_temporal_preserves_order(sample_df):
    train, test = split_temporal(sample_df, test_ratio=0.3)
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
