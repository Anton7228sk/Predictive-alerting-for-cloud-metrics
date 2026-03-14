import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_dataset
from src.data.preprocessing import create_features, create_targets
from src.models.ensemble import EnsembleModel
from src.models.isolation_forest import IsolationForestModel
from src.models.xgboost_model import XGBoostModel


@pytest.fixture(scope="module")
def feature_target_pair():
    df = generate_dataset(n_days=10, freq_minutes=5, seed=1)
    features = create_features(df, windows=[5, 15], lags=[1, 5])
    targets = create_targets(df.loc[features.index], lookahead_minutes=15)
    aligned = features.loc[targets.index]
    return aligned, targets


def test_isolation_forest_fit_predict(feature_target_pair):
    X, y = feature_target_pair
    model = IsolationForestModel(n_estimators=10, random_state=0)
    model.fit(X, y)
    scores = model.predict_proba(X)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_isolation_forest_not_fitted_raises():
    model = IsolationForestModel()
    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(X)


def test_isolation_forest_save_load(feature_target_pair):
    X, y = feature_target_pair
    model = IsolationForestModel(n_estimators=10, random_state=0)
    model.fit(X, y)
    scores_before = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "iso.pkl"
        model.save(save_path)

        loaded = IsolationForestModel()
        loaded.load(save_path)
        scores_after = loaded.predict_proba(X)

    np.testing.assert_array_almost_equal(scores_before, scores_after)


def test_xgboost_fit_predict(feature_target_pair):
    X, y = feature_target_pair
    model = XGBoostModel(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, y)
    scores = model.predict_proba(X)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_xgboost_not_fitted_raises():
    model = XGBoostModel()
    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(X)


def test_xgboost_save_load(feature_target_pair):
    X, y = feature_target_pair
    model = XGBoostModel(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, y)
    scores_before = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "xgb.pkl"
        model.save(save_path)

        loaded = XGBoostModel()
        loaded.load(save_path)
        scores_after = loaded.predict_proba(X)

    np.testing.assert_array_almost_equal(scores_before, scores_after)


def test_ensemble_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        EnsembleModel(iso_weight=0.5, xgb_weight=0.8)


def test_ensemble_fit_predict(feature_target_pair):
    X, y = feature_target_pair
    model = EnsembleModel(
        iso_weight=0.3,
        xgb_weight=0.7,
        iso_params={"n_estimators": 10},
        xgb_params={"n_estimators": 50, "max_depth": 3},
    )
    model.fit(X, y)
    scores = model.predict_proba(X)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_ensemble_predict_binary(feature_target_pair):
    X, y = feature_target_pair
    model = EnsembleModel(
        iso_params={"n_estimators": 10},
        xgb_params={"n_estimators": 50, "max_depth": 3},
    )
    model.fit(X, y)
    preds = model.predict(X, threshold=0.5)
    assert set(preds).issubset({0, 1})


def test_ensemble_save_load(feature_target_pair):
    X, y = feature_target_pair
    model = EnsembleModel(
        iso_params={"n_estimators": 10},
        xgb_params={"n_estimators": 50, "max_depth": 3},
    )
    model.fit(X, y)
    scores_before = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "ensemble"
        model.save(save_dir)

        loaded = EnsembleModel()
        loaded.load(save_dir)
        scores_after = loaded.predict_proba(X)

    np.testing.assert_array_almost_equal(scores_before, scores_after)


def test_ensemble_scores_are_weighted_combination(feature_target_pair):
    X, y = feature_target_pair

    iso_model = IsolationForestModel(n_estimators=10, random_state=42)
    iso_model.fit(X, y)
    iso_scores = iso_model.predict_proba(X)

    xgb_model = XGBoostModel(n_estimators=50, max_depth=3, random_state=42)
    xgb_model.fit(X, y)
    xgb_scores = xgb_model.predict_proba(X)

    ensemble = EnsembleModel(
        iso_weight=0.3,
        xgb_weight=0.7,
        iso_params={"n_estimators": 10, "random_state": 42},
        xgb_params={"n_estimators": 50, "max_depth": 3, "random_state": 42},
    )
    ensemble.fit(X, y)
    ensemble_scores = ensemble.predict_proba(X)

    expected = 0.3 * iso_scores + 0.7 * xgb_scores
    np.testing.assert_allclose(ensemble_scores, expected, rtol=1e-5)
