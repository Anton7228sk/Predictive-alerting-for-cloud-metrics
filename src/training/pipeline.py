import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocessing import create_features, create_targets, split_temporal
from src.evaluation.metrics import full_evaluation_report
from src.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._feature_config = config.get("features", {})
        self._model_config = config.get("model", {})

    def _build_model(self) -> EnsembleModel:
        weights = self._model_config.get("ensemble_weights", {})
        iso_params = self._model_config.get("isolation_forest", {})
        xgb_params = self._model_config.get("xgboost", {})
        return EnsembleModel(
            iso_weight=weights.get("isolation_forest", 0.3),
            xgb_weight=weights.get("xgboost", 0.7),
            iso_params=iso_params or None,
            xgb_params=xgb_params or None,
        )

    def run(self, df: pd.DataFrame, artifacts_dir: str = "./artifacts") -> dict[str, Any]:
        logger.info("Starting training pipeline with %d rows", len(df))

        window_size = self._feature_config.get("window_size", 60)
        windows = self._feature_config.get("windows", [5, 15, 30, 60])
        lags = self._feature_config.get("lags", [1, 5, 15, 30])
        lookahead = self._feature_config.get("lookahead_minutes", 15)

        features = create_features(df, window_size=window_size, windows=windows, lags=lags)
        targets = create_targets(df.loc[features.index], lookahead_minutes=lookahead)
        features = features.loc[targets.index]
        targets = targets.loc[features.index]

        train_features, test_features = split_temporal(features, test_ratio=0.2)
        train_targets = targets.loc[train_features.index]
        test_targets = targets.loc[test_features.index]

        logger.info(
            "Train: %d rows (%d positive) | Test: %d rows (%d positive)",
            len(train_features), int(train_targets.sum()),
            len(test_features), int(test_targets.sum()),
        )

        model = self._build_model()
        model.fit(train_features, train_targets)

        default_threshold = self.config.get("alerting", {}).get("threshold", 0.65)
        test_scores = model.predict_proba(test_features)
        report = full_evaluation_report(
            y_true=test_targets,
            scores=test_scores,
            timestamps=test_features.index,
            lookahead_minutes=lookahead,
            default_threshold=default_threshold,
        )

        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "model")
        with open(out / "metrics.pkl", "wb") as f:
            pickle.dump(report, f)

        logger.info("Artifacts saved to %s", out)

        return {
            "metrics": report,
            "artifact_location": str(out),
            "train_size": len(train_features),
            "test_size": len(test_features),
        }
