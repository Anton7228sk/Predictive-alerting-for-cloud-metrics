import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import boto3
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
        self._storage_config = config.get("storage", {})

    def _build_model(self) -> EnsembleModel:
        weights = self._model_config.get("ensemble_weights", {})
        iso_weight = weights.get("isolation_forest", 0.3)
        xgb_weight = weights.get("xgboost", 0.7)

        iso_params = self._model_config.get("isolation_forest", {})
        if isinstance(iso_params.get("contamination"), str):
            iso_params = dict(iso_params)

        xgb_params = self._model_config.get("xgboost", {})

        return EnsembleModel(
            iso_weight=iso_weight,
            xgb_weight=xgb_weight,
            iso_params=iso_params or None,
            xgb_params=xgb_params or None,
        )

    def _save_artifacts_local(self, model: EnsembleModel, metrics: dict) -> Path:
        local_path = Path(self._storage_config.get("local_fallback", "./artifacts"))
        model_dir = local_path / "model"
        model.save(model_dir)

        with open(local_path / "metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

        logger.info("Artifacts saved locally to %s", local_path)
        return local_path

    def _save_artifacts_s3(self, model: EnsembleModel, metrics: dict) -> str:
        bucket = self._storage_config["bucket"]
        prefix = self._storage_config.get("prefix", "models/")

        local_path = self._save_artifacts_local(model, metrics)
        s3 = boto3.client("s3")

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                key = prefix + str(file_path.relative_to(local_path)).replace("\\", "/")
                s3.upload_file(str(file_path), bucket, key)
                logger.info("Uploaded %s to s3://%s/%s", file_path.name, bucket, key)

        return f"s3://{bucket}/{prefix}"

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        logger.info("Starting training pipeline with %d rows", len(df))

        windows = self._feature_config.get("windows", [5, 15, 30, 60])
        lags = self._feature_config.get("lags", [1, 5, 15, 30])
        lookahead = self._feature_config.get("lookahead_minutes", 15)

        features = create_features(df, windows=windows, lags=lags)
        targets = create_targets(df.loc[features.index], lookahead_minutes=lookahead)

        aligned_features = features.loc[targets.index]
        aligned_targets = targets.loc[features.index]

        train_features, test_features = split_temporal(aligned_features, test_ratio=0.2)
        train_targets = aligned_targets.loc[train_features.index]
        test_targets = aligned_targets.loc[test_features.index]

        logger.info(
            "Train: %d rows (%d positive) | Test: %d rows (%d positive)",
            len(train_features),
            int(train_targets.sum()),
            len(test_features),
            int(test_targets.sum()),
        )

        model = self._build_model()
        model.fit(train_features, train_targets)

        test_scores = model.predict_proba(test_features)
        report = full_evaluation_report(
            y_true=test_targets,
            scores=test_scores,
            timestamps=test_features.index,
        )

        logger.info("Evaluation results: %s", report)

        storage_type = self._storage_config.get("type", "local")
        try:
            if storage_type == "s3":
                artifact_location = self._save_artifacts_s3(model, report)
            else:
                artifact_location = str(self._save_artifacts_local(model, report))
        except Exception as exc:
            logger.warning("Primary storage failed (%s), falling back to local: %s", storage_type, exc)
            artifact_location = str(self._save_artifacts_local(model, report))

        return {
            "metrics": report,
            "artifact_location": artifact_location,
            "train_size": len(train_features),
            "test_size": len(test_features),
        }
