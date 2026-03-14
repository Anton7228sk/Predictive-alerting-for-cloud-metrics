import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseModel
from .isolation_forest import IsolationForestModel
from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    def __init__(
        self,
        iso_weight: float = 0.3,
        xgb_weight: float = 0.7,
        iso_params: dict | None = None,
        xgb_params: dict | None = None,
    ) -> None:
        if abs(iso_weight + xgb_weight - 1.0) > 1e-6:
            raise ValueError("iso_weight + xgb_weight must equal 1.0")

        self.iso_weight = iso_weight
        self.xgb_weight = xgb_weight

        self.iso_model = IsolationForestModel(**(iso_params or {}))
        self.xgb_model = XGBoostModel(**(xgb_params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        logger.info("Fitting ensemble model")
        self.iso_model.fit(X, y)
        self.xgb_model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        iso_scores = self.iso_model.predict_proba(X)
        xgb_scores = self.xgb_model.predict_proba(X)
        return self.iso_weight * iso_scores + self.xgb_weight * xgb_scores

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.iso_model.save(path / "isolation_forest.pkl")
        self.xgb_model.save(path / "xgboost.pkl")

        meta = {
            "iso_weight": self.iso_weight,
            "xgb_weight": self.xgb_weight,
        }
        with open(path / "ensemble_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("Ensemble model saved to %s", path)

    def load(self, path: Path) -> "EnsembleModel":
        path = Path(path)

        with open(path / "ensemble_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        self.iso_weight = meta["iso_weight"]
        self.xgb_weight = meta["xgb_weight"]
        self.iso_model.load(path / "isolation_forest.pkl")
        self.xgb_model.load(path / "xgboost.pkl")

        logger.info("Ensemble model loaded from %s", path)
        return self
