import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model: XGBClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        pos = y.sum()
        neg = len(y) - pos
        scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

        logger.info(
            "Training XGBoost: %d positive, %d negative (scale_pos_weight=%.2f)",
            pos,
            neg,
            scale_pos_weight,
        )

        split = int(len(X) * 0.85)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="aucpr",
            random_state=self.random_state,
            use_label_encoder=False,
            verbosity=0,
        )

        self._model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self._model.best_iteration
        logger.info("XGBoost best iteration: %d", best_iter)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: Path) -> "XGBoostModel":
        path = Path(path)
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("XGBoost model loaded from %s", path)
        return self
