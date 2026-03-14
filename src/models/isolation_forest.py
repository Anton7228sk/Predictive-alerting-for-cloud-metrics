import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .base import BaseModel

logger = logging.getLogger(__name__)


class IsolationForestModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model: IsolationForest | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IsolationForestModel":
        normal_mask = ~y.astype(bool)
        X_normal = X[normal_mask]

        logger.info(
            "Training IsolationForest on %d normal samples (of %d total)",
            len(X_normal),
            len(X),
        )

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_normal)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")

        raw_scores = self._model.score_samples(X)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s == min_s:
            return np.zeros(len(X))
        normalized = (raw_scores - min_s) / (max_s - min_s)
        return 1.0 - normalized

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("IsolationForest model saved to %s", path)

    def load(self, path: Path) -> "IsolationForestModel":
        path = Path(path)
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("IsolationForest model loaded from %s", path)
        return self
