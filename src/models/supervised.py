"""
RESINET-IDAM: Supervised Model Wrappers Module (src/models/supervised.py)
Provides standard model wrappers for tabular supervised classification:
- RandomForestBaseline
- HistGradientBoostingBaseline (LightGBM histogram algorithm)
- SGDBaseline (Regularized linear classifier)
"""

import time
import tracemalloc
from typing import Dict, Any, Optional
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator


class BaseModelWrapper:
    """Base class providing consistent fit, predict, timing, and serialization."""

    def __init__(self, name: str, estimator: BaseEstimator):
        self.name = name
        self.estimator = estimator
        self.fit_time: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.params: Dict[str, Any] = estimator.get_params()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits estimator while tracking execution time and peak memory."""
        tracemalloc.start()
        t0 = time.time()
        
        self.estimator.fit(X, y)
        
        self.fit_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / (1024 * 1024)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates class predictions."""
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generates class probability estimates."""
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        elif hasattr(self.estimator, "decision_function"):
            # Platt scaling approximation via sigmoid on decision function
            df = self.estimator.decision_function(X)
            prob_1 = 1.0 / (1.0 + np.exp(-np.clip(df, -500, 500)))
            return np.column_stack([1.0 - prob_1, prob_1])
        else:
            raise NotImplementedError(f"Estimator {self.name} does not support probability estimation.")

    def save(self, filepath: Path) -> Path:
        """Serializes trained model artifact using joblib."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: Path) -> "BaseModelWrapper":
        """Loads serialized model artifact."""
        return joblib.load(filepath)


class RandomForestBaseline(BaseModelWrapper):
    """Random Forest classifier baseline for tabular network flows."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        super().__init__(name="RandomForest", estimator=estimator)


class HistGradientBoostingBaseline(BaseModelWrapper):
    """Histogram-based Gradient Boosting classifier baseline (LightGBM algorithm)."""

    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.1,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        estimator = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            class_weight=class_weight,
            random_state=random_state,
        )
        super().__init__(name="HistGradientBoosting", estimator=estimator)


class SGDBaseline(BaseModelWrapper):
    """Regularized linear classifier baseline using ElasticNet penalty."""

    def __init__(
        self,
        loss: str = "log_loss",
        penalty: str = "elasticnet",
        l1_ratio: float = 0.15,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        estimator = SGDClassifier(
            loss=loss,
            penalty=penalty,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
        super().__init__(name="SGDClassifier", estimator=estimator)

