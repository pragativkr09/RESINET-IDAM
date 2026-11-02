"""
RESINET-IDAM: Unsupervised Anomaly Model Wrappers Module (src/models/anomaly.py)
Provides standard wrappers for unsupervised telemetry anomaly detection:
- IsolationForestBaseline (tree-based isolation algorithm)
- OneClassSVMBaseline (kernel support vector boundary estimation with controlled subsampling)

Key design principles:
- Anomaly scores are standardized so that HIGHER SCORE = MORE ANOMALOUS.
- Baseline thresholds (tau_95, tau_99) are calibrated strictly on training anomaly scores.
- Support controlled representative subsampling for One-Class SVM to guarantee O(N) memory/time scalability.
- Clean serialization and deserialization via joblib.
"""

import time
import tracemalloc
from typing import Dict, Any, Optional
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator


class BaseAnomalyWrapper:
    """Base class providing consistent fit, scoring, threshold calibration, and serialization."""

    def __init__(self, name: str, estimator: BaseEstimator):
        self.name = name
        self.estimator = estimator
        self.fit_time: float = 0.0
        self.score_time: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.thresholds: Dict[str, float] = {}
        self.params: Dict[str, Any] = estimator.get_params()

    def fit(self, X: np.ndarray):
        """Fits estimator on training partition while tracking time and memory."""
        tracemalloc.start()
        t0 = time.time()

        self._fit_internal(X)

        self.fit_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / (1024 * 1024)
        return self

    def _fit_internal(self, X: np.ndarray):
        self.estimator.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Computes anomaly score for each sample.
        Convention: HIGHER score = MORE ANOMALOUS.
        """
        raise NotImplementedError

    def calibrate_thresholds(self, scores_train: np.ndarray) -> Dict[str, float]:
        """
        Calibrates empirical anomaly score thresholds strictly on the baseline training scores.
        tau_95: 95th percentile (nominal 5% false alarm rate on training baseline)
        tau_99: 99th percentile (nominal 1% false alarm rate on training baseline)
        """
        self.thresholds = {
            "tau_90": float(np.percentile(scores_train, 90)),
            "tau_95": float(np.percentile(scores_train, 95)),
            "tau_99": float(np.percentile(scores_train, 99)),
        }
        return self.thresholds

    def predict_anomaly(self, X: np.ndarray, threshold_key: str = "tau_95") -> np.ndarray:
        """Flags samples exceeding the specified calibrated anomaly threshold."""
        if threshold_key not in self.thresholds:
            raise ValueError(f"Threshold '{threshold_key}' not calibrated. Calibrated keys: {list(self.thresholds.keys())}")
        scores = self.score_samples(X)
        threshold = self.thresholds[threshold_key]
        return (scores >= threshold).astype(int)

    def save(self, filepath: Path) -> Path:
        """Serializes trained model artifact using joblib."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: Path) -> "BaseAnomalyWrapper":
        """Loads serialized model artifact."""
        return joblib.load(filepath)


class IsolationForestBaseline(BaseAnomalyWrapper):
    """
    Isolation Forest baseline for tabular telemetry anomaly detection.
    Scikit-learn's score_samples returns opposite of anomaly score (lower is more anomalous).
    We invert the sign so HIGHER = MORE ANOMALOUS.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        estimator = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        super().__init__(name="IsolationForest", estimator=estimator)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Invert scikit-learn sign: higher score = more anomalous
        return -self.estimator.score_samples(X)


class OneClassSVMBaseline(BaseAnomalyWrapper):
    """
    One-Class Support Vector Machine with RBF kernel.
    To avoid quadratic memory and runtime blowup on large telemetry datasets (>100k rows),
    supports controlled representative subsampling of training samples.
    Scikit-learn decision_function is positive inside the boundary and negative outside.
    We invert the sign so HIGHER = MORE ANOMALOUS.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.03,
        max_train_samples: int = 10000,
        random_state: int = 42,
    ):
        estimator = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )
        super().__init__(name="OneClassSVM", estimator=estimator)
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.n_samples_trained_: int = 0
        self.n_support_vectors_: int = 0

    def _fit_internal(self, X: np.ndarray):
        if len(X) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            sub_idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X_fit = X[sub_idx]
        else:
            X_fit = X

        self.estimator.fit(X_fit)
        self.n_samples_trained_ = len(X_fit)
        self.n_support_vectors_ = len(self.estimator.support_)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Invert decision function: higher score = more anomalous
        return -self.estimator.decision_function(X)

