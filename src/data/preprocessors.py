"""
RESINET-IDAM: Preprocessing Module
Provides leak-free transformer pipelines that strictly enforce:
- Fitting only on training partitions (X_train)
- Applying only .transform() to validation (X_val) and test (X_test)
- Clean isolation of feature scalers and encoders
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler


class LeakFreeScaler(BaseEstimator, TransformerMixin):
    """
    Enforces that scaling parameters (mean, std, median, IQR) are estimated
    STRICTLY from the training partition and never leak test information.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}. Choose 'robust' or 'standard'")
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None):
        """Fit scaler strictly on training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            
        # Clean infinite and NaN values defensively
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X_mat)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform validation or test data using parameters learned from training data."""
        if not self.is_fitted:
            raise RuntimeError("LeakFreeScaler must be fitted on X_train before transforming!")
            
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X_mat)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> np.ndarray:
        """Fit on X_train and transform X_train."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Path) -> Path:
        """Serializes scaler to disk."""
        import joblib
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: Path) -> "LeakFreeScaler":
        """Loads serialized scaler from disk."""
        import joblib
        return joblib.load(filepath)
