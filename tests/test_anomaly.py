"""
RESINET-IDAM: Anomaly Model Unit Tests (tests/test_anomaly.py)
Tests unsupervised anomaly wrappers and baseline calibration logic.
"""

import numpy as np
from src.models.anomaly import IsolationForestBaseline, OneClassSVMBaseline
from src.data.preprocessors import LeakFreeScaler


def test_isolation_forest_wrapper():
    """Verify IsolationForestBaseline fits, computes inverted scores, and calibrates thresholds."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(1000, 3)
    X_test = rng.randn(200, 3)

    model = IsolationForestBaseline(n_estimators=20, random_state=42)
    model.fit(X_train)

    scores_train = model.score_samples(X_train)
    scores_test = model.score_samples(X_test)

    assert len(scores_train) == 1000
    assert len(scores_test) == 200

    thresholds = model.calibrate_thresholds(scores_train)
    assert "tau_90" in thresholds
    assert "tau_95" in thresholds
    assert "tau_99" in thresholds
    assert thresholds["tau_90"] <= thresholds["tau_95"] <= thresholds["tau_99"]

    preds = model.predict_anomaly(X_test, "tau_95")
    assert len(preds) == 200
    assert set(np.unique(preds)).issubset({0, 1})


def test_one_class_svm_subsampling():
    """Verify OneClassSVMBaseline subsamples large training sets and scales gracefully."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(5000, 2)
    X_test = rng.randn(500, 2)

    model = OneClassSVMBaseline(max_train_samples=500, random_state=42)
    model.fit(X_train)

    assert model.n_samples_trained_ == 500
    scores_train = model.score_samples(X_train)
    scores_test = model.score_samples(X_test)

    assert len(scores_train) == 5000
    assert len(scores_test) == 500

    thresholds = model.calibrate_thresholds(scores_train)
    assert thresholds["tau_90"] <= thresholds["tau_95"] <= thresholds["tau_99"]


def test_ton_iot_leak_free_scaling():
    """Verify that LeakFreeScaler fits strictly on baseline train partition."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(1000, 2) * 5.0 + 10.0
    X_test = rng.randn(500, 2) * 100.0 + 500.0  # radically different distribution

    scaler = LeakFreeScaler(scaler_type="standard")
    scaler.fit(X_train)

    # Learned parameters should match X_train, not X_test
    np.testing.assert_allclose(scaler.scaler.mean_, np.mean(X_train, axis=0), rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler.scale_, np.std(X_train, axis=0), rtol=1e-5)

