"""
Unit Tests: Data Leakage Prevention (tests/test_leakage.py)
Validates that:
- OT-ICS GroupKFold produces ZERO session leakage across partitions
- UNSW-NB15 decontaminated test set has ZERO feature overlap with D5
- ChronologicalSplitter strictly preserves the arrow of time (t_train < t_val < t_test)
- LeakFreeScaler cannot transform before fitting and learns parameters only from X_train
"""

import numpy as np
import pandas as pd
from src.data.loaders import OTICSLoader, UNSWLoader
from src.data.splitters import (
    SessionGroupSplitter,
    ChronologicalSplitter,
)
from src.data.preprocessors import LeakFreeScaler


def test_otics_session_leakage_zero():
    """Verify OT-ICS 5-fold cross-validation has exactly ZERO overlapping sessions."""
    df = OTICSLoader.load()
    splitter = SessionGroupSplitter(n_splits=5)
    
    for fold, train_idx, test_idx in splitter.split(df):
        train_sessions = set(df.iloc[train_idx]["global_session_id"])
        test_sessions = set(df.iloc[test_idx]["global_session_id"])
        overlap = train_sessions.intersection(test_sessions)
        assert len(overlap) == 0, f"Fold {fold} has {len(overlap)} overlapping sessions!"


def test_unsw_decontaminated_leakage_zero():
    """Verify decontaminated D8 has exactly ZERO overlapping feature vectors with D5."""
    df_d5, _, df_d8_decontam = UNSWLoader.load()
    feature_cols = UNSWLoader.get_feature_columns(df_d5)
    
    d5_unique_feats = df_d5[feature_cols].drop_duplicates()
    merged = df_d8_decontam.merge(d5_unique_feats, on=feature_cols, how="inner")
    
    assert len(merged) == 0, f"Decontaminated D8 contains {len(merged)} leaking rows matching D5!"


def test_chronological_temporal_integrity():
    """Verify ChronologicalSplitter strictly preserves temporal order and never leaks future data."""
    # Create synthetic chronological timeseries
    n = 1000
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "source_file": ["day1.csv"] * 500 + ["day2.csv"] * 500,
        "feature1": np.random.randn(n),
        "Label": ["BENIGN"] * 800 + ["ATTACK"] * 200,
    })
    
    train_df, val_df, test_df, audit = ChronologicalSplitter.split_intra_day(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        group_by_col="source_file", label_col="Label"
    )
    
    # Assert partition sizes
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150
    
    # Assert chronological boundaries for each day
    for grp in ["day1.csv", "day2.csv"]:
        sub_tr = train_df[train_df["source_file"] == grp]
        sub_va = val_df[val_df["source_file"] == grp]
        sub_te = test_df[test_df["source_file"] == grp]
        
        # Max timestamp in train must strictly precede min timestamp in val
        assert sub_tr["timestamp"].max() < sub_va["timestamp"].min()
        # Max timestamp in val must strictly precede min timestamp in test
        assert sub_va["timestamp"].max() < sub_te["timestamp"].min()


def test_leak_free_scaler_isolation():
    """Verify LeakFreeScaler estimates statistics strictly on training data."""
    scaler = LeakFreeScaler(scaler_type="standard")
    
    # Transform before fit must raise error
    raised = False
    try:
        scaler.transform(np.array([[1.0, 2.0]]))
    except RuntimeError:
        raised = True
    assert raised, "Expected RuntimeError when transforming before fit"
        
    X_train = np.array([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])
    X_test = np.array([[100.0, 200.0], [200.0, 400.0]])
    
    scaler.fit(X_train)
    assert np.allclose(scaler.scaler.mean_, [20.0, 40.0]), "Mean must match X_train mean exactly"
    
    # Transforming X_test must use X_train mean (20, 40)
    X_test_scaled = scaler.transform(X_test)
    expected_first_row = (X_test[0] - scaler.scaler.mean_) / scaler.scaler.scale_
    assert np.allclose(X_test_scaled[0], expected_first_row)
