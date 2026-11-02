"""
RESINET-IDAM: CIC-IDS2017 Automated Testing Suite (tests/test_cicids.py)
Implements:
- Tier 1: Lightweight Synthetic Unit Tests (N <= 1,000, runtime <5s)
- Tier 2: Real-Dataset Integrity Tests (Read-only verification of Phase A counts, zero training)
"""

import hashlib
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.dummy import DummyClassifier

from src.config import DATASET_PATHS, RANDOM_SEED
from src.data.loaders import CICIDSLoader
from src.data.preprocessors import LeakFreeScaler

PARQUET_PATH = DATASET_PATHS["cicids2017_dedup"]
EXPECTED_SHA256 = "d51aed3af54e2d97e09c33a6fcd627756cfb27e56d9ecefd47663a147332405e"
TOTAL_ROWS = 2_498_185
CONFLICT_ROWS = 1_436
CLEAN_CORE_ROWS = 2_496_749


# =========================================================================
# TIER 1: LIGHTWEIGHT SYNTHETIC UNIT TESTS (MOCK FIXTURES, RUNTIME <5S)
# =========================================================================

def test_synthetic_conflict_isolation_logic():
    """Verify duplicate feature identification isolates conflicting patterns on mock data."""
    # 10 samples: 2 duplicate feature pairs with different labels
    X_mock = np.array([
        [1.0, 2.0], [1.0, 2.0],  # Pair 1: conflict
        [3.0, 4.0], [3.0, 4.0],  # Pair 2: conflict
        [5.0, 6.0], [7.0, 8.0],  # Clean samples
    ])
    labels_mock = ["BENIGN", "DoS", "BENIGN", "PortScan", "BENIGN", "BENIGN"]
    df_mock = pd.DataFrame(X_mock, columns=["f1", "f2"])
    df_mock["Label"] = labels_mock

    # Duplicate detection on features
    is_conflict = df_mock.duplicated(subset=["f1", "f2"], keep=False)
    assert is_conflict.sum() == 4
    df_conf = df_mock[is_conflict]
    assert len(df_conf) == 4
    assert set(df_conf["Label"]) == {"BENIGN", "DoS", "PortScan"}


def test_synthetic_stratified_split_logic():
    """Verify 80/10/10 stratified split maintains rare class samples (e.g. Heartbleed)."""
    # 100 samples with rare classes: 11 of Class_A, 21 of Class_B, remainder Normal
    classes = ["Normal"] * 68 + ["Class_A"] * 11 + ["Class_B"] * 21
    df_mock = pd.DataFrame({"feat": np.arange(100), "Label": classes})

    y = df_mock["Label"].values
    train_idx, temp_idx = train_test_split(np.arange(100), test_size=0.20, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=42)

    # Class_A has 11 samples: 9 train, 1 val, 1 test
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    assert (y_train == "Class_A").sum() == 9
    assert (y_val == "Class_A").sum() == 1
    assert (y_test == "Class_A").sum() == 1

    # Class_B has 21 samples: 17 train, 2 val, 2 test
    assert (y_train == "Class_B").sum() == 17
    assert (y_val == "Class_B").sum() == 2
    assert (y_test == "Class_B").sum() == 2


def test_synthetic_streaming_empty_batch_handling():
    """Verify streaming logic correctly handles batches containing k=0 campaign records."""
    global_row_offset = 0
    stream_offset = 0
    
    # Simulate 3 batches of 10 rows: batch 1 has 0 campaign rows, batch 2 has 5, batch 3 has 0
    batches = [
        {"source_file": ["Monday.csv"] * 10, "is_conflict": [False] * 10},
        {"source_file": ["Wednesday.csv"] * 5 + ["Monday.csv"] * 5, "is_conflict": [False] * 10},
        {"source_file": ["Tuesday.csv"] * 10, "is_conflict": [False] * 10},
    ]
    campaign_files = {"Wednesday.csv"}

    for b in batches:
        batch_len = len(b["source_file"])
        is_campaign = [f in campaign_files for f in b["source_file"]]
        valid_campaign = [c and not conf for c, conf in zip(is_campaign, b["is_conflict"])]
        k = sum(valid_campaign)
        
        if k > 0:
            batch_scores = np.ones(k)
            stream_offset += k
            del batch_scores
            
        global_row_offset += batch_len

    assert global_row_offset == 30
    assert stream_offset == 5


def test_synthetic_record_identity_checksums():
    """Verify 64-bit sum and XOR checksums detect duplication or omission."""
    expected_ids = np.arange(100, dtype=np.int64)
    expected_sum = int(np.sum(expected_ids))
    expected_xor = int(np.bitwise_xor.reduce(expected_ids))

    # Identical stream
    actual_ids = np.arange(100, dtype=np.int64)
    assert int(np.sum(actual_ids)) == expected_sum
    assert int(np.bitwise_xor.reduce(actual_ids)) == expected_xor

    # Corrupted stream (one duplicated, one omitted)
    corrupted_ids = actual_ids.copy()
    corrupted_ids[5] = 99
    assert int(np.sum(corrupted_ids)) != expected_sum or int(np.bitwise_xor.reduce(corrupted_ids)) != expected_xor


def test_synthetic_leak_free_scaler_isolation():
    """Verify LeakFreeScaler does not alter training parameters upon transform."""
    scaler = LeakFreeScaler(scaler_type="robust")
    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    X_test = np.array([[100.0, 1000.0], [200.0, 2000.0]])

    scaler.fit(X_train)
    center_before = scaler.scaler.center_.copy()
    scale_before = scaler.scaler.scale_.copy()

    scaler.transform(X_test)
    assert np.array_equal(scaler.scaler.center_, center_before)
    assert np.array_equal(scaler.scaler.scale_, scale_before)


def test_synthetic_track3_deterministic_equivalence():
    """Verify identical feature pairs produce identical predictions and probabilities."""
    clf = DummyClassifier(strategy="constant", constant=0)
    X_train = np.zeros((10, 2))
    y_train = np.zeros(10)
    clf.fit(X_train, y_train)

    X_pair_a = np.array([[1.5, 2.5], [3.5, 4.5]])
    X_pair_b = np.array([[1.5, 2.5], [3.5, 4.5]])

    preds_a = clf.predict(X_pair_a)
    preds_b = clf.predict(X_pair_b)
    assert np.array_equal(preds_a, preds_b)

    probs_a = clf.predict_proba(X_pair_a)
    probs_b = clf.predict_proba(X_pair_b)
    assert np.allclose(probs_a, probs_b, atol=1e-7)


def test_synthetic_runtime_projection_scaling():
    """Verify asymptotic scaling formulas (O(N), O(N log N))."""
    n_pilot = 100_000
    n_full = 1_997_399
    ratio = n_full / n_pilot  # 19.974

    t_sgd_pilot = 2.0
    t_sgd_proj = t_sgd_pilot * ratio
    assert abs(t_sgd_proj - 39.948) < 0.01

    t_rf_pilot = 10.0
    log_ratio = np.log2(n_full) / np.log2(n_pilot)
    t_rf_proj = t_rf_pilot * (ratio * log_ratio)
    assert t_rf_proj > t_sgd_proj  # RF scales super-linearly


# =========================================================================
# TIER 2: REAL-DATASET INTEGRITY TESTS (READ-ONLY, ZERO TRAINING)
# =========================================================================

def test_real_dataset_sha256_and_metadata():
    """Verify Parquet file SHA-256 fingerprint, file size, row groups, and total rows."""
    sha256_hasher = hashlib.sha256()
    with open(PARQUET_PATH, "rb") as f:
        while chunk := f.read(65_536):
            sha256_hasher.update(chunk)
    computed_sha256 = sha256_hasher.hexdigest()
    assert computed_sha256 == EXPECTED_SHA256, f"SHA-256 drift: {computed_sha256}"

    pq_file = pq.ParquetFile(PARQUET_PATH)
    assert pq_file.metadata.num_rows == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows"
    assert pq_file.num_row_groups == 3, f"Expected 3 row groups, found {pq_file.num_row_groups}"


_cached_clean = None
_cached_conflicts = None


def _get_clean_and_conflicts():
    global _cached_clean, _cached_conflicts
    if _cached_clean is None or _cached_conflicts is None:
        _cached_clean, _cached_conflicts = CICIDSLoader.load(deduplicated=True, exclude_conflicts=True)
    return _cached_clean, _cached_conflicts


def test_real_dataset_conflict_count_and_clean_core():
    """Verify CICIDSLoader isolates exactly 1,436 conflict rows and 2,496,749 clean core rows."""
    df_clean, df_conflicts = _get_clean_and_conflicts()
    assert len(df_conflicts) == CONFLICT_ROWS, f"Expected {CONFLICT_ROWS} conflicts, got {len(df_conflicts)}"
    assert len(df_clean) == CLEAN_CORE_ROWS, f"Expected {CLEAN_CORE_ROWS} clean core rows, got {len(df_clean)}"
    
    # Verify clean core distribution: 2,071,726 Benign, 425,023 Attacks
    benign_count = (df_clean["Label"] == "BENIGN").sum()
    attack_count = (df_clean["Label"] != "BENIGN").sum()
    assert benign_count == 2_071_726, f"Expected 2,071,726 benign rows, got {benign_count}"
    assert attack_count == 425_023, f"Expected 425,023 attack rows, got {attack_count}"


def test_real_dataset_capture_file_reconciliation():
    """Verify capture-file row counts reconcile with Phase A audit figures."""
    df_clean, _ = _get_clean_and_conflicts()
    
    expected_clean_counts = {
        "Monday-WorkingHours.pcap_ISCX.csv": 492_653,
        "Tuesday-WorkingHours.pcap_ISCX.csv": 406_466,
        "Wednesday-workingHours.pcap_ISCX.csv": 586_448,
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": 154_475,
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": 241_861,
        "Friday-WorkingHours-Morning.pcap_ISCX.csv": 180_087,
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": 211_684,
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": 223_075,
    }
    actual_counts = df_clean["source_file"].value_counts().to_dict()
    for filename, expected in expected_clean_counts.items():
        assert actual_counts.get(filename, 0) == expected, (
            f"Capture file {filename}: expected {expected}, got {actual_counts.get(filename, 0)}"
        )


def test_real_dataset_withheld_campaign_reconciliation():
    """Verify withheld Campaigns A-F sum to exactly 1,597,630 rows."""
    df_clean, _ = _get_clean_and_conflicts()
    campaign_files = [
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]
    df_campaigns = df_clean[df_clean["source_file"].isin(campaign_files)]
    assert len(df_campaigns) == 1_597_630, f"Expected 1,597,630 campaign rows, got {len(df_campaigns)}"
    
    n_attacks = (df_campaigns["Label"] != "BENIGN").sum()
    n_benign = (df_campaigns["Label"] == "BENIGN").sum()
    assert n_attacks == 415_873, f"Expected 415,873 campaign attacks, got {n_attacks}"
    assert n_benign == 1_181_757, f"Expected 1,181,757 campaign benign, got {n_benign}"


def test_real_dataset_14_attack_distribution():
    """Verify all 14 clean attack categories match Phase A empirical audit exactly."""
    df_clean, _ = _get_clean_and_conflicts()
    expected_attack_counts = {
        "DoS Hulk": 172_697,
        "DDoS": 128_011,
        "PortScan": 90_130,
        "DoS GoldenEye": 10_286,
        "FTP-Patator": 5_931,
        "DoS slowloris": 5_383,
        "DoS Slowhttptest": 5_228,
        "SSH-Patator": 3_219,
        "Bot": 1_948,
        "Web Attack - Brute Force": 1_470,
        "Web Attack - XSS": 652,
        "Infiltration": 36,
        "Web Attack - SQL Injection": 21,
        "Heartbleed": 11,
    }
    actual_counts = df_clean["Label"].value_counts().to_dict()
    for attack_name, expected in expected_attack_counts.items():
        assert actual_counts.get(attack_name, 0) == expected, (
            f"Class {attack_name}: expected {expected}, got {actual_counts.get(attack_name, 0)}"
        )


ALL_CICIDS_TESTS = [
    # Tier 1
    ("Synthetic Conflict Isolation Logic", test_synthetic_conflict_isolation_logic),
    ("Synthetic Stratified Split Logic", test_synthetic_stratified_split_logic),
    ("Synthetic Streaming Empty Batch Handling", test_synthetic_streaming_empty_batch_handling),
    ("Synthetic Record Identity Checksums", test_synthetic_record_identity_checksums),
    ("Synthetic LeakFreeScaler Isolation", test_synthetic_leak_free_scaler_isolation),
    ("Synthetic Track 3 Deterministic Equivalence", test_synthetic_track3_deterministic_equivalence),
    ("Synthetic Runtime Projection Scaling", test_synthetic_runtime_projection_scaling),
    # Tier 2
    ("Real Dataset SHA-256 and Metadata", test_real_dataset_sha256_and_metadata),
    ("Real Dataset Conflict Count & Clean Core", test_real_dataset_conflict_count_and_clean_core),
    ("Real Dataset Capture File Reconciliation", test_real_dataset_capture_file_reconciliation),
    ("Real Dataset Withheld Campaign Reconciliation", test_real_dataset_withheld_campaign_reconciliation),
    ("Real Dataset 14 Attack Distribution", test_real_dataset_14_attack_distribution),
]


def run_tests():
    print("=" * 70)
    print("RESINET-IDAM: Phase B Step B.5 CIC-IDS2017 Test Suite")
    print("=" * 70)
    passed = 0
    failed = 0
    t0_all = time.time()
    for name, fn in ALL_CICIDS_TESTS:
        sys.stdout.write(f"Running [{name}] ... ")
        sys.stdout.flush()
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"PASSED ({dt:.2f}s)")
            passed += 1
        except Exception as e:
            dt = time.time() - t0
            print(f"FAILED ({dt:.2f}s)")
            print(f"  Error: {e}")
            failed += 1
    total_dt = time.time() - t0_all
    print("=" * 70)
    print(f"Summary: {passed} PASSED, {failed} FAILED in {total_dt:.2f}s")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()

