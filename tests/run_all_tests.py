"""
RESINET-IDAM: Test Runner Module (tests/run_all_tests.py)
Executes all unit tests in tests/test_loaders.py and tests/test_leakage.py
using Python's built-in testing harness.
"""

import sys
import time
from pathlib import Path

# Add workspace to sys.path
sys.path.insert(0, str(Path(".").resolve()))

from tests.test_loaders import (
    test_otics_loader,
    test_unsw_loader,
    test_ton_iot_loader_pruning,
    test_cicids_loader_conflict_isolation,
)
from tests.test_leakage import (
    test_otics_session_leakage_zero,
    test_unsw_decontaminated_leakage_zero,
    test_chronological_temporal_integrity,
    test_leak_free_scaler_isolation,
)
from tests.test_anomaly import (
    test_isolation_forest_wrapper,
    test_one_class_svm_subsampling,
    test_ton_iot_leak_free_scaling,
)
from tests.test_cicids import ALL_CICIDS_TESTS
from tests.test_phase_c_ui import (
    test_benchmark_loader,
    test_sample_traffic_loader,
    test_model_artifact_loaders,
    test_multiclass_inference_engine,
    test_anomaly_inference_engine,
    test_pdf_report_generation,
    test_mysql_manager_buffering,
    test_styles_and_page_entry_points,
)

TESTS = [
    ("OT-ICS Loader Integrity & Sessions", test_otics_loader),
    ("UNSW-NB15 Loader & Decontaminated D8", test_unsw_loader),
    ("ToN_IoT Empty Row Pruning (34,876 Rows)", test_ton_iot_loader_pruning),
    ("CIC-IDS2017 Conflicting Pattern Isolation (1,436 Rows)", test_cicids_loader_conflict_isolation),
    ("OT-ICS GroupKFold Zero Session Leakage", test_otics_session_leakage_zero),
    ("UNSW-NB15 Decontaminated Zero Feature Leakage", test_unsw_decontaminated_leakage_zero),
    ("ChronologicalSplitter Strict Temporal Ordering", test_chronological_temporal_integrity),
    ("LeakFreeScaler Training-Only Estimation", test_leak_free_scaler_isolation),
    ("IsolationForest Anomaly Scoring & Thresholds", test_isolation_forest_wrapper),
    ("OneClassSVM Subsampling & Scoring", test_one_class_svm_subsampling),
    ("ToN_IoT Leak-Free Scaler Isolation", test_ton_iot_leak_free_scaling),
] + [(f"CIC-IDS2017: {name}", fn) for name, fn in ALL_CICIDS_TESTS] + [
    ("Phase C: Benchmark JSON Loader", test_benchmark_loader),
    ("Phase C: Sample Traffic NetFlow (60 Flows)", test_sample_traffic_loader),
    ("Phase C: Production Model & Scaler Artifacts", test_model_artifact_loaders),
    ("Phase C: Track 1 Multiclass Inference Engine", test_multiclass_inference_engine),
    ("Phase C: Track 2 Open-Set Anomaly Inference Engine", test_anomaly_inference_engine),
    ("Phase C: Dynamic PDF Security Report Engine", test_pdf_report_generation),
    ("Phase C: Direct MySQL Connector & Offline Buffer", test_mysql_manager_buffering),
    ("Phase C: Cyber Theme Styles & 9 Page Entry Points", test_styles_and_page_entry_points),
]


def run_tests():
    print("=" * 70)
    print("RESINET-IDAM: Comprehensive Test Suite (Phases A through B.5)")
    print("=" * 70)
    
    passed = 0
    failed = 0
    start_total = time.time()
    
    for name, test_fn in TESTS:
        sys.stdout.write(f"Running [{name}] ... ")
        sys.stdout.flush()
        t0 = time.time()
        try:
            test_fn()
            dt = time.time() - t0
            print(f"PASSED ({dt:.2f}s)")
            passed += 1
        except Exception as e:
            dt = time.time() - t0
            print(f"FAILED ({dt:.2f}s)")
            print(f"  Error: {e}")
            failed += 1
            
    total_dt = time.time() - start_total
    print("=" * 70)
    print(f"Test Summary: {passed} PASSED, {failed} FAILED in {total_dt:.2f}s")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()

