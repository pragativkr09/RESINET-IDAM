"""
RESINET-IDAM: Phase C UI and Integration Unit Tests (tests/test_phase_c_ui.py)
Validates UI utilities, multiclass & anomaly inference engines, PDF generation,
and MySQL direct connection/buffering behavior.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui import utils
from src.db.mysql_manager import MySQLManager


def test_benchmark_loader():
    """Verify loading of verified Phase B Step B.5 benchmark JSON."""
    bench = utils.load_benchmark_data()
    assert isinstance(bench, dict), "Benchmark data must be a dictionary"
    assert "track_1_closed_set_multiclass" in bench, "Track 1 multiclass section missing"
    assert "track_2_open_set_streaming" in bench, "Track 2 streaming section missing"
    macro_f1 = bench["track_1_closed_set_multiclass"]["test_metrics"]["macro_f1"]
    assert macro_f1 > 0.80, f"Expected Macro F1 > 0.80, got {macro_f1}"


def test_sample_traffic_loader():
    """Verify sample traffic integrity (60 rows, 80 columns)."""
    sample = utils.load_sample_traffic()
    assert len(sample) == 60, f"Expected 60 rows, got {len(sample)}"
    assert len(sample.columns) == 80, f"Expected 80 columns, got {len(sample.columns)}"
    assert "Label" in sample.columns, "Label column must exist"
    assert "Destination Port" in sample.columns, "Destination Port must exist"


def test_model_artifact_loaders():
    """Verify model and scaler joblib artifacts exist and load correctly."""
    clf, scaler = utils.load_multiclass_artifacts()
    assert clf is not None, "Classifier model must not be None"
    assert scaler is not None, "Classifier scaler must not be None"
    assert hasattr(clf, "predict"), "Classifier must have predict method"

    iso, iso_scaler = utils.load_anomaly_artifacts()
    assert iso is not None, "Isolation Forest model must not be None"
    assert iso_scaler is not None, "Isolation Forest scaler must not be None"
    assert hasattr(iso, "score_samples") or hasattr(iso, "predict_anomaly"), "Isolation Forest must have score_samples"


def test_multiclass_inference_engine():
    """Verify end-to-end Track 1 inference on sample flows."""
    sample = utils.load_sample_traffic()
    feature_cols = [c for c in sample.columns if c not in ["source_file", "Label"]]
    test_slice = sample[feature_cols].iloc[:10]

    results_df, summary = utils.run_multiclass_inference(test_slice)
    assert len(results_df) == 10, f"Expected 10 rows in result, got {len(results_df)}"
    assert "Predicted_Class" in results_df.columns, "Predicted_Class column missing"
    assert "Confidence" in results_df.columns, "Confidence column missing"
    assert "Threat_Severity" in results_df.columns, "Threat_Severity column missing"


def test_anomaly_inference_engine():
    """Verify end-to-end Track 2 anomaly scoring on sample flows."""
    sample = utils.load_sample_traffic()
    feature_cols = [c for c in sample.columns if c not in ["source_file", "Label"]]
    test_slice = sample[feature_cols].iloc[:10]

    results_df, summary = utils.run_anomaly_inference(test_slice, threshold=0.475982)
    assert len(results_df) == 10, f"Expected 10 rows in result, got {len(results_df)}"
    assert "Anomaly_Score" in results_df.columns, "Anomaly_Score column missing"
    assert "Is_Anomaly" in results_df.columns, "Is_Anomaly column missing"
    assert summary["total_flows"] == 10, "Summary total_flows mismatch"


def test_pdf_report_generation():
    """Verify programmatic PDF generation via ReportLab."""
    summary_data = {
        "timestamp": "2026-11-01 12:00:00 UTC",
        "auditor": "Test Harness",
        "total_flows": 60,
        "malicious_flows": 56,
        "benign_flows": 4,
        "macro_f1": 0.8108,
        "weighted_f1": 0.9984,
        "accuracy": 0.9985,
        "class_counts": {"BENIGN": 4, "DoS Hulk": 4, "DDoS": 4},
    }
    pdf_bytes = utils.generate_pdf_report(summary_data, report_title="CI/CD Test Report")
    assert isinstance(pdf_bytes, bytes), "PDF report must return bytes"
    assert len(pdf_bytes) > 2000, f"PDF byte count too small: {len(pdf_bytes)}"
    assert pdf_bytes.startswith(b"%PDF"), "Must start with standard PDF magic number"


def test_mysql_manager_buffering():
    """Verify direct MySQL connection check and in-memory queueing when offline."""
    db = MySQLManager()
    status, msg, info = db.check_connection_status()
    assert isinstance(status, bool), "Status must be boolean"
    assert "mysql.connector" == info["driver"], "Must use mysql.connector"
    assert "localhost" == info["host"], "Target host must be localhost"

    # Test queueing detection alert
    db.log_detection_alert(
        flow_id="test_unit_001",
        detection_engine="Track 1 Multiclass",
        predicted_class="DoS Hulk",
        confidence_score=99.5,
        anomaly_score=0.74,
        threat_severity="HIGH",
        dest_port=80,
    )
    alerts = db.get_recent_alerts(limit=5)
    assert len(alerts) >= 1, "Alert must be retrieved from buffer"
    assert any(a["flow_id"] == "test_unit_001" for a in alerts), "Logged alert must be present in buffer"


def test_styles_and_page_entry_points():
    """Verify cyber CSS definitions and entry points across all 9 UI pages."""
    from src.ui.styles import apply_cyber_css, CYBER_CSS, get_cyber_css
    assert callable(apply_cyber_css), "apply_cyber_css must be callable"
    assert callable(get_cyber_css), "get_cyber_css must be callable"
    assert isinstance(CYBER_CSS, str) and len(CYBER_CSS) > 500, "CYBER_CSS must be non-empty"

    from src.ui.pages import (
        p1_dashboard,
        p2_dataset_explorer,
        p3_multiclass_ids,
        p4_anomaly_detection,
        p5_model_performance,
        p6_threat_intel,
        p7_explainable_ai,
        p8_reports,
        p9_project_info,
    )
    pages = [
        p1_dashboard, p2_dataset_explorer, p3_multiclass_ids,
        p4_anomaly_detection, p5_model_performance, p6_threat_intel,
        p7_explainable_ai, p8_reports, p9_project_info,
    ]
    for idx, page_mod in enumerate(pages, 1):
        assert hasattr(page_mod, "render_page"), f"Page {idx} ({page_mod.__name__}) must define render_page"
        assert callable(getattr(page_mod, "render_page")), f"Page {idx} render_page must be callable"



def run_all_phase_c_tests():
    print("=" * 60)
    print("RUNNING RESINET-IDAM PHASE C SUITE")
    print("=" * 60)

    tests = [
        ("test_benchmark_loader", test_benchmark_loader),
        ("test_sample_traffic_loader", test_sample_traffic_loader),
        ("test_model_artifact_loaders", test_model_artifact_loaders),
        ("test_multiclass_inference_engine", test_multiclass_inference_engine),
        ("test_anomaly_inference_engine", test_anomaly_inference_engine),
        ("test_pdf_report_generation", test_pdf_report_generation),
        ("test_mysql_manager_buffering", test_mysql_manager_buffering),
        ("test_styles_and_page_entry_points", test_styles_and_page_entry_points),
    ]

    passed = 0
    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            raise e

    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} TESTS PASSED CLEANLY")
    print("=" * 60)


if __name__ == "__main__":
    run_all_phase_c_tests()
