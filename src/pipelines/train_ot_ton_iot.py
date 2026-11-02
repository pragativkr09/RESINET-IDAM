"""
RESINET-IDAM: Phase B - Step B.4: ToN_IoT Per-Device Anomaly Detection Pipeline
Executes unsupervised baseline profiling and anomaly detection across all 7 ToN_IoT telemetry streams:
1. Fridge
2. Garage Door
3. GPS Tracker
4. Modbus
5. Motion Light
6. Thermostat
7. Weather

MANDATORY SCIENTIFIC CONSTRAINTS:
- No supervised attack detection metrics (NO Accuracy, Precision, Recall, F1, ROC-AUC, or PR-AUC).
- Unsupervised anomaly scoring and baseline profiling only.
- Baseline profiling partition: first 70% chronological/sequential records.
- Holdout evaluation partition: remaining 30% records.
- Preprocessing (LeakFreeScaler) fitted strictly on the baseline training partition.
- Threshold calibration (tau_90, tau_95, tau_99) strictly from training baseline anomaly scores.
- Unsupervised diagnostics: Quantiles, Empirical Alarm Rates, Score Separation Ratios (SSR), Temporal Drift.
- Explicit documentation that score separation across scenario metadata is NOT proof of attack detection.
- Artifacts saved to artifacts/models/ and artifacts/benchmarks/.
"""

import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd

from src.config import TON_IOT_DEVICES, ARTIFACTS_DIR, BENCHMARKS_DIR, MODELS_DIR
from src.data.loaders import ToNIoTLoader
from src.data.preprocessors import LeakFreeScaler
from src.models.anomaly import IsolationForestBaseline, OneClassSVMBaseline


DEVICE_ENCODERS = {
    "fridge": {"temp_condition": {"low": 0.0, "high": 1.0}},
    "garage_door": {
        "door_state": {"closed": 0.0, "open": 1.0},
        "sphone_signal": {"false": 0.0, "true": 1.0},
    },
    "gps_tracker": {},
    "modbus": {},
    "motion_light": {
        "motion_status": {"false": 0.0, "true": 1.0},
        "light_status": {"off": 0.0, "on": 1.0},
    },
    "thermostat": {"ac_state": {"false": 0.0, "true": 1.0}},
    "weather": {},
}

META_COLS = ["date", "time", "inferred_scenario_meta", "source_file", "device_type"]


def run_ton_iot_anomaly_pipeline() -> Dict[str, Any]:
    """Runs the full unsupervised anomaly profiling pipeline across all 7 ToN_IoT devices."""
    print("=" * 80)
    print("RESINET-IDAM: Phase B - Step B.4: ToN_IoT Unsupervised Telemetry Profiling")
    print("=" * 80)

    overall_t0 = time.time()
    tracemalloc.start()

    benchmark_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "step": "B.4",
        "pipeline": "ToN_IoT Per-Device Anomaly Detection & Baseline Profiling",
        "paradigms": {
            "nature": "Unsupervised telemetry profiling and boundary estimation",
            "supervised_metrics_prohibited": True,
            "disclaimer": (
                "Score separation across scenario metadata reflects recording environment conditions "
                "and simulator artifacts, NOT row-level verified attack detection."
            ),
        },
        "devices": {},
    }

    total_active_rows = 0

    for dev in TON_IOT_DEVICES:
        print(f"\n[{dev.upper()}] Loading and preparing telemetry...")
        dev_t0 = time.time()

        # 1. Load dataset with simulator idling pruning
        raw_df = ToNIoTLoader.load_device(dev, prune_empty=False)
        raw_count = len(raw_df)

        df = ToNIoTLoader.load_device(dev, prune_empty=True)
        pruned_count = len(df)
        empty_pruned = raw_count - pruned_count
        total_active_rows += pruned_count

        print(f"  Total records: {pruned_count:,} (pruned {empty_pruned:,} simulator idling rows)")

        # 2. Encode categorical columns deterministically
        df_encoded = df.copy()
        encoder_map = DEVICE_ENCODERS.get(dev, {})
        for col, mapping in encoder_map.items():
            df_encoded[col] = (
                df_encoded[col].astype(str).str.strip().str.lower().map(mapping).fillna(0.0)
            )

        # 3. Identify telemetry feature columns
        feature_cols = [c for c in df.columns if c not in META_COLS]
        print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

        # Data quality checks
        null_counts = {c: int(df[c].isnull().sum()) for c in feature_cols}
        has_nulls = any(v > 0 for v in null_counts.values())
        if hass_nulls := has_nulls:
            raise ValueError(f"Unexpected null values in {dev} telemetry features: {null_counts}")

        # 4. Chronological / sequential split (70% baseline training, 30% holdout test)
        n_total = len(df_encoded)
        n_train = int(n_total * 0.7)
        n_test = n_total - n_train

        train_indices = np.arange(0, n_train)
        test_indices = np.arange(n_train, n_total)

        df_train = df_encoded.iloc[train_indices].copy()
        df_test = df_encoded.iloc[test_indices].copy()

        X_train_raw = df_train[feature_cols].astype(float).values
        X_test_raw = df_test[feature_cols].astype(float).values

        print(f"  Baseline Training partition: {n_train:,} records ({n_train/n_total*100:.1f}%)")
        print(f"  Holdout Test partition:       {n_test:,} records ({n_test/n_total*100:.1f}%)")

        # 5. Leak-free Feature Scaling (StandardScaler fitted strictly on baseline training partition)
        scaler = LeakFreeScaler(scaler_type="standard")
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # Save scaler artifact
        scaler_path = MODELS_DIR / f"ton_iot_{dev}_scaler.joblib"
        scaler.save(scaler_path)

        # Feature audit stats
        feature_audit = {}
        for idx, col in enumerate(feature_cols):
            feature_audit[col] = {
                "dtype": str(df[col].dtype),
                "train_mean": float(np.mean(X_train_raw[:, idx])),
                "train_std": float(np.std(X_train_raw[:, idx])),
                "train_min": float(np.min(X_train_raw[:, idx])),
                "train_max": float(np.max(X_train_raw[:, idx])),
                "test_mean": float(np.mean(X_test_raw[:, idx])),
                "test_std": float(np.std(X_test_raw[:, idx])),
                "zero_variance": bool(np.std(X_train_raw[:, idx]) == 0),
            }

        # 6. Train Models and Compute Unsupervised Diagnostics
        models = {
            "IsolationForest": IsolationForestBaseline(
                n_estimators=100, contamination="auto", random_state=42, n_jobs=-1
            ),
            "OneClassSVM": OneClassSVMBaseline(
                kernel="rbf", gamma="scale", nu=0.03, max_train_samples=10000, random_state=42
            ),
        }

        device_results: Dict[str, Any] = {
            "record_counts": {
                "raw_parquet": raw_count,
                "empty_pruned": empty_pruned,
                "active_telemetry": pruned_count,
                "baseline_train": n_train,
                "holdout_test": n_test,
            },
            "features": feature_cols,
            "feature_audit": feature_audit,
            "scenario_distribution": {
                "train": df_train["inferred_scenario_meta"].value_counts().to_dict(),
                "test": df_test["inferred_scenario_meta"].value_counts().to_dict(),
                "total": df["inferred_scenario_meta"].value_counts().to_dict(),
            },
            "models": {},
        }

        for model_name, model in models.items():
            print(f"  Fitting {model_name}...")
            model.fit(X_train)

            # Score training and test partitions
            t_score0 = time.time()
            scores_train = model.score_samples(X_train)
            scores_test = model.score_samples(X_test)
            score_time = time.time() - t_score0
            model.score_time = score_time

            # Calibrate baseline thresholds
            thresholds = model.calibrate_thresholds(scores_train)

            # Save model artifact
            model_path = MODELS_DIR / f"ton_iot_{dev}_{model_name.lower()}.joblib"
            model.save(model_path)

            # Unsupervised Diagnostics:
            # A. Quantiles
            quantiles_train = {
                "min": float(np.min(scores_train)),
                "p25": float(np.percentile(scores_train, 25)),
                "p50_median": float(np.percentile(scores_train, 50)),
                "p75": float(np.percentile(scores_train, 75)),
                "p90": float(np.percentile(scores_train, 90)),
                "p95": float(np.percentile(scores_train, 95)),
                "p99": float(np.percentile(scores_train, 99)),
                "max": float(np.max(scores_train)),
                "mean": float(np.mean(scores_train)),
                "std": float(np.std(scores_train)),
            }
            quantiles_test = {
                "min": float(np.min(scores_test)),
                "p25": float(np.percentile(scores_test, 25)),
                "p50_median": float(np.percentile(scores_test, 50)),
                "p75": float(np.percentile(scores_test, 75)),
                "p90": float(np.percentile(scores_test, 90)),
                "p95": float(np.percentile(scores_test, 95)),
                "p99": float(np.percentile(scores_test, 99)),
                "max": float(np.max(scores_test)),
                "mean": float(np.mean(scores_test)),
                "std": float(np.std(scores_test)),
            }

            # B. Empirical Alarm Rates on holdout test partition
            tau_90 = thresholds["tau_90"]
            tau_95 = thresholds["tau_95"]
            tau_99 = thresholds["tau_99"]

            alarm_rate_90 = float(np.mean(scores_test >= tau_90) * 100.0)
            alarm_rate_95 = float(np.mean(scores_test >= tau_95) * 100.0)
            alarm_rate_99 = float(np.mean(scores_test >= tau_99) * 100.0)

            # C. Score Separation Ratio (SSR)
            ssr = float(np.mean(scores_test) / np.mean(scores_train)) if np.mean(scores_train) != 0 else 1.0

            # D. Scenario-level breakdown on test partition
            scenario_breakdown = {}
            test_scenarios = df_test["inferred_scenario_meta"].values
            for scen in np.unique(test_scenarios):
                scen_mask = test_scenarios == scen
                scen_scores = scores_test[scen_mask]
                scen_mean = float(np.mean(scen_scores))
                scen_ssr = float(scen_mean / np.mean(scores_train)) if np.mean(scores_train) != 0 else 1.0
                scen_alarm_95 = float(np.mean(scen_scores >= tau_95) * 100.0)
                scenario_breakdown[scen] = {
                    "count": int(np.sum(scen_mask)),
                    "mean_score": scen_mean,
                    "ssr_vs_baseline": scen_ssr,
                    "alarm_rate_tau_95_pct": scen_alarm_95,
                }

            # E. Temporal variation across 10 chronological deciles
            temporal_deciles = []
            chunk_size = n_total // 10
            # Combine scores across entire timeline
            scores_all = np.empty(n_total, dtype=float)
            scores_all[train_indices] = scores_train
            scores_all[test_indices] = scores_test

            for decile_idx in range(10):
                c_start = decile_idx * chunk_size
                c_end = (decile_idx + 1) * chunk_size if decile_idx < 9 else n_total
                chunk_scores = scores_all[c_start:c_end]
                temporal_deciles.append({
                    "decile": decile_idx + 1,
                    "partition": "baseline_train" if c_end <= n_train else ("mixed" if c_start < n_train else "holdout_test"),
                    "row_range": [int(c_start), int(c_end)],
                    "mean_score": float(np.mean(chunk_scores)),
                    "std_score": float(np.std(chunk_scores)),
                    "alarm_rate_tau_95_pct": float(np.mean(chunk_scores >= tau_95) * 100.0),
                })

            # Record model results
            device_results["models"][model_name] = {
                "fit_time_seconds": float(model.fit_time),
                "score_time_seconds": float(score_time),
                "peak_memory_mb": float(model.peak_memory_mb),
                "calibrated_thresholds": thresholds,
                "quantiles": {
                    "baseline_train": quantiles_train,
                    "holdout_test": quantiles_test,
                },
                "empirical_alarm_rates_holdout": {
                    "alarm_rate_tau_90_pct": alarm_rate_90,
                    "alarm_rate_tau_95_pct": alarm_rate_95,
                    "alarm_rate_tau_99_pct": alarm_rate_99,
                    "nominal_train_tau_90_pct": 10.0,
                    "nominal_train_tau_95_pct": 5.0,
                    "nominal_train_tau_99_pct": 1.0,
                },
                "score_separation_ratio_holdout_vs_train": ssr,
                "scenario_breakdown": scenario_breakdown,
                "temporal_deciles": temporal_deciles,
                "model_parameters": model.params,
            }

            print(
                f"    {model_name}: fit={model.fit_time:.2f}s, score={score_time:.2f}s | "
                f"Train 95th={tau_95:.4f} | Test Alarm@95={alarm_rate_95:.2f}% | SSR={ssr:.4f}"
            )

        benchmark_report["devices"][dev] = device_results
        print(f"  Finished {dev} in {time.time()-dev_t0:.2f}s")

    _, overall_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_pipeline_time = time.time() - overall_t0

    benchmark_report["summary"] = {
        "devices_profiled": len(TON_IOT_DEVICES),
        "total_active_telemetry_records": total_active_rows,
        "total_pipeline_time_seconds": float(total_pipeline_time),
        "peak_pipeline_memory_mb": float(overall_peak / (1024 * 1024)),
    }

    # Save benchmark report to disk
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_file = BENCHMARKS_DIR / "ton_iot_unsupervised_benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Step B.4 Complete in {total_pipeline_time:.2f}s (Peak Memory: {overall_peak/(1024*1024):.2f} MB)")
    print(f"Saved benchmark report: {benchmark_file}")
    print("=" * 80)

    return benchmark_report


if __name__ == "__main__":
    run_ton_iot_anomaly_pipeline()
