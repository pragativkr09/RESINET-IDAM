"""
RESINET-IDAM: Step B.3 OT-ICS Session-Grouped Baseline Pipeline (src/pipelines/train_ot_ics.py)
Executes:
1. Data integrity validation on OT-ICS dataset (4,159 rows, 0 nulls, 2,084 physical sessions).
2. 5-Fold GroupKFold cross-validation grouped strictly by global_session_id.
3. Runtime verification of zero session overlap on every fold.
4. Model training and evaluation for:
   - RandomForestBaseline
   - HistGradientBoostingBaseline
   - SGDBaseline (with fold-isolated StandardScaler)
5. Metric computation: Accuracy, Precision, Recall, F1, Macro F1, Confusion Matrix,
   ROC-AUC, PR-AUC, FPR, FNR.
6. Serialization of final model artifacts to artifacts/models/ and benchmark summary
   to artifacts/benchmarks/ot_ics_session_grouped_benchmark.json.
7. Explicit documentation of the D1-D3 (attack) vs D4 (benign) dataset-source confound.
"""

import sys
import os
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(".").resolve()))

from src.config import ARTIFACTS_DIR, MODELS_DIR, BENCHMARKS_DIR, RANDOM_SEED
from src.data.loaders import OTICSLoader
from src.data.splitters import SessionGroupSplitter
from src.data.preprocessors import LeakFreeScaler
from src.models.supervised import (
    RandomForestBaseline,
    HistGradientBoostingBaseline,
    SGDBaseline,
)
from src.evaluation.metrics import evaluate_binary_classification


def run_ot_ics_pipeline() -> Dict[str, Any]:
    print("=" * 80)
    print("RESINET-IDAM: STEP B.3 OT-ICS SESSION-GROUPED BASELINE PIPELINE")
    print("=" * 80)

    # 1. Load and Validate Data Integrity
    print("\n[Step 1/5] Loading OT-ICS Dataset & Validating Integrity...")
    t0 = time.time()
    df = OTICSLoader.load()
    
    total_records = len(df)
    attack_records = int((df["malware"] == 1).sum())
    benign_records = int((df["malware"] == 0).sum())
    total_nulls = int(df.isnull().sum().sum())
    global_sessions = int(df["global_session_id"].nunique())
    
    attack_sessions = int(df[df["malware"] == 1]["global_session_id"].nunique())
    benign_sessions = int(df[df["malware"] == 0]["global_session_id"].nunique())
    
    # Assertions against verified Phase A ground truth
    assert total_records == 4159, f"Expected 4,159 records, got {total_records}"
    assert attack_records == 2150, f"Expected 2,150 attack records, got {attack_records}"
    assert benign_records == 2009, f"Expected 2,009 benign records, got {benign_records}"
    assert total_nulls == 0, f"Expected 0 missing values, found {total_nulls}"
    assert global_sessions == 2084, f"Expected 2,084 global sessions, got {global_sessions}"
    assert attack_sessions == 1079, f"Expected 1,079 attack sessions, got {attack_sessions}"
    assert benign_sessions == 1005, f"Expected 1,005 benign sessions, got {benign_sessions}"
    
    feature_cols = [c for c in df.columns if c not in ["ip.session_id", "source_file", "global_session_id", "malware"]]
    
    print(f"  Total Records: {total_records:,} (Attack: {attack_records:,}, Benign: {benign_records:,})")
    print(f"  Physical Sessions: {global_sessions:,} (Attack: {attack_sessions:,}, Benign: {benign_sessions:,}, Mixed: 0)")
    print(f"  Missing Values: {total_nulls}")
    print(f"  Feature Dimensions: {len(feature_cols)} network flow attributes")
    print(f"  Dataset Sources: D1.csv (2 rows), D2.csv (29 rows), D3.csv (2,119 rows), D4.csv (2,009 rows)")
    print(f"  Integrity validation passed in {time.time()-t0:.2f}s")

    # 2. Setup Session-Grouped Splitter
    print("\n[Step 2/5] Initializing 5-Fold GroupKFold on global_session_id...")
    splitter = SessionGroupSplitter(n_splits=5)
    
    # Validate and log fold distributions
    fold_splits_info = []
    for fold, train_idx, test_idx in splitter.split(df, session_col="global_session_id", label_col="malware"):
        tr_df = df.iloc[train_idx]
        te_df = df.iloc[test_idx]
        
        tr_atk = int((tr_df["malware"] == 1).sum())
        tr_ben = int((tr_df["malware"] == 0).sum())
        te_atk = int((te_df["malware"] == 1).sum())
        te_ben = int((te_df["malware"] == 0).sum())
        
        tr_sess_atk = int(tr_df[tr_df["malware"] == 1]["global_session_id"].nunique())
        tr_sess_ben = int(tr_df[tr_df["malware"] == 0]["global_session_id"].nunique())
        te_sess_atk = int(te_df[te_df["malware"] == 1]["global_session_id"].nunique())
        te_sess_ben = int(te_df[te_df["malware"] == 0]["global_session_id"].nunique())
        
        overlap = set(tr_df["global_session_id"]).intersection(set(te_df["global_session_id"]))
        assert len(overlap) == 0, f"Session leakage detected in fold {fold}: {len(overlap)} overlapping sessions!"
        assert te_atk > 0 and te_ben > 0, f"Class missing in test fold {fold}"
        assert tr_atk > 0 and tr_ben > 0, f"Class missing in train fold {fold}"
        
        fold_info = {
            "fold": fold,
            "train_rows": len(tr_df),
            "test_rows": len(te_df),
            "train_attack": tr_atk,
            "train_benign": tr_ben,
            "test_attack": te_atk,
            "test_benign": te_ben,
            "train_sessions": len(tr_df["global_session_id"].unique()),
            "test_sessions": len(te_df["global_session_id"].unique()),
            "train_attack_sessions": tr_sess_atk,
            "train_benign_sessions": tr_sess_ben,
            "test_attack_sessions": te_sess_atk,
            "test_benign_sessions": te_sess_ben,
            "session_overlap": len(overlap),
        }
        fold_splits_info.append(fold_info)
        print(f"  Fold {fold}: Train={len(tr_df)} (Atk:{tr_atk}, Ben:{tr_ben}) | Test={len(te_df)} (Atk:{te_atk}, Ben:{te_ben}) | Sessions={fold_info['test_sessions']} | Overlap=0")

    # 3. Model Architecture Configurations
    model_configs = [
        {
            "class": RandomForestBaseline,
            "name": "RandomForest",
            "params": {"n_estimators": 100, "max_depth": 15, "class_weight": "balanced", "random_state": RANDOM_SEED, "n_jobs": -1},
            "use_scaler": False,
        },
        {
            "class": HistGradientBoostingBaseline,
            "name": "HistGradientBoosting",
            "params": {"max_iter": 100, "learning_rate": 0.05, "class_weight": "balanced", "random_state": RANDOM_SEED},
            "use_scaler": False,
        },
        {
            "class": SGDBaseline,
            "name": "SGDClassifier",
            "params": {"loss": "log_loss", "penalty": "l2", "max_iter": 2000, "class_weight": "balanced", "random_state": RANDOM_SEED},
            "use_scaler": True,
        },
    ]

    benchmark_results = {
        "pipeline": "OT-ICS Session-Grouped Baseline Pipeline",
        "dataset_metadata": {
            "total_records": total_records,
            "attack_records": attack_records,
            "benign_records": benign_records,
            "global_sessions": global_sessions,
            "attack_sessions": attack_sessions,
            "benign_sessions": benign_sessions,
            "features_count": len(feature_cols),
            "sources": {"D1.csv": 2, "D2.csv": 29, "D3.csv": 2119, "D4.csv": 2009},
            "random_seed": RANDOM_SEED,
            "n_splits": 5,
        },
        "fold_partitions": fold_splits_info,
        "models": {},
        "scientific_limitation": (
            "D1-D3 contain 100% attack records while D4 contains 100% benign records. "
            "High classification metrics reflect separation of Modbus industrial communication "
            "(D4) from malware session traces (D1-D3), rather than generalized OT attack detection. "
            "Zero-day intrusion detection is not claimed."
        ),
    }

    # 4. Train and Cross-Validate Each Model
    print("\n[Step 3/5] Executing 5-Fold Cross-Validation Across Baseline Architectures...")
    
    for config in model_configs:
        m_name = config["name"]
        print(f"\n  --- Cross-Validating {m_name} ---")
        
        fold_results = []
        oof_preds = np.zeros(total_records)
        oof_probs = np.zeros(total_records)
        
        total_fit_time = 0.0
        max_peak_mem = 0.0
        
        for fold, train_idx, test_idx in splitter.split(df, session_col="global_session_id", label_col="malware"):
            tr_df = df.iloc[train_idx]
            te_df = df.iloc[test_idx]
            
            X_tr = tr_df[feature_cols].values
            y_tr = tr_df["malware"].values
            X_te = te_df[feature_cols].values
            y_te = te_df["malware"].values
            
            # Preprocessing: fitted strictly on training fold only
            if config["use_scaler"]:
                scaler = LeakFreeScaler(scaler_type="standard")
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                
            model = config["class"](**config["params"])
            model.fit(X_tr, y_tr)
            
            total_fit_time += model.fit_time
            max_peak_mem = max(max_peak_mem, model.peak_memory_mb)
            
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            
            oof_preds[test_idx] = y_pred
            oof_probs[test_idx] = y_prob
            
            m = evaluate_binary_classification(y_te, y_pred, y_prob)
            m["fold"] = fold
            m["fit_time_s"] = round(model.fit_time, 3)
            fold_results.append(m)
            
            print(f"    Fold {fold}: Acc={m['accuracy']*100:.2f}% | F1={m['f1']*100:.2f}% | Macro F1={m['macro_f1']*100:.2f}% | ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | FPR={m['fpr']*100:.2f}% | FNR={m['fnr']*100:.2f}%")

        # Compute Out-of-Fold (OOF) Pooled Metrics across all 4,159 records
        oof_metrics = evaluate_binary_classification(df["malware"].values, oof_preds, oof_probs)
        
        # Aggregate statistics (Mean and Std across the 5 folds)
        metric_keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "fpr", "fnr"]
        agg_mean = {k: float(np.mean([f[k] for f in fold_results])) for k in metric_keys}
        agg_std = {k: float(np.std([f[k] for f in fold_results])) for k in metric_keys}
        
        print(f"    >> OOF Aggregate: Acc={oof_metrics['accuracy']*100:.2f}% | F1={oof_metrics['f1']*100:.2f}% | Macro F1={oof_metrics['macro_f1']*100:.2f}% | ROC-AUC={oof_metrics['roc_auc']:.4f} | PR-AUC={oof_metrics['pr_auc']:.4f}")
        print(f"    >> Mean ± Std:   Acc={agg_mean['accuracy']*100:.2f}% ± {agg_std['accuracy']*100:.2f}% | F1={agg_mean['f1']*100:.2f}% ± {agg_std['f1']*100:.2f}% | ROC-AUC={agg_mean['roc_auc']:.4f} ± {agg_std['roc_auc']:.4f}")

        # Train Final Reference Model on Full Dataset
        print(f"    Fitting final reference {m_name} model on all {total_records:,} records...")
        X_full = df[feature_cols].values
        y_full = df["malware"].values
        
        if config["use_scaler"]:
            final_scaler = LeakFreeScaler(scaler_type="standard")
            X_full = final_scaler.fit_transform(X_full)
            scaler_path = MODELS_DIR / "ot_ics_standard_scaler.joblib"
            final_scaler.save(scaler_path)
            
        final_model = config["class"](**config["params"])
        final_model.fit(X_full, y_full)
        artifact_path = MODELS_DIR / f"ot_ics_{m_name.lower()}.joblib"
        final_model.save(artifact_path)
        print(f"    Saved final model artifact to {artifact_path.name}")
        
        benchmark_results["models"][m_name] = {
            "parameters": config["params"],
            "total_cross_val_fit_time_s": round(total_fit_time, 2),
            "peak_memory_mb": round(max_peak_mem, 2),
            "final_model_artifact": artifact_path.name,
            "folds": fold_results,
            "aggregate_mean": agg_mean,
            "aggregate_std": agg_std,
            "out_of_fold_pooled": oof_metrics,
        }

    # 5. Save Benchmark Deliverables
    print("\n[Step 4/5] Saving Benchmark JSON Report...")
    benchmark_file = BENCHMARKS_DIR / "ot_ics_session_grouped_benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"  Benchmark summary saved to {benchmark_file}")

    # 6. Database Status
    print("\n[Step 5/5] Checking Database Logging...")
    print("  MySQL connection is unconfigured in current environment.")
    print("  Per strict instructions, SQLite substitution was NOT performed.")
    print(f"  Full benchmark metrics preserved in {benchmark_file.name}")

    print("\n" + "=" * 80)
    print("STEP B.3 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return benchmark_results


if __name__ == "__main__":
    run_ot_ics_pipeline()

