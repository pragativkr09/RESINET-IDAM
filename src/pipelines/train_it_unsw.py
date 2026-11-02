"""
RESINET-IDAM: Step B.2 UNSW-NB15 Dual-Evaluation Pipeline (src/pipelines/train_it_unsw.py)
Executes:
1. Training baseline models strictly on D5 (175,341 records).
2. Dual evaluation on:
   - Track 1: Original D8 (82,332 records)
   - Track 2: Decontaminated D8 (73,791 records)
3. Calculation of metric leakage delta (Delta = Original - Decontaminated).
4. Serialization of models to artifacts/models/ and benchmark results to artifacts/benchmarks/.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(".").resolve()))

from src.config import ARTIFACTS_DIR, MODELS_DIR, BENCHMARKS_DIR, RANDOM_SEED
from src.data.loaders import UNSWLoader
from src.data.preprocessors import LeakFreeScaler
from src.models.supervised import (
    RandomForestBaseline,
    HistGradientBoostingBaseline,
    SGDBaseline,
)
from src.evaluation.metrics import (
    evaluate_binary_classification,
    compute_metric_difference,
)


def run_unsw_dual_evaluation_pipeline() -> Dict[str, Any]:
    print("=" * 80)
    print("RESINET-IDAM: STEP B.2 UNSW-NB15 DUAL-EVALUATION PIPELINE")
    print("=" * 80)
    
    # 1. Load Datasets
    print("\n[Step 1/5] Loading UNSW-NB15 Parquet Datasets...")
    t0 = time.time()
    df_d5, df_d8_orig, df_d8_decontam = UNSWLoader.load()
    feature_cols = UNSWLoader.get_feature_columns(df_d5)
    
    print(f"  Training Set (D5): {len(df_d5):,} rows | Attack: {(df_d5['label']==1).sum():,} ({(df_d5['label']==1).mean()*100:.2f}%), Normal: {(df_d5['label']==0).sum():,}")
    print(f"  Original Test Set (D8): {len(df_d8_orig):,} rows | Attack: {(df_d8_orig['label']==1).sum():,} ({(df_d8_orig['label']==1).mean()*100:.2f}%), Normal: {(df_d8_orig['label']==0).sum():,}")
    print(f"  Decontaminated Test Set (D8): {len(df_d8_decontam):,} rows | Attack: {(df_d8_decontam['label']==1).sum():,} ({(df_d8_decontam['label']==1).mean()*100:.2f}%), Normal: {(df_d8_decontam['label']==0).sum():,}")
    print(f"  Purged Overlapping Records: {len(df_d8_orig) - len(df_d8_decontam):,} rows")
    print(f"  Feature Dimensions: {len(feature_cols)} network flow attributes")
    print(f"  Dataset loading completed in {time.time()-t0:.2f}s")
    
    # 2. Check Multiclass Ground-Truth Availability
    print("\n[Step 2/5] Checking Multiclass Ground-Truth Labels...")
    has_attack_cat = "attack_cat" in df_d5.columns
    if not has_attack_cat:
        print("  [INFO] Column 'attack_cat' is NOT present in the provided D5/D8 datasets.")
        print("  [STRICT POLICY] Reliable multiclass ground truth is unavailable. Multiclass classification is disqualified.")
        print("  [FOCUS] Proceeding with rigorous Binary Classification (0 = Normal, 1 = Attack).")
    
    # 3. Extract Arrays & Fit Preprocessing Strictly on D5
    print("\n[Step 3/5] Fitting Preprocessing Transformers (Strictly on D5)...")
    X_train = df_d5[feature_cols].values
    y_train = df_d5["label"].values
    
    X_test_orig = df_d8_orig[feature_cols].values
    y_test_orig = df_d8_orig["label"].values
    
    X_test_decontam = df_d8_decontam[feature_cols].values
    y_test_decontam = df_d8_decontam["label"].values
    
    scaler = LeakFreeScaler(scaler_type="robust")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_orig_scaled = scaler.transform(X_test_orig)
    X_test_decontam_scaled = scaler.transform(X_test_decontam)
    
    scaler_path = MODELS_DIR / "unsw_nb15_robust_scaler.joblib"
    scaler.save(scaler_path)
    print(f"  RobustScaler fitted strictly on D5 and saved to {scaler_path.name}")
    
    # 4. Instantiate Models
    models = [
        RandomForestBaseline(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED),
        HistGradientBoostingBaseline(max_iter=150, learning_rate=0.1, class_weight="balanced", random_state=RANDOM_SEED),
        SGDBaseline(loss="log_loss", penalty="elasticnet", l1_ratio=0.15, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    ]
    
    benchmark_results = {
        "pipeline": "UNSW-NB15 Dual-Evaluation Benchmark",
        "dataset_info": {
            "d5_train_rows": len(df_d5),
            "d8_orig_rows": len(df_d8_orig),
            "d8_decontam_rows": len(df_d8_decontam),
            "purged_overlap_rows": len(df_d8_orig) - len(df_d8_decontam),
            "features_count": len(feature_cols),
            "random_seed": RANDOM_SEED,
        },
        "models": {},
    }
    
    # 5. Train & Evaluate Each Model
    print("\n[Step 4/5] Training & Evaluating Baseline Models...")
    
    for model in models:
        m_name = model.name
        print(f"\n  --- Training {m_name} ---")
        
        # Select appropriate feature representation (tree models use raw, linear models use scaled)
        if m_name == "SGDClassifier":
            X_tr = X_train_scaled
            X_te_orig = X_test_orig_scaled
            X_te_decontam = X_test_decontam_scaled
        else:
            X_tr = X_train
            X_te_orig = X_test_orig
            X_te_decontam = X_test_decontam
            
        # Fit on D5
        model.fit(X_tr, y_train)
        print(f"    Fit time: {model.fit_time:.2f}s | Peak Memory: {model.peak_memory_mb:.2f} MB")
        
        # Track 1: Original D8 Evaluation
        t_eval_orig = time.time()
        y_pred_orig = model.predict(X_te_orig)
        y_prob_orig = model.predict_proba(X_te_orig)[:, 1]
        eval_orig_time = time.time() - t_eval_orig
        metrics_orig = evaluate_binary_classification(y_test_orig, y_pred_orig, y_prob_orig)
        metrics_orig["inference_time_s"] = round(eval_orig_time, 4)
        
        # Track 2: Decontaminated D8 Evaluation
        t_eval_decontam = time.time()
        y_pred_decontam = model.predict(X_te_decontam)
        y_prob_decontam = model.predict_proba(X_te_decontam)[:, 1]
        eval_decontam_time = time.time() - t_eval_decontam
        metrics_decontam = evaluate_binary_classification(y_test_decontam, y_pred_decontam, y_prob_decontam)
        metrics_decontam["inference_time_s"] = round(eval_decontam_time, 4)
        
        # Calculate Leakage Delta
        delta = compute_metric_difference(metrics_orig, metrics_decontam)
        
        # Save Model Artifact
        artifact_path = MODELS_DIR / f"unsw_nb15_{m_name.lower()}.joblib"
        model.save(artifact_path)
        print(f"    Saved model artifact to {artifact_path.name}")
        
        # Print comparative summary
        print(f"    [Track 1 - Original D8]     Acc: {metrics_orig['accuracy']*100:.2f}% | F1: {metrics_orig['f1']*100:.2f}% | Macro F1: {metrics_orig['macro_f1']*100:.2f}% | PR-AUC: {metrics_orig['pr_auc']:.4f} | ROC-AUC: {metrics_orig['roc_auc']:.4f}")
        print(f"    [Track 2 - Decontam D8]     Acc: {metrics_decontam['accuracy']*100:.2f}% | F1: {metrics_decontam['f1']*100:.2f}% | Macro F1: {metrics_decontam['macro_f1']*100:.2f}% | PR-AUC: {metrics_decontam['pr_auc']:.4f} | ROC-AUC: {metrics_decontam['roc_auc']:.4f}")
        print(f"    [Leakage Delta (Orig-Decont)] Delta_Acc: {delta['delta_accuracy']*100:+.2f}% | Delta_F1: {delta['delta_f1']*100:+.2f}% | Delta_Macro_F1: {delta['delta_macro_f1']*100:+.2f}% | Delta_PR-AUC: {delta['delta_pr_auc']:+.4f} | Delta_ROC-AUC: {delta['delta_roc_auc']:+.4f}")
        
        benchmark_results["models"][m_name] = {
            "parameters": model.params,
            "fit_time_seconds": round(model.fit_time, 2),
            "peak_memory_mb": round(model.peak_memory_mb, 2),
            "artifact_file": artifact_path.name,
            "track_1_original_d8": metrics_orig,
            "track_2_decontaminated_d8": metrics_decontam,
            "leakage_delta": delta,
        }
        
    # 6. Save Benchmark JSON
    print("\n[Step 5/5] Saving Benchmark Deliverables...")
    benchmark_file = BENCHMARKS_DIR / "unsw_nb15_dual_evaluation_benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"  Benchmark summary saved to {benchmark_file}")
    
    # 7. Check MySQL Logging Status
    print("\n[Database Status]")
    print("  MySQL connection is NOT configured in the current environment.")
    print("  Per strict instructions, SQLite substitution was NOT performed.")
    print(f"  All experiment metadata is preserved in {benchmark_file.name}")
    
    print("\n" + "=" * 80)
    print("STEP B.2 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return benchmark_results


if __name__ == "__main__":
    run_unsw_dual_evaluation_pipeline()
