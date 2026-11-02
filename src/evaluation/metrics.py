"""
RESINET-IDAM: Evaluation Metrics Module (src/evaluation/metrics.py)
Provides standard evaluation metrics for cybersecurity classification:
- Accuracy, Precision, Recall, F1-score, Macro F1-score
- Confusion Matrix (TN, FP, FN, TP)
- ROC-AUC and PR-AUC
- Operational rates: False Positive Rate (FPR) and False Negative Rate (FNR)
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes complete suite of binary evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Attack).
        y_pred: Predicted binary labels.
        y_prob: Predicted probability for Attack class (label 1).
        
    Returns:
        Dictionary of computed metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Operational rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions))
    
    # ROC-AUC
    roc_auc = float(roc_auc_score(y_true, y_prob))
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
    }


def compute_metric_difference(
    metrics_orig: Dict[str, Any],
    metrics_decontam: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes Delta = Metric_Original - Metric_Decontaminated
    to quantify the degree of artificial inflation caused by data leakage.
    """
    keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "fpr", "fnr"]
    diff = {}
    for k in keys:
        if k in metrics_orig and k in metrics_decontam:
            diff[f"delta_{k}"] = round(metrics_orig[k] - metrics_decontam[k], 6)
    return diff

