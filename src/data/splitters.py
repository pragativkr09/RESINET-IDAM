"""
RESINET-IDAM: Data Splitters Module
Provides leakage-free, mathematically verified data partitioning strategies:
- ChronologicalSplitter: Strict temporal partitioning with class presence audit
- SessionGroupSplitter: GroupKFold cross-validation grouped by global_session_id
- OpenSetCampaignSplitter: Inter-campaign holdout for novel/unseen attack detection
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import GroupKFold

from src.config import CICIDS_CAMPAIGNS


class ChronologicalSplitter:
    """
    Implements strict chronological partitioning without temporal leakage.
    Guarantees the arrow of time: t_train < t_val < t_test.
    Never artificially shuffles future records into training.
    """

    @staticmethod
    def split_intra_day(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        group_by_col: str = "source_file",
        label_col: str = "Label",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits records chronologically within each capture group/day.
        
        Args:
            df: DataFrame sorted chronologically.
            train_ratio: Fraction of initial chronological records for training (e.g. 0.70).
            val_ratio: Fraction of subsequent records for validation (e.g. 0.15).
            test_ratio: Fraction of final records for testing (e.g. 0.15).
            group_by_col: Column indicating the daily capture group.
            label_col: Ground truth label column.
            
        Returns:
            (train_df, val_df, test_df, audit_report)
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        train_parts = []
        val_parts = []
        test_parts = []
        
        audit_report = {
            "per_group": {},
            "overall": {
                "train_classes": set(),
                "val_classes": set(),
                "test_classes": set(),
                "missing_from_train": set(),
                "missing_from_val": set(),
                "missing_from_test": set(),
            }
        }
        
        # Iterate through each capture group in chronological file order
        groups = df[group_by_col].unique() if group_by_col in df.columns else ["all"]
        
        for grp in groups:
            sub = df[df[group_by_col] == grp] if grp != "all" else df
            n = len(sub)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            # Pure slice preserving chronological order
            train_slice = sub.iloc[:n_train]
            val_slice = sub.iloc[n_train:n_train + n_val]
            test_slice = sub.iloc[n_train + n_val:]
            
            train_parts.append(train_slice)
            val_parts.append(val_slice)
            test_parts.append(test_slice)
            
            all_classes = set(sub[label_col].unique())
            t_classes = set(train_slice[label_col].unique())
            v_classes = set(val_slice[label_col].unique())
            te_classes = set(test_slice[label_col].unique())
            
            audit_report["per_group"][str(grp)] = {
                "total_rows": n,
                "train_rows": len(train_slice),
                "val_rows": len(val_slice),
                "test_rows": len(test_slice),
                "all_classes": sorted(list(all_classes)),
                "train_classes": sorted(list(t_classes)),
                "val_classes": sorted(list(v_classes)),
                "test_classes": sorted(list(te_classes)),
                "missing_from_train": sorted(list(all_classes - t_classes)),
                "missing_from_val": sorted(list(all_classes - v_classes)),
                "missing_from_test": sorted(list(all_classes - te_classes)),
            }
            
            audit_report["overall"]["train_classes"].update(t_classes)
            audit_report["overall"]["val_classes"].update(v_classes)
            audit_report["overall"]["test_classes"].update(te_classes)
            
        train_df = pd.concat(train_parts, axis=0)
        val_df = pd.concat(val_parts, axis=0)
        test_df = pd.concat(test_parts, axis=0)
        
        total_classes = set(df[label_col].unique())
        audit_report["overall"]["missing_from_train"] = sorted(list(total_classes - audit_report["overall"]["train_classes"]))
        audit_report["overall"]["missing_from_val"] = sorted(list(total_classes - audit_report["overall"]["val_classes"]))
        audit_report["overall"]["missing_from_test"] = sorted(list(total_classes - audit_report["overall"]["test_classes"]))
        
        # Convert sets to sorted lists for JSON serializability
        audit_report["overall"]["train_classes"] = sorted(list(audit_report["overall"]["train_classes"]))
        audit_report["overall"]["val_classes"] = sorted(list(audit_report["overall"]["val_classes"]))
        audit_report["overall"]["test_classes"] = sorted(list(audit_report["overall"]["test_classes"]))
        
        return train_df, val_df, test_df, audit_report


class SessionGroupSplitter:
    """
    Implements GroupKFold cross-validation grouped by global_session_id for OT-ICS.
    Formally guarantees zero session overlap between training and test partitions.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)

    def split(self, df: pd.DataFrame, session_col: str = "global_session_id", label_col: str = "malware"):
        """
        Yields (train_indices, test_indices) with asserted zero session leakage.
        """
        if session_col not in df.columns:
            raise KeyError(f"Session column '{session_col}' not found in dataframe")
            
        groups = df[session_col].values
        y = df[label_col].values if label_col in df.columns else np.zeros(len(df))
        
        for fold, (train_idx, test_idx) in enumerate(self.gkf.split(df, y, groups=groups)):
            train_sessions = set(groups[train_idx])
            test_sessions = set(groups[test_idx])
            overlap = train_sessions.intersection(test_sessions)
            
            # Assert zero leakage at runtime
            if len(overlap) > 0:
                raise RuntimeError(f"Session leakage detected in fold {fold}: {len(overlap)} overlapping sessions!")
                
            yield fold, train_idx, test_idx


class OpenSetCampaignSplitter:
    """
    Implements inter-campaign leave-one-attack-out holdout for CIC-IDS2017 open-set evaluation.
    Enables evaluation of novel, previously unseen attack categories.
    """

    @staticmethod
    def get_campaign_splits(
        df: pd.DataFrame,
        mode: str = "verified_benign_only",
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Partitions data into a training baseline and multiple withheld attack campaigns.
        
        Args:
            df: CIC-IDS2017 DataFrame containing 'source_file' and 'Label'.
            mode: 
                - 'verified_benign_only': Trains strictly on Monday 100% verified Benign traffic.
                - 'known_attacks_baseline': Trains on Monday (Benign) + Tuesday (Patator) + Wednesday (DoS).
                
        Returns:
            (train_baseline_df, withheld_campaigns_dict)
        """
        if mode == "verified_benign_only":
            train_mask = df["source_file"] == CICIDS_CAMPAIGNS["Monday"]
            train_df = df[train_mask].copy()
        elif mode == "known_attacks_baseline":
            known_files = [
                CICIDS_CAMPAIGNS["Monday"],
                CICIDS_CAMPAIGNS["Tuesday"],
                CICIDS_CAMPAIGNS["Wednesday"],
            ]
            train_mask = df["source_file"].isin(known_files)
            train_df = df[train_mask].copy()
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Define withheld evaluation campaigns
        withheld_campaigns = {
            "Thursday_Web_Attacks": df[df["source_file"] == CICIDS_CAMPAIGNS["Thursday_Morning"]].copy(),
            "Thursday_Infiltration": df[df["source_file"] == CICIDS_CAMPAIGNS["Thursday_Afternoon"]].copy(),
            "Friday_Botnet": df[df["source_file"] == CICIDS_CAMPAIGNS["Friday_Morning"]].copy(),
            "Friday_PortScan": df[df["source_file"] == CICIDS_CAMPAIGNS["Friday_Afternoon_PortScan"]].copy(),
            "Friday_DDoS": df[df["source_file"] == CICIDS_CAMPAIGNS["Friday_Afternoon_DDoS"]].copy(),
        }
        
        return train_df, withheld_campaigns

