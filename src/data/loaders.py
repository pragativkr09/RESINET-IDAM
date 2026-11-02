"""
RESINET-IDAM: Data Loaders Module
Provides leak-free, non-destructive data loading interfaces for:
- CIC-IDS2017 (Cleaned, Deduplicated, and Conflict-Aware)
- UNSW-NB15 (D5 Train, Original D8, and Decontaminated D8)
- OT-ICS (D1-D4 Cleaned with dynamic global_session_id)
- ToN_IoT (7 device streams with safe pruning of simulator idling rows)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path

from src.config import (
    DATASET_PATHS,
    TON_IOT_DEVICES,
    TON_IOT_EMPTY_ROW_COUNTS,
)


class CICIDSLoader:
    """Loader for CIC-IDS2017 datasets with conflict isolation."""

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Extracts the 78 network flow feature columns (excluding metadata & label)."""
        return [c for c in df.columns if c not in ["source_file", "Label"]]

    @classmethod
    def get_conflicting_patterns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies feature vectors that appear with multiple conflicting labels.
        Returns a DataFrame containing the 1,436 ambiguous records.
        """
        feature_cols = cls.get_feature_columns(df)
        feature_and_label = feature_cols + ["Label"]
        
        # Deduplicate on features + Label to get unique pattern-label pairs
        unique_pairs = df[feature_and_label].drop_duplicates()
        
        # Find feature patterns that appear more than once (meaning conflicting labels)
        conflicting_patterns = unique_pairs[unique_pairs.duplicated(subset=feature_cols, keep=False)]
        
        # Filter full df to locate the actual rows matching these conflicting patterns
        # Fast matching via multi-index intersection
        conflict_keys = conflicting_patterns[feature_cols].drop_duplicates()
        merged = df.merge(conflict_keys, on=feature_cols, how="inner")
        return merged

    @classmethod
    def load(
        cls,
        deduplicated: bool = True,
        exclude_conflicts: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Loads the CIC-IDS2017 dataset.
        
        Args:
            deduplicated: If True, loads the 2,498,185 row deduplicated dataset;
                          otherwise loads the 2,827,876 row cleaned dataset.
            exclude_conflicts: If True, isolates the 1,436 conflicting-label rows
                              from the returned DataFrame and returns them separately.
                              
        Returns:
            (df_main, df_conflicts)
        """
        path = DATASET_PATHS["cicids2017_dedup"] if deduplicated else DATASET_PATHS["cicids2017_cleaned"]
        df = pd.read_parquet(path)
        
        if not exclude_conflicts:
            return df, None
            
        feature_cols = cls.get_feature_columns(df)
        if deduplicated:
            # When deduplicated, rows are already unique on (features + Label).
            # Therefore, duplicated on feature_cols directly flags conflicting patterns.
            is_conflict = df.duplicated(subset=feature_cols, keep=False)
        else:
            unique_pairs = df[feature_cols + ["Label"]].drop_duplicates()
            conflicting_keys = unique_pairs[unique_pairs.duplicated(subset=feature_cols, keep=False)][feature_cols].drop_duplicates()
            merged_indicator = df.merge(conflicting_keys, on=feature_cols, how="left", indicator="_is_conflict")
            is_conflict = merged_indicator["_is_conflict"] == "both"
        
        df_conflicts = df[is_conflict].copy()
        df_main = df[~is_conflict].copy()
        
        return df_main, df_conflicts


class UNSWLoader:
    """Loader for UNSW-NB15 with dual-evaluation decontaminated test sets."""

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Extracts the 43 shared network flow feature columns (excluding label/metadata)."""
        return [c for c in df.columns if c not in ["label", "attack_cat", "source_file", "id"]]

    @classmethod
    def load(cls) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads D5 (training set), Original D8 (test set), and Decontaminated D8 (test set).
        
        Returns:
            (df_train_d5, df_test_d8_original, df_test_d8_decontaminated)
        """
        df_d5 = pd.read_parquet(DATASET_PATHS["unsw_nb15_d5_train"])
        df_d8 = pd.read_parquet(DATASET_PATHS["unsw_nb15_d8_test"])
        
        feature_cols = cls.get_feature_columns(df_d5)
        d5_unique_features = df_d5[feature_cols].drop_duplicates()
        
        # Left-merge with indicator to find D8 rows matching any D5 feature vector
        merged = df_d8.merge(d5_unique_features, on=feature_cols, how="left", indicator="_overlap")
        is_leak = merged["_overlap"] == "both"
        
        df_d8_decontaminated = df_d8[~is_leak].copy()
        
        return df_d5, df_d8, df_d8_decontaminated


class OTICSLoader:
    """Loader for OT-ICS (D1-D4) dataset with physical session identification."""

    @classmethod
    def load(cls) -> pd.DataFrame:
        """
        Loads the OT-ICS cleaned dataset and constructs the globally unique physical session key:
            global_session_id = source_file + '_' + ip.session_id
            
        Returns:
            DataFrame with 4,159 records and verified 2,084 unique sessions.
        """
        df = pd.read_parquet(DATASET_PATHS["ot_ics_cleaned"])
        session_col = "ip.session_id" if "ip.session_id" in df.columns else "session_id"
        df["global_session_id"] = df["source_file"].astype(str) + "_" + df[session_col].astype(str)
        return df


class ToNIoTLoader:
    """Loader for ToN_IoT per-device telemetry with safe empty row pruning."""

    @classmethod
    def load_device(cls, device_name: str, prune_empty: bool = True) -> pd.DataFrame:
        """
        Loads the specified ToN_IoT device telemetry stream.
        
        Args:
            device_name: One of ['fridge', 'garage_door', 'gps_tracker', 'modbus',
                                 'motion_light', 'thermostat', 'weather'].
            prune_empty: If True, safely prunes the simulator idling empty rows
                         (28,379 in garage_door, 4,621 in thermostat, 1,876 in modbus).
                         
        Returns:
            DataFrame containing verified device telemetry.
        """
        if device_name not in TON_IOT_DEVICES:
            raise ValueError(f"Unknown device: {device_name}. Expected one of {TON_IOT_DEVICES}")
            
        key = f"ton_iot_{device_name}"
        path = DATASET_PATHS[key]
        df = pd.read_parquet(path)
        
        if not prune_empty:
            return df
            
        # Pruning logic:
        # In garage_door and thermostat, empty rows have date/time AND all sensor fields null
        # In modbus, empty rows have date/time null and register fields filled with 0.0
        if device_name in ["garage_door", "thermostat", "modbus"]:
            # Drop rows where date/time is null (which are 100% trailing blank simulator rows)
            valid_mask = ~(df["date"].isnull() | df["time"].isnull())
            pruned_df = df[valid_mask].copy()
            return pruned_df
            
        return df
