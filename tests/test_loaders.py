"""
Unit Tests: Data Loaders Module (tests/test_loaders.py)
Validates dataset loading integrity, empty row pruning, session key generation,
and conflicting-pattern isolation.
"""

import pandas as pd
from src.data.loaders import (
    CICIDSLoader,
    UNSWLoader,
    OTICSLoader,
    ToNIoTLoader,
)
from src.config import TON_IOT_DEVICES


def test_otics_loader():
    """Verify OT-ICS loader creates global_session_id and preserves exact label counts."""
    df = OTICSLoader.load()
    assert len(df) == 4159, f"Expected 4,159 rows, got {len(df)}"
    assert (df["malware"] == 1).sum() == 2150, "Expected exactly 2,150 attack rows"
    assert (df["malware"] == 0).sum() == 2009, "Expected exactly 2,009 benign rows"
    assert "global_session_id" in df.columns, "global_session_id column missing"
    assert df["global_session_id"].nunique() == 2084, "Expected exactly 2,084 unique sessions"


def test_unsw_loader():
    """Verify UNSW-NB15 loader correctly produces D5, Original D8, and Decontaminated D8."""
    df_d5, df_d8_orig, df_d8_decontam = UNSWLoader.load()
    assert len(df_d5) == 175341, f"Expected 175,341 rows in D5, got {len(df_d5)}"
    assert len(df_d8_orig) == 82332, f"Expected 82,332 rows in original D8, got {len(df_d8_orig)}"
    assert len(df_d8_decontam) == 73791, f"Expected 73,791 rows in decontaminated D8, got {len(df_d8_decontam)}"
    assert len(df_d8_orig) - len(df_d8_decontam) == 8541, "Expected exactly 8,541 leaked rows purged"


def test_ton_iot_loader_pruning():
    """Verify ToN_IoT loader successfully prunes the 34,876 empty simulator idling rows."""
    expected_rows = {
        "garage_door": 239955 - 28379,  # 211,576
        "thermostat": 175342 - 4621,    # 170,721
        "modbus": 376022 - 1876,        # 374,146
        "fridge": 232382,
        "gps_tracker": 236982,
        "motion_light": 182653,
        "weather": 285632,
    }
    
    for dev in TON_IOT_DEVICES:
        df = ToNIoTLoader.load_device(dev, prune_empty=True)
        assert len(df) == expected_rows[dev], (
            f"Device {dev}: expected {expected_rows[dev]} rows, got {len(df)}"
        )
        # Verify no missing timestamps in pruned dataset
        assert not df["date"].isnull().any(), f"Device {dev} has null dates after pruning"
        assert not df["time"].isnull().any(), f"Device {dev} has null times after pruning"


def test_cicids_loader_conflict_isolation():
    """Verify CIC-IDS2017 loader can isolate the 1,436 conflicting-label rows."""
    df_main, df_conflicts = CICIDSLoader.load(deduplicated=True, exclude_conflicts=True)
    assert len(df_conflicts) == 1436, f"Expected 1,436 conflicting rows, got {len(df_conflicts)}"
    assert len(df_main) == 2498185 - 1436, (
        f"Expected {2498185 - 1436} clean non-conflicting rows, got {len(df_main)}"
    )
