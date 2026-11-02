"""
RESINET-IDAM: Global Configuration Module
Defines all directory paths, global random states, dataset schemas,
and hyperparameter defaults for Phase B pipelines.
"""

from pathlib import Path
import os

# Base Directories
WORKSPACE_ROOT = Path(os.path.abspath(".")).resolve()
DATASET_DIR = WORKSPACE_ROOT
PROCESSED_DIR = DATASET_DIR / "processed"
REPORTS_DIR = DATASET_DIR / "reports"
DICTIONARIES_DIR = DATASET_DIR / "dictionaries"
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

# Global Random State
RANDOM_SEED = 42

# Dataset Paths (Frozen Parquet Stores)
DATASET_PATHS = {
    # IT Pipeline
    "cicids2017_cleaned": PROCESSED_DIR / "it_cicids2017_cleaned.parquet",
    "cicids2017_dedup": PROCESSED_DIR / "it_cicids2017_deduplicated.parquet",
    "unsw_nb15_d5_train": PROCESSED_DIR / "it_unsw_nb15_d5_train.parquet",
    "unsw_nb15_d8_test": PROCESSED_DIR / "it_unsw_nb15_d8_test.parquet",
    
    # OT Pipeline
    "ot_ics_cleaned": PROCESSED_DIR / "ot_ics_d1_d4_cleaned.parquet",
    
    # ToN_IoT Per-Device Datasets
    "ton_iot_fridge": PROCESSED_DIR / "ot_ton_iot_fridge.parquet",
    "ton_iot_garage_door": PROCESSED_DIR / "ot_ton_iot_garage_door.parquet",
    "ton_iot_gps_tracker": PROCESSED_DIR / "ot_ton_iot_gps_tracker.parquet",
    "ton_iot_modbus": PROCESSED_DIR / "ot_ton_iot_modbus.parquet",
    "ton_iot_motion_light": PROCESSED_DIR / "ot_ton_iot_motion_light.parquet",
    "ton_iot_thermostat": PROCESSED_DIR / "ot_ton_iot_thermostat.parquet",
    "ton_iot_weather": PROCESSED_DIR / "ot_ton_iot_weather.parquet",
}

# ToN_IoT Device Types
TON_IOT_DEVICES = [
    "fridge",
    "garage_door",
    "gps_tracker",
    "modbus",
    "motion_light",
    "thermostat",
    "weather",
]

# Verified Known Idle / Blank Row Counts in Raw ToN_IoT (to be pruned on load)
TON_IOT_EMPTY_ROW_COUNTS = {
    "garage_door": 28379,
    "thermostat": 4621,
    "modbus": 1876,
    "fridge": 0,
    "gps_tracker": 0,
    "motion_light": 0,
    "weather": 0,
}

# CIC-IDS2017 15 Classes
CICIDS_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - SQL Injection",
    "Heartbleed",
]

# Campaign Mapping by Day for Open-Set Evaluation
CICIDS_CAMPAIGNS = {
    "Monday": "Monday-WorkingHours.pcap_ISCX.csv",  # 100% Benign
    "Tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",  # FTP/SSH Patator
    "Wednesday": "Wednesday-workingHours.pcap_ISCX.csv",  # DoS attacks + Heartbleed
    "Thursday_Morning": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",  # Web Attacks
    "Thursday_Afternoon": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",  # Infiltration
    "Friday_Morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",  # Botnet
    "Friday_Afternoon_PortScan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",  # PortScan
    "Friday_Afternoon_DDoS": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",  # DDoS
}

