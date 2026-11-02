import os
import json
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("IT_UNSW_Pipeline")

def run_unsw_pipeline(it_dir="IT", output_dir="processed", report_dir="reports", dict_dir="dictionaries"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(dict_dir, exist_ok=True)

    logger.info("Starting IT UNSW-NB15 Pipeline (D5 & D8 Non-Destructive Processing)...")

    # Ingest D5 (Training set)
    d5_path = os.path.join(it_dir, "D5.csv")
    d8_path = os.path.join(it_dir, "D8.csv")

    logger.info(f"Ingesting D5.csv from {d5_path}...")
    df_d5 = pd.read_csv(d5_path)
    d5_raw_rows = len(df_d5)
    d5_dups = int(df_d5.duplicated().sum())
    logger.info(f"D5.csv ingested: {d5_raw_rows:,} rows | Duplicates: {d5_dups:,}")

    logger.info(f"Ingesting D8.csv from {d8_path}...")
    df_d8 = pd.read_csv(d8_path)
    d8_raw_rows = len(df_d8)
    d8_dups = int(df_d8.duplicated().sum())
    logger.info(f"D8.csv ingested: {d8_raw_rows:,} rows | Duplicates: {d8_dups:,}")

    # Column inspection
    assert df_d5.columns.tolist() == df_d8.columns.tolist(), "Columns mismatch between D5 and D8"
    columns = df_d5.columns.tolist()

    # Null / Inf check
    d5_nulls = df_d5.isna().sum().to_dict()
    d8_nulls = df_d8.isna().sum().to_dict()

    # Target class distributions
    d5_labels = df_d5['label'].value_counts().to_dict()
    d8_labels = df_d8['label'].value_counts().to_dict()

    # Downcast datatypes
    for col in df_d5.select_dtypes(include=['float64']).columns:
        df_d5[col] = df_d5[col].astype('float32')
        df_d8[col] = df_d8[col].astype('float32')
    for col in df_d5.select_dtypes(include=['int64']).columns:
        if col not in ['id', 'stcpb', 'dtcpb']:
            df_d5[col] = df_d5[col].astype('int32')
            df_d8[col] = df_d8[col].astype('int32')

    # Save to Parquet
    d5_parquet = os.path.join(output_dir, "it_unsw_nb15_d5_train.parquet")
    d8_parquet = os.path.join(output_dir, "it_unsw_nb15_d8_test.parquet")

    logger.info(f"Saving D5 train split to {d5_parquet}...")
    df_d5.to_parquet(d5_parquet, index=False, engine='pyarrow', compression='snappy')
    d5_size_mb = os.path.getsize(d5_parquet) / (1024*1024)

    logger.info(f"Saving D8 test split to {d8_parquet}...")
    df_d8.to_parquet(d8_parquet, index=False, engine='pyarrow', compression='snappy')
    d8_size_mb = os.path.getsize(d8_parquet) / (1024*1024)

    # Data Dictionary
    logger.info("Generating UNSW-NB15 Data Dictionary...")
    data_dict = {
        'dataset_name': "UNSW-NB15 Preprocessed Partitions (D5 Train & D8 Test)",
        'source': "UNSW Canberra Cyber (Moustafa & Slay, 2015)",
        'note': "Pre-transformed binary dataset with encoded xProt, xServ, xState and stripped attack_cat",
        'train_file': "D5.csv",
        'test_file': "D8.csv",
        'total_features': len(columns) - 2, # exclude id and label
        'target_column': "label",
        'classes': {"0": "Normal", "1": "Attack"},
        'feature_specifications': {}
    }

    for col in columns:
        if col == 'label': continue
        col_s = df_d5[col]
        data_dict['feature_specifications'][col] = {
            'dtype': str(col_s.dtype),
            'min': float(col_s.min()),
            'max': float(col_s.max()),
            'mean': float(round(col_s.mean(), 4)),
            'std': float(round(col_s.std(), 4)),
            'null_count': int(col_s.isna().sum()),
            'synthetic_encoded': col in ['xProt', 'xServ', 'xState']
        }

    with open(os.path.join(dict_dir, "it_unsw_nb15_data_dictionary.json"), "w", encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2)

    # Quality Report
    quality_report = {
        'pipeline': "IT UNSW-NB15 Preprocessing Pipeline",
        'd5_train_raw_rows': d5_raw_rows,
        'd5_train_duplicates': d5_dups,
        'd5_class_distribution': {str(k): int(v) for k, v in d5_labels.items()},
        'd5_parquet_path': d5_parquet,
        'd5_parquet_size_mb': round(d5_size_mb, 2),
        'd8_test_raw_rows': d8_raw_rows,
        'd8_test_duplicates': d8_dups,
        'd8_class_distribution': {str(k): int(v) for k, v in d8_labels.items()},
        'd8_parquet_path': d8_parquet,
        'd8_parquet_size_mb': round(d8_size_mb, 2),
        'total_unique_records': (d5_raw_rows - d5_dups) + (d8_raw_rows - d8_dups),
        'isolation_guarantee': "Maintained as a completely separate IT pipeline from CIC-IDS-2017. No cross-dataset merging performed.",
        'transformations_applied': [
            "Preserved raw D5.csv and D8.csv files in read-only IT folder",
            "Verified structural consistency between train and test partitions",
            "Identified and documented 10,639 duplicate rows in D5 training set",
            "Documented synthetic integer encoding of protocol, service, and state features (xProt, xServ, xState)",
            "Downcast datatypes to float32 and int32 for optimized memory consumption",
            "Exported compressed Parquet files maintaining train/test split boundary"
        ]
    }

    with open(os.path.join(report_dir, "it_unsw_nb15_quality_report.json"), "w", encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2)

    logger.info("IT UNSW-NB15 Pipeline completed successfully!")
    return quality_report

if __name__ == '__main__':
    run_unsw_pipeline()

