import os
import glob
import json
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("IT_CICIDS2017_Pipeline")

CANONICAL_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

EXCLUDED_DUPLICATE_FILES = ["D6.csv", "D7.csv"]

def clean_label(label_val):
    if not isinstance(label_val, str):
        return str(label_val)
    s = label_val.strip()
    # Normalize unicode replacement chars
    s = s.replace('\ufffd', '-').replace('\x96', '-').replace('–', '-')
    # Standardize names
    if "Web Attack" in s:
        if "Brute Force" in s: return "Web Attack - Brute Force"
        if "XSS" in s: return "Web Attack - XSS"
        if "Sql" in s or "SQL" in s: return "Web Attack - SQL Injection"
    if s == "BENIGN": return "BENIGN"
    if s == "Bot": return "Bot"
    if s == "DDoS": return "DDoS"
    if s == "PortScan": return "PortScan"
    if s == "Infiltration": return "Infiltration"
    if s == "FTP-Patator": return "FTP-Patator"
    if s == "SSH-Patator": return "SSH-Patator"
    if s == "DoS Hulk": return "DoS Hulk"
    if s == "DoS GoldenEye": return "DoS GoldenEye"
    if s == "DoS slowloris": return "DoS slowloris"
    if s == "DoS Slowhttptest": return "DoS Slowhttptest"
    if s == "Heartbleed": return "Heartbleed"
    return s

def run_cicids2017_pipeline(it_dir="IT", output_dir="processed", report_dir="reports", dict_dir="dictionaries"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(dict_dir, exist_ok=True)

    logger.info("Starting IT CIC-IDS-2017 Non-Destructive Ingestion Pipeline...")
    logger.info(f"Target canonical files count: {len(CANONICAL_FILES)}")
    logger.info(f"Explicitly excluding duplicate files: {EXCLUDED_DUPLICATE_FILES}")

    file_reports = []
    combined_dfs = []
    total_raw_rows = 0

    for fname in CANONICAL_FILES:
        fpath = os.path.join(it_dir, fname)
        if not os.path.exists(fpath):
            logger.error(f"File not found: {fpath}")
            continue

        fsize = os.path.getsize(fpath)
        logger.info(f"Ingesting {fname} ({fsize / (1024*1024):.2f} MB)...")

        # Read CSV with latin1 to handle raw bytes cleanly
        df = pd.read_csv(fpath, encoding='latin1', low_memory=False)
        raw_count = len(df)
        total_raw_rows += raw_count

        # 1. Clean Column Names (strip whitespace)
        cols_before = df.columns.tolist()
        df.columns = df.columns.str.strip()
        cols_after = df.columns.tolist()
        renamed_count = sum(1 for c1, c2 in zip(cols_before, cols_after) if c1 != c2)

        # 2. Clean Label Column
        if 'Label' in df.columns:
            df['Label'] = df['Label'].apply(clean_label)
            label_dist = df['Label'].value_counts().to_dict()
        else:
            label_dist = {}

        # 3. Check and Document Infinite / NaN Values
        rate_cols = [c for c in ['Flow Bytes/s', 'Flow Packets/s'] if c in df.columns]
        inf_report = {}
        for rc in rate_cols:
            infs = int(np.isinf(pd.to_numeric(df[rc], errors='coerce')).sum())
            nans = int(df[rc].isna().sum())
            inf_report[rc] = {'inf_count': infs, 'nan_count': nans}

        # 4. In-file duplicate count
        in_file_dups = int(df.duplicated().sum())

        # Tag source file for traceability
        df['source_file'] = fname

        file_reports.append({
            'file': fname,
            'size_mb': round(fsize / (1024*1024), 2),
            'raw_rows': raw_count,
            'in_file_duplicates': in_file_dups,
            'whitespace_padded_columns_cleaned': renamed_count,
            'rate_anomalies': inf_report,
            'label_distribution': label_dist
        })

        combined_dfs.append(df)

    logger.info(f"Concatenating {len(combined_dfs)} ingested files into master IT DataFrame...")
    master_df = pd.concat(combined_dfs, ignore_index=True)
    logger.info(f"Total raw concatenated rows: {len(master_df):,}")

    # 5. Coerce Rate Columns to Float and replace np.inf with NaN
    for rc in ['Flow Bytes/s', 'Flow Packets/s']:
        if rc in master_df.columns:
            master_df[rc] = pd.to_numeric(master_df[rc], errors='coerce')
            master_df[rc] = master_df[rc].replace([np.inf, -np.inf], np.nan)

    # 6. Global Duplicate Audit
    # Separate feature columns from metadata
    feature_cols = [c for c in master_df.columns if c not in ['source_file']]
    feature_cols_no_label = [c for c in feature_cols if c != 'Label']

    total_exact_dups = int(master_df.duplicated(subset=feature_cols).sum())
    logger.info(f"Total cross-file exact duplicate flows: {total_exact_dups:,} ({total_exact_dups / len(master_df) * 100:.2f}%)")

    # Duplicate breakdown by class
    dup_mask = master_df.duplicated(subset=feature_cols, keep='first')
    dup_by_class = master_df[dup_mask]['Label'].value_counts().to_dict()

    duplicate_report = {
        'total_flows_ingested': len(master_df),
        'total_duplicate_flows': total_exact_dups,
        'duplicate_percentage': round(total_exact_dups / len(master_df) * 100, 2),
        'duplicate_distribution_by_class': {str(k): int(v) for k, v in dup_by_class.items()},
        'per_file_summary': file_reports
    }

    with open(os.path.join(report_dir, "it_cicids2017_duplicate_report.json"), "w", encoding='utf-8') as f:
        json.dump(duplicate_report, f, indent=2)

    # 7. Generate Cleaned Full Version (preserving legitimate repetitions, removing inf/NaN rates)
    # Filter rows where rate columns are NaN (4,628 rows out of 2.83M, <0.16%)
    valid_rate_mask = master_df['Flow Bytes/s'].notna() & master_df['Flow Packets/s'].notna()
    cleaned_df = master_df[valid_rate_mask].copy()
    dropped_nan_rate_count = len(master_df) - len(cleaned_df)
    logger.info(f"Dropped {dropped_nan_rate_count:,} records with NaN/Inf rates (<0.16% total volume). Cleaned total: {len(cleaned_df):,}")

    # Downcast float64 to float32 and int64 to int32 to optimize memory and disk I/O
    for col in cleaned_df.select_dtypes(include=['float64']).columns:
        cleaned_df[col] = cleaned_df[col].astype('float32')
    for col in cleaned_df.select_dtypes(include=['int64']).columns:
        if col != 'Destination Port':
            cleaned_df[col] = cleaned_df[col].astype('int32')

    cleaned_parquet_path = os.path.join(output_dir, "it_cicids2017_cleaned.parquet")
    logger.info(f"Saving cleaned dataset to {cleaned_parquet_path}...")
    cleaned_df.to_parquet(cleaned_parquet_path, index=False, engine='pyarrow', compression='snappy')
    cleaned_size_mb = os.path.getsize(cleaned_parquet_path) / (1024*1024)
    logger.info(f"Cleaned Parquet saved successfully ({cleaned_size_mb:.2f} MB, {len(cleaned_df):,} rows)")

    # 8. Generate Deduplicated Version (Strictly removing redundant capture replay artifacts)
    dedup_parquet_path = os.path.join(output_dir, "it_cicids2017_deduplicated.parquet")
    dedup_df = cleaned_df.drop_duplicates(subset=feature_cols, keep='first').copy()
    logger.info(f"Saving deduplicated dataset to {dedup_parquet_path} ({len(dedup_df):,} rows)...")
    dedup_df.to_parquet(dedup_parquet_path, index=False, engine='pyarrow', compression='snappy')
    dedup_size_mb = os.path.getsize(dedup_parquet_path) / (1024*1024)
    logger.info(f"Deduplicated Parquet saved successfully ({dedup_size_mb:.2f} MB, {len(dedup_df):,} rows)")

    # 9. Build Comprehensive Data Dictionary
    logger.info("Generating IT CIC-IDS-2017 Data Dictionary...")
    data_dict = {
        'dataset_name': "CIC-IDS-2017 Cleaned Benchmark Corpus",
        'source': "Canadian Institute for Cybersecurity (UNB)",
        'canonical_files_ingested': CANONICAL_FILES,
        'excluded_files': EXCLUDED_DUPLICATE_FILES,
        'total_features': len(feature_cols_no_label),
        'target_column': "Label",
        'classes': [str(c) for c in cleaned_df['Label'].unique()],
        'feature_specifications': {}
    }

    # Profile top 20 numerical features for data dictionary summary
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        col_series = cleaned_df[col]
        data_dict['feature_specifications'][col] = {
            'dtype': str(col_series.dtype),
            'min': float(col_series.min()),
            'max': float(col_series.max()),
            'mean': float(round(col_series.mean(), 4)),
            'std': float(round(col_series.std(), 4)),
            'q25': float(round(col_series.quantile(0.25), 4)),
            'median': float(round(col_series.median(), 4)),
            'q75': float(round(col_series.quantile(0.75), 4)),
            'null_count': int(col_series.isna().sum())
        }

    with open(os.path.join(dict_dir, "it_cicids2017_data_dictionary.json"), "w", encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2)

    # 10. Master Quality Report
    quality_report = {
        'pipeline': "IT CIC-IDS-2017 Preprocessing Pipeline",
        'raw_rows_ingested': total_raw_rows,
        'dropped_rate_inf_nan_rows': dropped_nan_rate_count,
        'cleaned_full_rows': len(cleaned_df),
        'deduplicated_rows': len(dedup_df),
        'cleaned_parquet_path': cleaned_parquet_path,
        'cleaned_parquet_size_mb': round(cleaned_size_mb, 2),
        'deduplicated_parquet_path': dedup_parquet_path,
        'deduplicated_parquet_size_mb': round(dedup_size_mb, 2),
        'label_distribution_cleaned': {str(k): int(v) for k, v in cleaned_df['Label'].value_counts().items()},
        'label_distribution_deduplicated': {str(k): int(v) for k, v in dedup_df['Label'].value_counts().items()},
        'transformations_applied': [
            "Preserved raw source files in read-only IT folder",
            "Excluded D6.csv and D7.csv as redundant exact duplicate files",
            "Sanitized leading and trailing whitespace from all 79 column names",
            "Normalized encoding corruptions in Label values (Web Attack - Brute Force, XSS, SQLi)",
            "Coerced Flow Bytes/s and Flow Packets/s to float and converted np.inf to NaN",
            "Excluded NaN rate rows (<0.16%) to prevent gradient divergence during training",
            "Preserved all legitimate attack-related volumetric and timing outliers without artificial clipping",
            "Exported full cleaned Parquet and strictly deduplicated Parquet versions for empirical comparison"
        ]
    }

    with open(os.path.join(report_dir, "it_cicids2017_quality_report.json"), "w", encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2)

    logger.info("IT CIC-IDS-2017 Pipeline completed successfully!")
    return quality_report

if __name__ == '__main__':
    run_cicids2017_pipeline()

