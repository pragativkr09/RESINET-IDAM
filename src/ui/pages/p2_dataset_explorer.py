"""
RESINET-IDAM: IT Dataset Explorer Page (src/ui/pages/p2_dataset_explorer.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from src.ui.components import (
    render_page_header,
    render_metric_card,
    render_offline_demarcation_banner,
)
from src.ui.utils import load_sample_traffic, get_expected_feature_names
from src.config import DATASET_PATHS


def render_dataset_explorer_page():
    render_page_header(
        title="IT Dataset Explorer",
        subtitle="Deep Telemetry Inspection & Statistical Profiling for CIC-IDS2017",
        badge="TELEMETRY AUDIT",
    )

    render_offline_demarcation_banner()

    st.markdown("## 🔍 Dataset Selection & Source Mode")
    source_mode = st.radio(
        "Select Telemetry Source to Inspect:",
        options=[
            "Verified Sample Telemetry (60 Representative Flows)",
            "CIC-IDS2017 Parquet Master Store (2,498,185 Records Metadata)",
            "Upload Custom Network Traffic CSV",
        ],
        horizontal=True,
    )

    expected_features = get_expected_feature_names()

    if source_mode == "Verified Sample Telemetry (60 Representative Flows)":
        df = load_sample_traffic()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Samples", f"{len(df):,}", "Representative Benchmark Sample")
        with col2:
            render_metric_card("Total Columns", f"{len(df.columns)}", "78 Features + 2 Metadata")
        with col3:
            render_metric_card("Missing Values", "0", "0.00% Missing Data")
        with col4:
            render_metric_card("Distinct Classes", f"{df['Label'].nunique()}", "Benign & Multi-Attack")

        st.markdown("### 📋 Sample Telemetry Preview")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 Class Distribution in Sample")
        st.bar_chart(df["Label"].value_counts())

        st.markdown("### 📈 Feature Statistical Summary")
        selected_features = st.multiselect(
            "Select Features for Statistical Breakdown:",
            options=expected_features,
            default=expected_features[:6],
        )
        if selected_features:
            st.dataframe(df[selected_features].describe().T, use_container_width=True)

    elif source_mode == "CIC-IDS2017 Parquet Master Store (2,498,185 Records Metadata)":
        st.info("Loading metadata from frozen Parquet store: `processed/it_cicids2017_deduplicated.parquet`")
        parquet_path = DATASET_PATHS["cicids2017_dedup"]
        pq_file = pq.ParquetFile(parquet_path)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Parquet Rows", f"{pq_file.metadata.num_rows:,}", "Verified Deduplicated Store")
        with col2:
            render_metric_card("Row Groups", f"{pq_file.num_row_groups}", "Optimized Stream Storage")
        with col3:
            render_metric_card("Conflicting Rows", "1,436", "718 Pairs Isolated")
        with col4:
            render_metric_card("Clean Core Rows", "2,496,749", "Zero Conflict Population")

        st.markdown("### 📂 Source Capture File Distribution (Phase A Audit)")
        capture_breakdown = [
            {"Capture Day": "Monday", "Source File": "Monday-WorkingHours.pcap_ISCX.csv", "Clean Records": "492,653", "Attacks": "0 (Baseline Clean)"},
            {"Capture Day": "Tuesday", "Source File": "Tuesday-WorkingHours.pcap_ISCX.csv", "Clean Records": "406,466", "Attacks": "FTP & SSH Patator"},
            {"Capture Day": "Wednesday", "Source File": "Wednesday-workingHours.pcap_ISCX.csv", "Clean Records": "586,448", "Attacks": "DoS Hulk, GoldenEye, slowloris, Heartbleed"},
            {"Capture Day": "Thursday Morning", "Source File": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Clean Records": "154,475", "Attacks": "Brute Force, XSS, SQL Injection"},
            {"Capture Day": "Thursday Afternoon", "Source File": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Clean Records": "241,861", "Attacks": "Infiltration Exploits"},
            {"Capture Day": "Friday Morning", "Source File": "Friday-WorkingHours-Morning.pcap_ISCX.csv", "Clean Records": "180,087", "Attacks": "ARES Botnet Telemetry"},
            {"Capture Day": "Friday Afternoon", "Source File": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Clean Records": "211,684", "Attacks": "PortScan Probing"},
            {"Capture Day": "Friday Afternoon", "Source File": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Clean Records": "223,075", "Attacks": "High-Volume LOIC DDoS"},
        ]
        st.dataframe(pd.DataFrame(capture_breakdown), use_container_width=True, hide_index=True)

        st.markdown("### 🛡️ Feature Schema Definition (78 Network Attributes)")
        st.write(", ".join([f"`{c}`" for c in expected_features]))

    else:
        # Custom CSV Upload
        uploaded_file = st.file_uploader("Upload Network Traffic CSV for Schema Audit:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_user = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded uploaded CSV: {uploaded_file.name} ({len(df_user):,} rows, {len(df_user.columns)} columns)")

                # Column alignment audit
                missing = [c for c in expected_features if c not in df_user.columns]
                extra = [c for c in df_user.columns if c not in expected_features and c not in ["source_file", "Label"]]

                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric_card("Uploaded Rows", f"{len(df_user):,}", uploaded_file.name)
                with col2:
                    matched = len(expected_features) - len(missing)
                    render_metric_card("Matched Features", f"{matched} / 78", f"{matched/78*100:.1f}% Schema Match")
                with col3:
                    render_metric_card("Missing Features", f"{len(missing)}", "Required for Inference")

                if missing:
                    st.warning(f"⚠️ Notice: Uploaded CSV is missing {len(missing)} required features for model evaluation: {', '.join(missing[:5])}...")
                else:
                    st.success("✅ Perfect Schema Compatibility: All 78 standard network flow features are present and ready for inference!")

                st.markdown("### 📋 Uploaded Data Preview (First 50 Rows)")
                st.dataframe(df_user.head(50), use_container_width=True)

                st.markdown("### 📊 Missing Value Audit")
                null_counts = df_user.isnull().sum()
                has_nulls = null_counts[null_counts > 0]
                if not has_nulls.empty:
                    st.dataframe(has_nulls.reset_index().rename(columns={"index": "Feature", 0: "Missing Rows"}))
                else:
                    st.info("Zero missing values detected across all uploaded columns.")

            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")
        else:
            st.info("Upload any network telemetry CSV file above to audit its schema, missing values, and column compatibility with the RESINET-IDAM model suite.")


# Standard entry point alias
render_page = render_dataset_explorer_page


