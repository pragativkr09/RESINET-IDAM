"""
RESINET-IDAM: Multiclass Intrusion Detection Page (src/ui/pages/p3_multiclass_ids.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.ui.components import (
    render_page_header,
    render_metric_card,
    render_offline_demarcation_banner,
    apply_cyber_theme_to_plotly,
)
from src.ui.utils import (
    load_sample_traffic,
    run_multiclass_inference,
    get_expected_feature_names,
)
from src.db.mysql_manager import db_manager


def render_multiclass_ids_page():
    render_page_header(
        title="Multiclass Intrusion Detection Engine",
        subtitle="Real-Time Supervised Flow Classification Across 15 Attack Categories",
        badge="INFERENCE ONLINE",
    )

    render_offline_demarcation_banner("Inference")

    # Traffic input section
    st.markdown("## 📥 Ingest Network Telemetry")
    input_source = st.radio(
        "Select Telemetry Ingestion Source:",
        options=[
            "Verified Sample Telemetry (Instant 1-Click Test)",
            "Upload Custom Network Traffic CSV",
        ],
        horizontal=True,
    )

    df_to_predict = None

    if input_source == "Verified Sample Telemetry (Instant 1-Click Test)":
        df_sample = load_sample_traffic()
        st.info(f"Loaded verified benchmark sample containing {len(df_sample)} network flows across diverse attack classes.")
        if st.button("🚀 Execute Multiclass Inference on Sample", type="primary"):
            df_to_predict = df_sample
    else:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV for Multiclass Scanning:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_user = pd.read_csv(uploaded_file)
                st.success(f"File uploaded: {uploaded_file.name} ({len(df_user):,} rows)")
                if st.button("🚀 Scan Uploaded Telemetry", type="primary"):
                    df_to_predict = df_user
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # Inference Execution & Results
    if df_to_predict is not None:
        with st.spinner("Executing Leak-Free Scaling & Random Forest Multi-Class Inference..."):
            try:
                results_df, summary = run_multiclass_inference(df_to_predict)

                # Log user action to MySQL audit log
                db_manager.log_audit(
                    event_type="INFERENCE_MULTICLASS",
                    user_action=f"Scanned {summary['total_flows']} network flows",
                    details=f"Detected {summary['malicious_flows']} attacks ({summary['attack_rate']}%)",
                )

                # Log individual alerts into MySQL
                for idx, row in results_df[results_df["Predicted_Class"] != "BENIGN"].iterrows():
                    db_manager.log_detection_alert(
                        flow_id=str(idx),
                        detection_engine="MULTICLASS_SUPERVISED",
                        predicted_class=row["Predicted_Class"],
                        confidence_score=row["Confidence"],
                        anomaly_score=0.0,
                        threat_severity=row["Threat_Severity"],
                        dest_port=int(row.get("Destination Port", 0)),
                    )

                st.success("✅ Multiclass Flow Classification Complete!")

                # Summary KPI Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card("Total Flows Scanned", f"{summary['total_flows']:,}", "Analyzed Telemetry Records")
                with col2:
                    render_metric_card(
                        "Malicious Flows",
                        f"{summary['malicious_flows']:,}",
                        f"{summary['attack_rate']}% Attack Ratio",
                        delta=f"{summary['malicious_flows']} Threats",
                        delta_color="inverse",
                    )
                with col3:
                    render_metric_card("Benign Flows", f"{summary['benign_flows']:,}", "Verified Normal Telemetry")
                with col4:
                    render_metric_card(
                        "Active Model",
                        "RandomForest",
                        "100 Estimators | Max Depth 25",
                        delta="0.8108 Macro F1",
                    )

                # Visualizations
                st.markdown("## 📊 Threat Attribution & Confidence Profile")
                v_col1, v_col2 = st.columns(2)

                with v_col1:
                    # Attack class breakdown
                    class_counts = summary["class_counts"]
                    df_classes = pd.DataFrame(list(class_counts.items()), columns=["Category", "Count"]).sort_values("Count", ascending=True)
                    fig_classes = px.bar(
                        df_classes,
                        x="Count",
                        y="Category",
                        orientation="h",
                        title="Attributed Attack Categories",
                        color="Count",
                        color_continuous_scale=["#0ea5e9", "#ef4444"],
                    )
                    apply_cyber_theme_to_plotly(fig_classes)
                    st.plotly_chart(fig_classes, use_container_width=True)

                with v_col2:
                    # Confidence histogram
                    fig_conf = px.histogram(
                        results_df,
                        x="Confidence",
                        nbins=20,
                        title="Model Prediction Confidence Distribution (%)",
                        color="Threat_Severity",
                        color_discrete_map={
                            "CRITICAL": "#ef4444",
                            "HIGH": "#f97316",
                            "MEDIUM": "#eab308",
                            "LOW": "#3b82f6",
                            "BENIGN": "#10b981",
                        },
                    )
                    apply_cyber_theme_to_plotly(fig_conf)
                    st.plotly_chart(fig_conf, use_container_width=True)

                # Detailed Telemetry Table
                st.markdown("## 📋 Inspected Telemetry & Classification Log")

                # Filter options
                filter_class = st.selectbox(
                    "Filter by Predicted Class:",
                    options=["ALL"] + list(summary["class_counts"].keys()),
                )

                display_df = results_df if filter_class == "ALL" else results_df[results_df["Predicted_Class"] == filter_class]

                display_cols = [
                    "Predicted_Class", "Confidence", "Threat_Severity",
                    "Destination Port", "Flow Duration", "Total Fwd Packets",
                    "Total Backward Packets", "Flow Bytes/s"
                ]
                # Include existing cols if present
                valid_display_cols = [c for c in display_cols if c in display_df.columns]
                st.dataframe(display_df[valid_display_cols], use_container_width=True)

                # Download Results CSV
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Inference Results (CSV)",
                    data=csv_bytes,
                    file_name="resinet_multiclass_inference_results.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Inference error: {e}")


# Standard entry point alias
render_page = render_multiclass_ids_page


