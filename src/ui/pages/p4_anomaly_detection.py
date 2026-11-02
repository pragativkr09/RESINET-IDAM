"""
RESINET-IDAM: Open-Set Anomaly Detection Page (src/ui/pages/p4_anomaly_detection.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import (
    render_page_header,
    render_metric_card,
    render_offline_demarcation_banner,
    apply_cyber_theme_to_plotly,
)
from src.ui.utils import (
    load_sample_traffic,
    run_anomaly_inference,
    load_benchmark_data,
)
from src.db.mysql_manager import db_manager


def render_anomaly_detection_page():
    render_page_header(
        title="Open-Set Anomaly Detection Engine",
        subtitle="Unsupervised Out-of-Distribution Threat Detection via Isolation Forest",
        badge="TRIPWIRE ONLINE",
    )

    render_offline_demarcation_banner("Anomaly")

    benchmark = load_benchmark_data()
    t2_base = benchmark.get("track_2_open_set_streaming", {}).get("baseline_training", {})
    thresholds = t2_base.get("calibrated_thresholds", {
        "tau_90": 0.475982,
        "tau_95": 0.551141,
        "tau_99": 0.629093,
    })

    # Threshold selection section
    st.markdown("## ⚙️ Calibrated Anomaly Threshold Configuration")
    thresh_col1, thresh_col2 = st.columns([1, 2])

    with thresh_col1:
        thresh_mode = st.selectbox(
            "Select Calibrated Operating Threshold:",
            options=[
                "tau_95: 0.551141 (Balanced 5% Nominal False Alarm Baseline)",
                "tau_90: 0.475982 (High Sensitivity 10% Nominal False Alarm Baseline)",
                "tau_99: 0.629093 (High Specificity 1% Nominal False Alarm Baseline)",
                "Custom Threshold Slider",
            ],
        )

    with thresh_col2:
        if "tau_95" in thresh_mode:
            active_threshold = thresholds.get("tau_95", 0.551141)
            st.info("Operating at **tau_95 (0.551141)**: Recommended operational balance between high threat detection (76.1% DoS, 80.6% Infiltration) and low false alarm rate (~6.9%).")
        elif "tau_90" in thresh_mode:
            active_threshold = thresholds.get("tau_90", 0.475982)
            st.info("Operating at **tau_90 (0.475982)**: Maximum sensitivity tripwire mode for high-security environments.")
        elif "tau_99" in thresh_mode:
            active_threshold = thresholds.get("tau_99", 0.629093)
            st.info("Operating at **tau_99 (0.629093)**: High specificity mode, flagging only severe statistical deviations.")
        else:
            active_threshold = st.slider("Select Custom Cutoff Threshold:", min_value=0.30, max_value=0.85, value=0.55, step=0.01)

    # Ingestion Source
    st.markdown("## 📥 Ingest Telemetry for Anomaly Scanning")
    input_source = st.radio(
        "Select Telemetry Source:",
        options=[
            "Verified Sample Telemetry (Instant 1-Click Test)",
            "Upload Custom Network Traffic CSV",
        ],
        horizontal=True,
    )

    df_to_score = None

    if input_source == "Verified Sample Telemetry (Instant 1-Click Test)":
        df_sample = load_sample_traffic()
        st.info(f"Loaded verified benchmark sample containing {len(df_sample)} network flows.")
        if st.button("🚀 Calculate Anomaly Scores on Sample", type="primary"):
            df_to_score = df_sample
    else:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV for Anomaly Scoring:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_user = pd.read_csv(uploaded_file)
                st.success(f"File uploaded: {uploaded_file.name} ({len(df_user):,} rows)")
                if st.button("🚀 Execute Anomaly Scoring", type="primary"):
                    df_to_score = df_user
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # Anomaly Inference Execution
    if df_to_score is not None:
        with st.spinner("Scoring Telemetry against Monday Benign Baseline..."):
            try:
                results_df, summary = run_anomaly_inference(df_to_score, active_threshold)

                # Log audit event
                db_manager.log_audit(
                    event_type="INFERENCE_ANOMALY",
                    user_action=f"Scored {summary['total_flows']} network flows (Threshold: {active_threshold})",
                    details=f"Identified {summary['anomalous_flows']} anomalies ({summary['anomaly_rate']}%)",
                )

                # Log alert
                for idx, row in results_df[results_df["Is_Anomaly"]].iterrows():
                    db_manager.log_detection_alert(
                        flow_id=str(idx),
                        detection_engine="OPEN_SET_ANOMALY",
                        predicted_class="ANOMALY_DEVIATION",
                        confidence_score=row["Anomaly_Score"],
                        anomaly_score=row["Anomaly_Score"],
                        threat_severity=row["Anomaly_Severity"],
                        dest_port=int(row.get("Destination Port", 0)),
                    )

                st.success("✅ Unsupervised Anomaly Scoring Complete!")

                # KPI Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card("Total Scanned Flows", f"{summary['total_flows']:,}", "Out-of-Distribution Inspection")
                with col2:
                    render_metric_card(
                        "Anomalies Flagged",
                        f"{summary['anomalous_flows']:,}",
                        f"{summary['anomaly_rate']}% of Inspected Traffic",
                        delta=f"{summary['anomalous_flows']} Flagged",
                        delta_color="inverse",
                    )
                with col3:
                    render_metric_card("Normal Baseline Flows", f"{summary['normal_flows']:,}", "Below Anomaly Cutoff")
                with col4:
                    render_metric_card(
                        "Applied Threshold",
                        f"{active_threshold:.4f}",
                        f"Score Range: [{summary['min_score']:.2f}, {summary['max_score']:.2f}]",
                        delta="Calibrated",
                    )

                # Visualizations
                st.markdown("## 📊 Anomaly Score Distribution & Cutoff Threshold")
                v_col1, v_col2 = st.columns(2)

                with v_col1:
                    # Score histogram with threshold line
                    fig_hist = px.histogram(
                        results_df,
                        x="Anomaly_Score",
                        nbins=30,
                        title="Continuous Anomaly Score Distribution",
                        color="Anomaly_Severity",
                        color_discrete_map={
                            "CRITICAL": "#ef4444",
                            "HIGH": "#f97316",
                            "MEDIUM": "#eab308",
                            "LOW": "#3b82f6",
                            "BENIGN": "#10b981",
                        },
                    )
                    fig_hist.add_vline(
                        x=active_threshold,
                        line_width=2,
                        line_dash="dash",
                        line_color="#38bdf8",
                        annotation_text=f"Threshold: {active_threshold:.4f}",
                        annotation_position="top left",
                    )
                    apply_cyber_theme_to_plotly(fig_hist)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with v_col2:
                    # Severity Breakdown Donut Chart
                    sev_counts = summary["severity_counts"]
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(sev_counts.keys()),
                        values=list(sev_counts.values()),
                        hole=0.5,
                        marker=dict(colors=["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"]),
                    )])
                    fig_pie.update_layout(title="Flagged Deviation Severity Tiers")
                    apply_cyber_theme_to_plotly(fig_pie)
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Table of Scored Flows
                st.markdown("## 📋 Inspected Telemetry Scores")
                show_anomalies_only = st.checkbox("Show Only Anomalies Exceeding Cutoff", value=True)
                table_df = results_df[results_df["Is_Anomaly"]] if show_anomalies_only else results_df

                display_cols = [
                    "Anomaly_Score", "Is_Anomaly", "Anomaly_Severity", "Destination Port",
                    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s"
                ]
                valid_cols = [c for c in display_cols if c in table_df.columns]
                st.dataframe(table_df[valid_cols], use_container_width=True)

                # Download Results CSV
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Anomaly Scores (CSV)",
                    data=csv_bytes,
                    file_name="resinet_anomaly_scores_results.csv",
                    mime="text/csv",
                )

                # Architectural Limitations & Operational Context
                st.markdown("## 🔬 Benchmark Methodology & Architectural Scope")
                st.markdown(
                    """
                    > **OPERATIONAL TRIPWIRE DESIGN:**
                    > - **Baseline Guarantee:** The Isolation Forest was trained strictly on Monday benign traffic ($N = 492,653$ flows). It has never seen attack flows during fitting.
                    > - **Volumetric Detection Strength:** As proven in Track 2 streaming benchmarks, the model achieves high detection recall on massive deviations such as DoS Hulk (76.10% DR, AUC 0.9189) and Infiltration (80.56% DR, AUC 0.9368).
                    > - **Single-Packet Stealth Limitations:** Stealthy probing attacks (such as single-packet PortScan, 0.11% DR) structurally mimic normal SYN/FIN handshakes, yielding lower anomaly scores in flow duration/volume dimensions.
                    > - **Hybrid Architecture:** To overcome this limitation, RESINET-IDAM pairs this unsupervised tripwire with the Track 1 Random Forest closed-set classifier (which detects PortScan with 99.8% precision).
                    """
                )

            except Exception as e:
                st.error(f"Scoring error: {e}")


# Standard entry point alias
render_page = render_anomaly_detection_page


