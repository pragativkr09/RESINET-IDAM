"""
RESINET-IDAM: Threat Intelligence & Analysis Page (src/ui/pages/p6_threat_intel.py)
Taxonomy, Campaign-wise evaluations (A-F), MITRE ATT&CK mappings, and empirical threat analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import (
    render_page_header,
    render_metric_card,
    render_threat_badge,
    render_offline_demarcation_banner,
    apply_cyber_theme_to_plotly,
)
from src.ui.utils import load_benchmark_data


def render_page():
    render_page_header(
        title="Threat Intelligence & Attack Analysis",
        subtitle="Empirical attack taxonomy, MITRE ATT&CK mappings, and multi-campaign evaluation",
        badge="MITRE ALIGNED",
    )
    render_offline_demarcation_banner()

    data = load_benchmark_data()
    track2 = data.get("track_2_open_set_streaming", {})
    campaigns = track2.get("campaign_evaluations", {})

    # Top Metric Ribbon
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            label="Total Attack Categories",
            value="14 Classes",
            subtext="Plus Benign Baseline",
            delta="Complete Taxonomy",
        )
    with col2:
        render_metric_card(
            label="Evaluated Campaigns",
            value="6 Campaigns",
            subtext="Withheld Friday & Thursday Flows",
            delta="A through F",
        )
    with col3:
        render_metric_card(
            label="Withheld Attack Volume",
            value="415,873 Flows",
            subtext="Zero-Day Anomaly Evaluation",
            delta="1.59M Total Flows",
        )
    with col4:
        render_metric_card(
            label="Highest Anomaly AUC",
            value="0.9368",
            subtext="Campaign C (Infiltration)",
            delta="High Sensitivity",
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚔️ Attack Taxonomy & MITRE ATT&CK",
        "🎯 Multi-Campaign Empirical Evaluations (A-F)",
        "🔍 Anomaly Detection Profile Analysis",
        "🛡️ Zero-Trust Defense Posture",
    ])

    # TAB 1: ATTACK TAXONOMY & MITRE
    with tab1:
        st.markdown("### IT Intrusion Taxonomy & Threat Profiles")
        st.markdown(
            "Every attack category in the CIC-IDS2017 dataset mapped to its MITRE ATT&CK technique, "
            "typical transport protocol, signature footprint, and severity tier."
        )

        threat_matrix = [
            {
                "Attack Category": "Heartbleed",
                "Severity": "CRITICAL",
                "MITRE ID": "T1190",
                "Technique Name": "Exploit Public-Facing Application",
                "Protocol": "TLS / TCP 443",
                "Flow Footprint": "Malformed heartbeat payload, small fwd packet size, large unencrypted memory leak bwd response",
            },
            {
                "Attack Category": "Infiltration",
                "Severity": "CRITICAL",
                "MITRE ID": "T1059",
                "Technique Name": "Command and Scripting Interpreter",
                "Protocol": "TCP Multi-Port",
                "Flow Footprint": "Low-and-slow lateral movement, anomalous inter-packet arrival times, abnormal client TCP window size",
            },
            {
                "Attack Category": "DDoS (LOIC/HOIC)",
                "Severity": "HIGH",
                "MITRE ID": "T1498.001",
                "Technique Name": "Direct Network Flood",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Massive packet burst, symmetric small fwd packet sizes, near-zero flow duration, flooded ACK/SYN flags",
            },
            {
                "Attack Category": "DoS Hulk",
                "Severity": "HIGH",
                "MITRE ID": "T1499.002",
                "Technique Name": "HTTP Flood / Resource Exhaustion",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Obfuscated user agents, distinct request parameters, high forward packet rates, exhausted server thread pools",
            },
            {
                "Attack Category": "DoS GoldenEye",
                "Severity": "HIGH",
                "MITRE ID": "T1499.002",
                "Technique Name": "Keep-Alive HTTP Exhaustion",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Persistent HTTP Keep-Alive connections, elevated Flow Duration, high idle times, starving web workers",
            },
            {
                "Attack Category": "DoS Slowloris",
                "Severity": "HIGH",
                "MITRE ID": "T1499.003",
                "Technique Name": "Slow Request DoS",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Extremely slow headers transmission, high Flow Duration, low packet counts, minimal bandwidth consumed",
            },
            {
                "Attack Category": "DoS Slowhttptest",
                "Severity": "HIGH",
                "MITRE ID": "T1499.003",
                "Technique Name": "Slow Read / Slow Post DoS",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Slow read buffer drainage (zero window probes), extended flow duration, low packet throughput",
            },
            {
                "Attack Category": "Botnet (ARES)",
                "Severity": "HIGH",
                "MITRE ID": "T1071.001",
                "Technique Name": "Web Protocols C2 Beaconing",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Periodic beaconing, fixed flow IAT intervals, small request/response byte lengths, command polling",
            },
            {
                "Attack Category": "Web Attack - SQL Injection",
                "Severity": "HIGH",
                "MITRE ID": "T1190",
                "Technique Name": "Exploit Public-Facing Application",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Altered HTTP query parameters, altered payload length, anomalous response sizes with DB error strings",
            },
            {
                "Attack Category": "Web Attack - XSS",
                "Severity": "MEDIUM",
                "MITRE ID": "T1059.007",
                "Technique Name": "JavaScript Injection",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Injected script tags in URI/POST data, standard HTTP flow header characteristics, reflected payload",
            },
            {
                "Attack Category": "Web Attack - Brute Force",
                "Severity": "MEDIUM",
                "MITRE ID": "T1110.001",
                "Technique Name": "Password Guessing",
                "Protocol": "HTTP / TCP 80",
                "Flow Footprint": "Rapid repetitive HTTP POST requests to login endpoints, high flow count from single host, small 401/403 responses",
            },
            {
                "Attack Category": "SSH-Patator",
                "Severity": "MEDIUM",
                "MITRE ID": "T1110.001",
                "Technique Name": "Password Guessing (SSH)",
                "Protocol": "SSH / TCP 22",
                "Flow Footprint": "Repeated key exchange handshakes, identical packet size sequence, uniform connection teardown durations",
            },
            {
                "Attack Category": "FTP-Patator",
                "Severity": "MEDIUM",
                "MITRE ID": "T1110.001",
                "Technique Name": "Password Guessing (FTP)",
                "Protocol": "FTP / TCP 21",
                "Flow Footprint": "Continuous USER/PASS command exchanges, rapid connection opens/closes, minimal data channel activity",
            },
            {
                "Attack Category": "PortScan",
                "Severity": "LOW",
                "MITRE ID": "T1046",
                "Technique Name": "Network Service Discovery",
                "Protocol": "TCP Multi-Port",
                "Flow Footprint": "Broad sweep across port ranges, isolated SYN packets without completing three-way handshake (RST response)",
            },
        ]

        df_tm = pd.DataFrame(threat_matrix)
        st.dataframe(df_tm, use_container_width=True, hide_index=True)

    # TAB 2: MULTI-CAMPAIGN EVALUATIONS
    with tab2:
        st.markdown("### Empirical Multi-Campaign Analysis (Campaigns A through F)")
        st.markdown(
            "Phase B Step B.5 withheld 6 complete attack campaigns ($N=1,597,630$ flows) to evaluate "
            "unsupervised Isolation Forest anomaly detection against zero-day attack shifts."
        )

        camp_summary = []
        for camp_name, details in campaigns.items():
            if camp_name.startswith("Combined"):
                continue
            t90 = details.get("threshold_evaluations", {}).get("tau_90", {})
            t95 = details.get("threshold_evaluations", {}).get("tau_95", {})
            t99 = details.get("threshold_evaluations", {}).get("tau_99", {})
            camp_summary.append({
                "Campaign": camp_name,
                "Total Flows": details.get("total_records", 0),
                "Attack Flows": details.get("attack_records", 0),
                "Attack %": f"{details.get('attack_records', 0) / max(1, details.get('total_records', 1)) * 100:.1f}%",
                "ROC-AUC": details.get("roc_auc", 0.0),
                "Det. Rate @ τ90": f"{t90.get('detection_rate', 0.0)*100:.2f}%",
                "FPR @ τ90": f"{t90.get('false_positive_rate', 0.0)*100:.2f}%",
                "Det. Rate @ τ95": f"{t95.get('detection_rate', 0.0)*100:.2f}%",
                "Det. Rate @ τ99": f"{t99.get('detection_rate', 0.0)*100:.2f}%",
            })

        df_camps = pd.DataFrame(camp_summary)

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_auc = px.bar(
                df_camps,
                x="Campaign",
                y="ROC-AUC",
                color="ROC-AUC",
                color_continuous_scale=["#ef4444", "#f59e0b", "#38bdf8", "#10b981"],
                text="ROC-AUC",
                title="Empirical ROC-AUC Across Withheld Zero-Day Campaigns",
            )
            fig_auc.update_layout(height=420, yaxis_range=[0, 1.05])
            fig_auc = apply_cyber_theme_to_plotly(fig_auc)
            st.plotly_chart(fig_auc, use_container_width=True)

        with col_c2:
            fig_vol = px.pie(
                df_camps,
                names="Campaign",
                values="Attack Flows",
                title="Distribution of Attack Flows Across Campaigns",
                hole=0.45,
                color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185", "#34d399"],
            )
            fig_vol.update_layout(height=420)
            fig_vol = apply_cyber_theme_to_plotly(fig_vol)
            st.plotly_chart(fig_vol, use_container_width=True)

        st.dataframe(df_camps, use_container_width=True, hide_index=True)

    # TAB 3: ANOMALY DETECTION PROFILE ANALYSIS
    with tab3:
        st.markdown("### Scientific Anomaly Profile Analysis")
        st.markdown(
            "Detailed empirical diagnosis of why different cyber attack vectors perform differently "
            "under single-flow statistical anomaly detection."
        )

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #38bdf8; margin-top: 0;">High Anomaly Detectability (AUC > 0.90)</h4>
                    <p><strong>Campaign A (DoS / Heartbleed) & Campaign C (Infiltration):</strong></p>
                    <ul style="color: #cbd5e1; font-size: 0.9rem;">
                        <li><strong>Volumetric Asymmetry:</strong> DoS Hulk and LOIC generate extreme forward packet counts and flow rates that diverge by orders of magnitude from typical client TCP flows.</li>
                        <li><strong>Memory Extraction Signatures:</strong> Heartbleed exploits return large unencrypted 64 KB memory segments, triggering extreme backward packet lengths not seen in normal HTTPS handshakes.</li>
                        <li><strong>Abnormal Handshake Mechanics:</strong> Infiltration lateral movement utilizes unusual TCP window sizes and non-standard segment configurations easily isolated by tree splits.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_a2:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #f59e0b; margin-top: 0;">Low Anomaly Detectability (AUC 0.53 - 0.68)</h4>
                    <p><strong>Campaign B (Web Attacks), Campaign D (Botnet), Campaign E (PortScan):</strong></p>
                    <ul style="color: #cbd5e1; font-size: 0.9rem;">
                        <li><strong>Statistical Camouflage:</strong> Web Attacks (SQLi, XSS) use standard HTTP GET/POST methods over standard port 80/443. At the NetFlow statistical layer, packet sizes and inter-arrival times mirror legitimate web surfing.</li>
                        <li><strong>Protocol Standard Compliance:</strong> Individual PortScan flows (SYN probes) look identical to legitimate connection setup attempts; anomaly detection requires multi-flow time-window correlation, not single-flow features.</li>
                        <li><strong>Low-Volume C2 Beaconing:</strong> ARES botnet agents sleep between check-ins, producing quiet, RFC-compliant HTTP transactions that evade single-flow volume thresholds.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "💡 **Defense Recommendation:** For low-detectability vectors (Web Attacks and PortScan), "
            "the system automatically routes traffic to Track 1 (Supervised Closed-Set Classifier), "
            "where semantic port and flag patterns achieve **>99.8% precision**, compensating for unsupervised limits."
        )

    # TAB 4: ZERO-TRUST DEFENSE POSTURE
    with tab4:
        st.markdown("### Zero-Trust & Defense-in-Depth Architecture")
        st.markdown(
            "RESINET-IDAM implements a defense-in-depth posture adhering to NIST SP 800-207 Zero-Trust principles."
        )

        st.markdown(
            """
            ```
            ┌────────────────────────────────────────────────────────────────────────┐
            │                     INCOMING TELEMETRY STREAM                          │
            └───────────────────────────────────┬────────────────────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │  LeakFreeScaler (78)  │
                                    └───────────┬───────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        │                                               │
            ┌───────────▼───────────┐                       ┌───────────▼───────────┐
            │   TRACK 1: MULTICLASS │                       │   TRACK 2: ANOMALY    │
            │   RandomForest (100)  │                       │   Isolation Forest    │
            └───────────┬───────────┘                       └───────────┬───────────┘
                        │ Known Attack Fingerprint                      │ Zero-Day Deviation Score
                        │ (P_class >= 0.50)                             │ (S_anomaly >= tau_threshold)
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │  DECISION ARBITRATOR  │
                                    └───────────┬───────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
            ┌───────▼───────┐           ┌───────▼───────┐           ┌───────▼───────┐
            │  KNOWN ATTACK │           │   ZERO-DAY    │           │ VERIFIED FLOW │
            │   IDENTIFIED  │           │   SUSPICION   │           │    BENIGN     │
            │ (Action: Alert)│          │ (Action: Triage)          │(Action: Audit)│
            └───────────────┘           └───────────────┘           └───────────────┘
            ```
            """
        )

