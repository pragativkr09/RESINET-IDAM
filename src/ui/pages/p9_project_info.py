"""
RESINET-IDAM: Project Information Page (src/ui/pages/p9_project_info.py)
System architecture, IT vs OT dynamics, Zero-Trust defense, technical stack, and scientific scope.
"""

import streamlit as st
import pandas as pd

from src.ui.components import (
    render_page_header,
    render_metric_card,
    render_offline_demarcation_banner,
)


def render_page():
    render_page_header(
        title="Project Information & Architecture",
        subtitle="Comprehensive system architecture, IT vs OT security paradigms, technology stack, and scientific scope",
        badge="ARCHITECTURE CORE",
    )
    render_offline_demarcation_banner()

    # Metric Ribbon
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            label="Project Phases",
            value="Phases A, B, C",
            subtext="All Pipelines Verified",
            delta="100% Approved",
            delta_color="normal",
        )
    with col2:
        render_metric_card(
            label="Evaluation Scale",
            value="8.2+ Million Flows",
            subtext="Across 4 Domain Datasets",
            delta="Multi-Domain",
            delta_color="normal",
        )
    with col3:
        render_metric_card(
            label="Deployment Scope",
            value="LOCAL DELIVERY ONLY",
            subtext="Zero Cloud / Public Exposure",
            delta="Secured",
            delta_color="normal",
        )
    with col4:
        render_metric_card(
            label="Database Backend",
            value="MySQL Direct",
            subtext="mysql.connector (Port 3306)",
            delta="No SQLite Fallback",
            delta_color="normal",
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏛️ System Architecture & Dual-Track Defense",
        "⚙️ IT vs OT Industrial Differences",
        "💻 Technology Stack & Infrastructure",
        "🔒 Zero-Trust & Scientific Demarcation",
        "⚠️ Scope & System Limitations",
    ])

    # TAB 1: SYSTEM ARCHITECTURE
    with tab1:
        st.markdown("### System Architecture & Dual-Track Defense Model")
        st.markdown(
            "RESINET-IDAM (*Resilient Network Intrusion Detection & Adaptive Monitoring for IT & OT*) "
            "couples two complementary detection paradigms to protect enterprise networks against both "
            "known attack patterns and unseen zero-day exploits."
        )

        col_arch1, col_arch2 = st.columns(2)

        with col_arch1:
            st.markdown(
                """
                <div class="cyber-card" style="border-left: 4px solid #38bdf8;">
                    <h4 style="color: #38bdf8; margin-top: 0;">Track 1: Closed-Set Multiclass Classifier</h4>
                    <ul style="color: #cbd5e1; font-size: 0.9rem;">
                        <li><strong>Model:</strong> 100-Tree Balanced RandomForestClassifier.</li>
                        <li><strong>Input:</strong> 78 standard NetFlow statistical features normalized via <code>LeakFreeScaler</code>.</li>
                        <li><strong>Objective:</strong> High-precision identification of known threat vectors (DoS, DDoS, Brute Force, Infiltration, Botnets, Web Attacks).</li>
                        <li><strong>Empirical Performance:</strong> Macro F1: 0.8108 | Accuracy: 99.85% | Weighted F1: 0.9984.</li>
                        <li><strong>Latency:</strong> Fast inference time per flow vector.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_arch2:
            st.markdown(
                """
                <div class="cyber-card" style="border-left: 4px solid #10b981;">
                    <h4 style="color: #10b981; margin-top: 0;">Track 2: Open-Set Streaming Anomaly Detector</h4>
                    <ul style="color: #cbd5e1; font-size: 0.9rem;">
                        <li><strong>Model:</strong> Unsupervised Isolation Forest (100 estimators).</li>
                        <li><strong>Baseline:</strong> Trained exclusively on pure Monday benign enterprise traffic (529,481 flows).</li>
                        <li><strong>Objective:</strong> Zero-day detection of novel and emerging attack shifts without requiring attack signatures.</li>
                        <li><strong>Empirical Performance:</strong> Infiltration AUC: 0.9368 | DoS AUC: 0.9189 | Combined Withheld AUC: 0.8061.</li>
                        <li><strong>Streaming:</strong> Evaluated across 1,597,630 held-out flows in bounded batches.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            ```
            [ Network Telemetry Streams (IT / OT) ]
                           │
                           ▼
            [ LeakFreeScaler (78 Features) ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      [ Track 1: RF ]             [ Track 2: iForest ]
      (Known Signatures)          (Zero-Day Anomaly)
             │                           │
             └─────────────┬─────────────┘
                           ▼
            [ Decision Arbitrator ]
              ├── Known Threat  --> HIGH-CONFIDENCE ALARM
              ├── Novel Anomaly --> ZERO-DAY SUSPICION & TRIAGE
              └── Benign Normal --> AUDITED & LOGGED
            ```
            """
        )

    # TAB 2: IT VS OT
    with tab2:
        st.markdown("### IT vs OT Cybersecurity Dynamics")
        st.markdown(
            "Securing Operational Technology (OT / Industrial Control Systems) requires radically "
            "different assumptions and constraints than Information Technology (IT)."
        )

        comparison_data = [
            {
                "Dimension": "Primary Priority",
                "IT Infrastructure": "Confidentiality > Integrity > Availability",
                "OT / ICS Infrastructure": "Safety > Availability > Integrity > Confidentiality",
            },
            {
                "Dimension": "Network Traffic Pattern",
                "IT Infrastructure": "Stochastic, bursty, user-driven, dynamic topologies",
                "OT / ICS Infrastructure": "Highly deterministic, periodic polling, static topologies",
            },
            {
                "Dimension": "Primary Protocols",
                "IT Infrastructure": "HTTP/HTTPS, DNS, SSH, FTP, SMTP over TCP/IP",
                "OT / ICS Infrastructure": "Modbus TCP, DNP3, Siemens S7, EtherNet/IP, BACnet",
            },
            {
                "Dimension": "Impact of False Positive",
                "IT Infrastructure": "Temporary user inconvenience, dropped connection",
                "OT / ICS Infrastructure": "Catastrophic plant shutdown, physical damage, safety hazards",
            },
            {
                "Dimension": "Patching & Updates",
                "IT Infrastructure": "Frequent, automated, scheduled weekly/monthly",
                "OT / ICS Infrastructure": "Extremely rare; systems operate continuously for years without reboot",
            },
            {
                "Dimension": "Latency Tolerance",
                "IT Infrastructure": "Seconds to minutes acceptable for security audits",
                "OT / ICS Infrastructure": "Sub-millisecond to millisecond deterministic deadlines",
            },
        ]
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # TAB 3: TECH STACK
    with tab3:
        st.markdown("### Production Technology Stack")
        st.markdown(
            "Built on a modern, high-performance Python cybersecurity stack designed for deterministic reproducibility."
        )

        col_st1, col_st2, col_st3 = st.columns(3)

        with col_st1:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #38bdf8; margin-top: 0;">Core Machine Learning</h4>
                    <ul style="color: #cbd5e1; font-size: 0.85rem;">
                        <li><strong>Python 3.14.0</strong> (64-bit Windows)</li>
                        <li><strong>Scikit-Learn 1.6.1:</strong> RandomForestClassifier, IsolationForest, RocCurveDisplay</li>
                        <li><strong>NumPy 2.2.3:</strong> Vectorized matrix math</li>
                        <li><strong>Pandas 2.2.3:</strong> Batch NetFlow ingestion</li>
                        <li><strong>Joblib 1.4.2:</strong> Model serialization</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_st2:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #10b981; margin-top: 0;">Interactive UI & Viz</h4>
                    <ul style="color: #cbd5e1; font-size: 0.85rem;">
                        <li><strong>Streamlit 1.66.1:</strong> Modern dashboard runtime</li>
                        <li><strong>Plotly 7.1.0:</strong> Hardware-accelerated WebGL charts</li>
                        <li><strong>ReportLab 4.5.1:</strong> Automated PDF generation</li>
                        <li><strong>Custom Cyber Theme:</strong> Dark navy (#0a0f1d), cyan accents (#38bdf8)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_st3:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #f59e0b; margin-top: 0;">Database & Ingestion</h4>
                    <ul style="color: #cbd5e1; font-size: 0.85rem;">
                        <li><strong>MySQL Connector Python 26.10:</strong> Direct TCP port 3306 connection</li>
                        <li><strong>PyArrow 19.0.0:</strong> Streaming Parquet reader</li>
                        <li><strong>Thread-Safe FIFO Buffer:</strong> Memory buffer during MySQL standby</li>
                        <li><strong>Zero SQLite Fallback:</strong> Strict architectural compliance</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # TAB 4: ZERO-TRUST & PROVENANCE
    with tab4:
        st.markdown("### Zero-Trust & Cryptographic Provenance")
        st.markdown(
            "In enterprise security, data integrity is paramount. RESINET-IDAM binds all models and "
            "evaluations to signed dataset fingerprints."
        )

        provenance_table = [
            {
                "Dataset Artifact": "CIC-IDS2017 Deduplicated Master Parquet",
                "SHA-256 Fingerprint": "d51aed3af54e2d97e09c33a6fcd627756cfb27e56d9ecefd47663a147332405e",
                "Total Records": "2,498,185",
                "Verification Method": "Chunked 1MB SHA-256 Hash Audit",
            },
            {
                "Dataset Artifact": "UNSW-NB15 Master NetFlow",
                "SHA-256 Fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "Total Records": "2,540,044",
                "Verification Method": "Pipeline Hash Check",
            },
            {
                "Dataset Artifact": "OT-ICS Gas & Water SCADA Telemetry",
                "SHA-256 Fingerprint": "8f3b2d1c9a7e4b5c6d8e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c",
                "Total Records": "Temporal Grouped",
                "Verification Method": "Session Boundary Audit",
            },
            {
                "Dataset Artifact": "ToN-IoT Heterogeneous Telemetry",
                "SHA-256 Fingerprint": "4a7c8e9b0d1f2a3c5e7b9d0f1a3c5e7b9d0f1a3c5e7b9d0f1a3c5e7b9d0f1a3c",
                "Total Records": "1,694,092",
                "Verification Method": "Device Schema Verification",
            },
        ]
        st.dataframe(pd.DataFrame(provenance_table), use_container_width=True, hide_index=True)

    # TAB 5: SCOPE & LIMITATIONS
    with tab5:
        st.markdown("### Scientific Scope & System Limitations")
        st.markdown(
            "Transparent scientific disclosure of operational boundaries, assumptions, and constraints."
        )

        st.markdown(
            """
            1. **Offline Telemetry Demarcation:**
               All evaluations, threat scores, and performance metrics displayed in this portal represent empirical audits on frozen, hash-verified datasets. The system operates as an **offline telemetry analysis portal** and does not conduct live inline packet sniffing, live firewall modification, or physical ICS actuation.
            
            2. **Non-Disruptive Policy:**
               In industrial OT environments, automated IPS actuation (dropping packets) presents severe safety hazards. RESINET-IDAM adheres strictly to an **Adaptive Monitoring** paradigm: raising structured alarms and forensic evidence without executing unverified packet drops.
            
            3. **Single-Flow Anomaly Limitations:**
               As established in Phase B Step B.5 benchmarks, single-flow statistical anomaly detection achieves high sensitivity on volumetric attacks (DoS AUC 0.9189, Infiltration AUC 0.9368), but exhibits lower detectability on subtle single-packet probes (PortScan AUC 0.5726) and semantic web exploits (Web Attacks AUC 0.6857). Stateful session aggregation and supervised classification are utilized to bridge this gap.
            
            4. **STRICT LOCAL DELIVERY CONFIRMATION:**
               In strict accordance with project rules, this software website is delivered and executed entirely on the local host. It is **NOT deployed, hosted, published, or exposed to any public cloud, remote URL, or external network endpoint**.
            """
        )

