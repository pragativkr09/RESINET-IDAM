"""
RESINET-IDAM: Explainable AI Page (src/ui/pages/p7_explainable_ai.py)
Feature importance rankings, domain semantics, ensemble decision logic, and XAI methodology.
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
from src.ui.utils import load_benchmark_data


FEATURE_SEMANTICS = {
    "Destination Port": {
        "domain": "Transport Layer Targeting",
        "description": "Identifies the network service being addressed (e.g., Port 21 FTP, Port 22 SSH, Port 80 HTTP, Port 443 HTTPS).",
        "why_attacks_trigger": "Brute-force attacks specifically target administrative ports (FTP/SSH), while Web Attacks and Heartbleed target web servers. Permutation importance of 0.1459 confirms it is the single most vital discriminator in the model.",
        "layer": "Layer 4 (Transport)",
    },
    "Init_Win_bytes_backward": {
        "domain": "TCP Stack Fingerprinting",
        "description": "Initial TCP window size (in bytes) advertised by the responding host in SYN-ACK packets.",
        "why_attacks_trigger": "Different operating systems (Linux, Windows, BSD) implement distinct TCP stack buffer sizes. Under Slowloris or DoS exhaustion, response buffer sizes shrink to zero, exposing server distress.",
        "layer": "Layer 4 (TCP Handshake)",
    },
    "Bwd Header Length": {
        "domain": "Protocol Framing Mechanics",
        "description": "Cumulative byte length of all backward TCP headers in the bidirectional flow.",
        "why_attacks_trigger": "Reveals TCP option negotiations (MSS, Window Scaling, Selective ACK) and highlights anomalous header-to-payload ratios typical of automated scanners.",
        "layer": "Layer 4 (Framing)",
    },
    "Bwd Packets/s": {
        "domain": "Response Flow Dynamics",
        "description": "Rate of packets returned from server to client per second.",
        "why_attacks_trigger": "During DoS/DDoS attacks, backward packet rate drops towards zero as target servers freeze. In contrast, benign file downloads exhibit high sustained backward packet rates.",
        "layer": "Flow Dynamics",
    },
    "Avg Bwd Segment Size": {
        "domain": "Response Payload Characterization",
        "description": "Average size of segments transmitted in the backward direction.",
        "why_attacks_trigger": "Heartbleed memory leaks return massive chunks of unallocated memory (up to 64KB), inflating this feature far above typical TLS responses.",
        "layer": "Payload / Transport",
    },
    "Bwd Packet Length Max": {
        "domain": "Maximum Ingress Payload",
        "description": "The largest individual packet returned from the destination host in the session.",
        "why_attacks_trigger": "Distinguishes normal HTTP 200 responses from truncated error codes (401 Unauthorized during brute force) or oversized exfiltration dumps.",
        "layer": "Flow Characterization",
    },
    "min_seg_size_forward": {
        "domain": "Client Framing Configuration",
        "description": "Minimum observed TCP segment size sent by the client.",
        "why_attacks_trigger": "Attack scripts (LOIC, Patator) often omit TCP options or use custom raw socket implementations with static 20-byte or 32-byte header sizes, exposing tool-specific fingerprints.",
        "layer": "Client Signature",
    },
    "Fwd Packet Length Max": {
        "domain": "Forward Injection Vector",
        "description": "Maximum packet size sent by the attacking/requesting host.",
        "why_attacks_trigger": "Buffer overflow exploits and SQL injection payloads inject oversized strings into HTTP POST parameters, creating distinct spikes in maximum forward packet size.",
        "layer": "Attack Delivery",
    },
    "Max Packet Length": {
        "domain": "Global Session Extrema",
        "description": "The maximum packet size observed across the entire bidirectional flow.",
        "why_attacks_trigger": "Captures MTU boundary interactions and identifies anomalous jumbo frames or fragmentation anomalies.",
        "layer": "Session Metrics",
    },
    "Total Length of Fwd Packets": {
        "domain": "Attack Volume & Payload Size",
        "description": "Cumulative bytes transmitted by the initiator across the entire session duration.",
        "why_attacks_trigger": "Volumetric DoS floods accumulate hundreds of megabytes rapidly, whereas stealthy recon (PortScan) produces negligible forward byte totals.",
        "layer": "Volumetric Analysis",
    },
    "Fwd IAT Mean": {
        "domain": "Transmission Cadence",
        "description": "Mean inter-arrival time between consecutive packets sent by the client.",
        "why_attacks_trigger": "Automated scripts blast packets with sub-millisecond intervals (mean ~0), whereas human browsing exhibits variable pacing with multi-second pauses.",
        "layer": "Timing Dynamics",
    },
    "Flow IAT Mean": {
        "domain": "Bidirectional Cadence",
        "description": "Mean inter-arrival time across all packets in the bidirectional flow.",
        "why_attacks_trigger": "Machine-speed floods show uniform microsecond distributions; slow-and-low attacks (Slowloris) intentionally maximize IAT to keep connections alive.",
        "layer": "Timing Dynamics",
    },
    "Fwd Header Length": {
        "domain": "Client Transport Framing",
        "description": "Cumulative forward TCP header length.",
        "why_attacks_trigger": "Reflects packet volume and header integrity checks across client requests.",
        "layer": "Layer 4 (Framing)",
    },
    "Flow IAT Std": {
        "domain": "Cadence Jitter & Variance",
        "description": "Standard deviation of packet inter-arrival times across the session.",
        "why_attacks_trigger": "Botnet C2 communications and automated floods have near-zero jitter (highly deterministic clocking), while human traffic has high natural variance.",
        "layer": "Statistical Profile",
    },
    "Flow Duration": {
        "domain": "Session Lifespan",
        "description": "Total duration of the network flow from first SYN/packet to FIN/RST or timeout (in microseconds).",
        "why_attacks_trigger": "DDoS and PortScans last fractions of a second; Slowloris and GoldenEye deliberately stretch flow duration across minutes to lock server threads.",
        "layer": "Session Lifespan",
    },
}


def render_page():
    render_page_header(
        title="Explainable AI & Feature Semantics",
        subtitle="Unraveling decision tree splits, Mean Decrease in Impurity, and cyber physical semantics",
        badge="TRANSPARENT AI",
    )
    render_offline_demarcation_banner()

    data = load_benchmark_data()
    track1 = data.get("track_1_closed_set_multiclass", {})
    top_features = track1.get("top_features", [])

    if not top_features:
        st.error("Top feature importances not found in benchmark data.")
        return

    df_feats = pd.DataFrame(top_features)

    # Top Metric Ribbon
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            label="Analyzed Features",
            value="78 Features",
            subtext="Complete NetFlow Vector",
            delta="Zero Missing",
        )
    with col2:
        render_metric_card(
            label="Top Discriminator",
            value="Destination Port",
            subtext="MDI: 0.0550 | Perm: 0.1459",
            delta="Rank 1",
            delta_color="normal",
        )
    with col3:
        render_metric_card(
            label="Ensemble Architecture",
            value="100 Estimators",
            subtext="Gini Impurity Criterion",
            delta="Deterministic Seed 42",
        )
    with col4:
        render_metric_card(
            label="Interpretability",
            value="MDI + Permutation",
            subtext="Empirical Validation",
            delta="Line-Rate Capable",
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top 15 Feature Importances",
        "🔬 Cyber Domain Semantic Deep Dive",
        "🌲 Tree Ensemble Decision Logic",
        "📐 XAI Scientific Methodology",
    ])

    # TAB 1: TOP 15 IMPORTANCES
    with tab1:
        st.markdown("### Top 15 Most Discriminative Flow Features")
        st.markdown(
            "Ranked by Mean Decrease in Impurity (MDI / Gini Importance) and validated against "
            "independent Permutation Importance on the held-out validation set."
        )

        metric_choice = st.radio(
            "Select Importance Metric for Visualization:",
            options=["MDI (Gini) Importance", "Permutation Importance (Mean)"],
            horizontal=True,
        )

        val_col = "mdi_importance" if "MDI" in metric_choice else "permutation_importance_mean"
        title_metric = "MDI (Gini) Importance" if "MDI" in metric_choice else "Permutation Importance (Mean F1 Degradation)"

        fig_imp = px.bar(
            df_feats.sort_values(by=val_col, ascending=True),
            x=val_col,
            y="feature",
            orientation="h",
            color=val_col,
            color_continuous_scale="Tealgrn" if "MDI" in metric_choice else "Blues",
            text=val_col,
            title=f"Top 15 Features by {title_metric}",
        )
        fig_imp.update_layout(height=520)
        fig_imp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_imp = apply_cyber_theme_to_plotly(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Tabular Feature Importance Data")
        st.dataframe(
            df_feats.style.format({
                "mdi_importance": "{:.6f}",
                "permutation_importance_mean": "{:.6f}",
                "permutation_importance_std": "{:.6f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # TAB 2: CYBER DOMAIN SEMANTICS
    with tab2:
        st.markdown("### Cyber Domain Semantic Interpretability")
        st.markdown(
            "Machine learning models in cybersecurity must not be black boxes. "
            "Below is the exact networking and protocol rationale explaining why each top feature drives detection."
        )

        selected_feat = st.selectbox(
            "Select a Feature to Inspect Domain Semantics:",
            options=[f["feature"] for f in top_features],
        )

        sem = FEATURE_SEMANTICS.get(selected_feat, {
            "domain": "Network Telemetry Metric",
            "description": "Standard NetFlow bidirectional statistic.",
            "why_attacks_trigger": "Diverges under non-standard traffic loads.",
            "layer": "Transport / Flow",
        })

        feat_row = df_feats[df_feats["feature"] == selected_feat].iloc[0]

        st.markdown(
            f"""
            <div class="cyber-card" style="border-left: 4px solid #38bdf8;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="color: #38bdf8; margin: 0;">{selected_feat}</h3>
                    <span class="badge-low">Rank #{feat_row['rank']}</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                    <strong>OSI / Functional Layer:</strong> {sem['layer']} | <strong>Domain Category:</strong> {sem['domain']}
                </p>
                <div style="margin: 16px 0;">
                    <strong style="color: #f8fafc;">Feature Definition:</strong>
                    <p style="color: #cbd5e1; margin-top: 4px;">{sem['description']}</p>
                </div>
                <div style="margin: 16px 0;">
                    <strong style="color: #38bdf8;">Cyberattack Mechanics & Detection Rationale:</strong>
                    <p style="color: #cbd5e1; margin-top: 4px;">{sem['why_attacks_trigger']}</p>
                </div>
                <div style="display: flex; gap: 20px; background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                    <div>MDI Importance: <span style="color: #38bdf8;">{feat_row['mdi_importance']:.6f}</span></div>
                    <div>Permutation Mean: <span style="color: #10b981;">{feat_row['permutation_importance_mean']:.6f}</span></div>
                    <div>Permutation Std: <span style="color: #94a3b8;">{feat_row['permutation_importance_std']:.6f}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TAB 3: TREE ENSEMBLE DECISION LOGIC
    with tab3:
        st.markdown("### Tree Ensemble Decision Logic")
        st.markdown(
            "How the 100-tree RandomForestClassifier computes class consensus without black-box opacity."
        )

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #38bdf8; margin-top: 0;">1. Gini Impurity Partitioning</h4>
                    <p style="color: #cbd5e1; font-size: 0.9rem;">
                        At each decision node $t$, the tree searches across all 78 candidate features to find the split threshold $\theta$ that maximizes the reduction in Gini Impurity:
                    </p>
                    <div style="background: rgba(15, 23, 42, 0.8); padding: 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38bdf8;">
                        ΔI_G = I_G(t) - [ (N_L / N) * I_G(t_L) + (N_R / N) * I_G(t_R) ]
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        Where $I_G(t) = 1 - \\sum_{k=1}^{15} p_k^2$. Features like <code>Destination Port</code> produce immediate, sharp drops in impurity because port numbers cleanly isolate services.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_t2:
            st.markdown(
                """
                <div class="cyber-card">
                    <h4 style="color: #10b981; margin-top: 0;">2. Balanced Voting Consensus</h4>
                    <p style="color: #cbd5e1; font-size: 0.9rem;">
                        Each of the 100 trees in the forest is trained on an independent bootstrap sample with <code>class_weight='balanced_subsample'</code>.
                        The forest combines all tree posterior probabilities into an ensemble consensus:
                    </p>
                    <div style="background: rgba(15, 23, 42, 0.8); padding: 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #10b981;">
                        P(y = c | x) = (1 / B) * Σ_{b=1}^{100} P_b(y = c | x)
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                        If the top posterior probability satisfies $P(y = c) \\ge 0.50$, the flow is assigned to that specific intrusion category; otherwise, flagged for secondary triage.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # TAB 4: XAI SCIENTIFIC METHODOLOGY
    with tab4:
        st.markdown("### XAI Scientific Methodology & Line-Rate Tradeoffs")
        st.markdown(
            "Evaluation of explainability methods for high-throughput enterprise network monitoring."
        )

        methods_table = [
            {
                "Method": "Mean Decrease in Impurity (MDI)",
                "Computational Cost": "Zero during Inference (Precomputed at Fit)",
                "Line-Rate Suitability": "★★★★★ (Optimal for real-time deployment)",
                "Scientific Limitation": "Slightly biased toward continuous features with high cardinality.",
            },
            {
                "Method": "Permutation Importance",
                "Computational Cost": "Moderate (Requires shuffling features on validation set)",
                "Line-Rate Suitability": "★★★☆☆ (Used during validation benchmarks)",
                "Scientific Limitation": "Breaks feature correlations; if features are collinear, both may look low.",
            },
            {
                "Method": "SHAP (Kernel / TreeSHAP)",
                "Computational Cost": "Extreme ($O(M \\cdot L \\cdot D^2)$ per flow vector)",
                "Line-Rate Suitability": "★☆☆☆☆ (Causes packet buffer overflow in production)",
                "Scientific Limitation": "Too slow for 10 Gbps packet arrival rates; reserved for offline post-incident forensics.",
            },
        ]
        df_methods = pd.DataFrame(methods_table)
        st.dataframe(df_methods, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Production Architecture Choice:** RESINET-IDAM utilizes **MDI feature importance pre-indexing combined with Permutation-tested discriminators** for real-time telemetry scoring, ensuring sub-millisecond per-flow inference without stalling the ingestion queue."
        )

