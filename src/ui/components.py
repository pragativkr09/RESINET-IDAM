"""
RESINET-IDAM: Reusable UI Components (src/ui/components.py)
Provides consistent styled cards, banners, headers, and indicators.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any


def render_page_header(title: str, subtitle: str, badge: Optional[str] = None):
    """Renders standard top header with cybersecurity branding."""
    badge_html = f'<span class="badge-low" style="margin-left: 12px; vertical-align: middle;">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <h1 style="display: inline-block; margin-bottom: 4px;">{title}</h1>{badge_html}
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 2px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    delta: Optional[str] = None,
    delta_color: str = "normal",
):
    """Renders a styled dark navy metric box."""
    delta_html = ""
    if delta:
        color = "#10b981" if delta_color == "normal" else "#ef4444"
        delta_html = f'<span style="color: {color}; font-size: 0.8rem; font-weight: 600; margin-left: 8px;">{delta}</span>'

    st.markdown(
        f"""
        <div class="cyber-card">
            <div class="metric-header">{label}</div>
            <div class="metric-value">{value} {delta_html}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_threat_badge(threat_level: str) -> str:
    """Returns HTML for a standardized threat level badge."""
    level = threat_level.upper()
    if level == "CRITICAL":
        return '<span class="badge-critical">CRITICAL</span>'
    elif level == "HIGH":
        return '<span class="badge-high">HIGH</span>'
    elif level == "MEDIUM":
        return '<span class="badge-medium">MEDIUM</span>'
    elif level == "LOW":
        return '<span class="badge-low">LOW</span>'
    else:
        return '<span class="badge-benign">BENIGN</span>'


def render_offline_demarcation_banner(context: str = "Analysis"):
    """Displays explicit banner ensuring users distinguish offline telemetry from live network feeds."""
    st.markdown(
        f"""
        <div class="offline-banner">
            <span style="font-size: 1.1rem; margin-right: 10px;">🔒</span>
            <div>
                <strong>OFFLINE TELEMETRY BENCHMARK:</strong>
                All displayed figures, telemetry profiles, and model scores represent rigorous empirical evaluation on frozen, hash-verified datasets.
                Live packet sniffing and real-time network actuation are strictly decoupled by design.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_cyber_theme_to_plotly(fig: go.Figure) -> go.Figure:
    """Applies consistent dark navy cybersecurity theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            gridcolor="rgba(56, 189, 248, 0.1)",
            zerolinecolor="rgba(56, 189, 248, 0.2)",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(56, 189, 248, 0.1)",
            zerolinecolor="rgba(56, 189, 248, 0.2)",
            showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(56, 189, 248, 0.2)",
            borderwidth=1,
        ),
    )
    return fig

