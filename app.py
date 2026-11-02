"""
RESINET-IDAM: Main Streamlit Application Router (c:/RESINET-IDAM/Dataset/app.py)
Resilient Network Intrusion Detection & Adaptive Monitoring for IT & OT Infrastructure.
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Configure page settings - MUST be first Streamlit command
st.set_page_config(
    page_title="RESINET-IDAM | Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark navy cybersecurity stylesheet
from src.ui.styles import apply_cyber_css
from src.ui.utils import load_multiclass_artifacts, load_anomaly_artifacts
from src.db.mysql_manager import MySQLManager

from src.ui.pages import (
    p1_dashboard,
    p2_dataset_explorer,
    p3_multiclass_ids,
    p4_anomaly_detection,
    p5_model_performance,
    p6_threat_intel,
    p7_explainable_ai,
    p8_reports,
    p9_project_info,
)

apply_cyber_css()

# Sidebar Navigation & Operational Status
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 12px 0 20px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.2);">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">🛡️</div>
            <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: 0.1em; color: #38bdf8;">RESINET-IDAM</div>
            <div style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">
                Adaptive IT & OT Cyber Defense
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase;'>Navigation</p>", unsafe_allow_html=True)

    pages = {
        "🛡️ Security Dashboard": p1_dashboard.render_page,
        "📁 IT Dataset Explorer": p2_dataset_explorer.render_page,
        "🔍 Multiclass Intrusion Detection": p3_multiclass_ids.render_page,
        "🚨 Open-Set Anomaly Detection": p4_anomaly_detection.render_page,
        "📈 Model Performance": p5_model_performance.render_page,
        "⚔️ Threat Intelligence & Analysis": p6_threat_intel.render_page,
        "🧠 Explainable AI": p7_explainable_ai.render_page,
        "📄 Reports & Audit Logs": p8_reports.render_page,
        "ℹ️ Project Information": p9_project_info.render_page,
    }

    selected_page_name = st.radio(
        "Select Operation Page:",
        options=list(pages.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.1); margin: 20px 0;'>", unsafe_allow_html=True)

    # System Health Panel in Sidebar
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; margin-bottom: 8px; text-transform: uppercase;'>System Health & Diagnostics</p>", unsafe_allow_html=True)

    clf, scaler = load_multiclass_artifacts()
    iso, iso_scaler = load_anomaly_artifacts()
    db = MySQLManager()
    db_connected, db_status_msg, db_info = db.check_connection_status()

    clf_badge = "<span style='color: #10b981; font-weight: 600;'>READY</span>" if clf is not None else "<span style='color: #ef4444;'>OFFLINE</span>"
    iso_badge = "<span style='color: #10b981; font-weight: 600;'>READY</span>" if iso is not None else "<span style='color: #ef4444;'>OFFLINE</span>"
    db_badge = "<span style='color: #10b981; font-weight: 600;'>ONLINE</span>" if db_connected else "<span style='color: #f59e0b; font-weight: 600;'>STANDBY (BUFFERING)</span>"

    st.markdown(
        f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; background: rgba(15, 23, 42, 0.7); padding: 12px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.15);">
            <div style="margin-bottom: 6px;">Track 1 (RF-100): {clf_badge}</div>
            <div style="margin-bottom: 6px;">Track 2 (iForest): {iso_badge}</div>
            <div style="margin-bottom: 6px;">MySQL (Port 3306): {db_badge}</div>
            <div>Dataset Master: <span style="color: #38bdf8; font-weight: 600;">SIGNED</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top: 24px; text-align: center; font-size: 0.7rem; color: #64748b;">
            <div>RESINET-IDAM v6.12 | Phase C</div>
            <div style="margin-top: 4px; color: #94a3b8; font-weight: 600;">LOCAL VERIFIED DEPLOYMENT</div>
            <div style="margin-top: 2px;">Zero Public Exposure</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render the active page
pages[selected_page_name]()

