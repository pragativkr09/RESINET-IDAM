"""
RESINET-IDAM: Modern Cybersecurity UI Theme & Custom CSS (src/ui/styles.py)
Provides a premium dark navy cybersecurity design with cyan/blue accents.
"""

CYBER_CSS = """
<style>
/* Main Background and Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, #0d1527 0%, #080d1a 100%);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0c1322 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-weight: 500;
}

/* Custom Card Container */
.cyber-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.cyber-card:hover {
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 6px 24px rgba(14, 165, 233, 0.15);
}

.metric-header {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    line-height: 1.2;
}

.metric-subtext {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 6px;
}

/* Badges */
.badge-critical {
    background-color: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.5);
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    display: inline-block;
}

.badge-high {
    background-color: rgba(249, 115, 22, 0.2);
    color: #f97316;
    border: 1px solid rgba(249, 115, 22, 0.5);
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.75rem;
    display: inline-block;
}

.badge-medium {
    background-color: rgba(234, 179, 8, 0.2);
    color: #eab308;
    border: 1px solid rgba(234, 179, 8, 0.5);
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.75rem;
    display: inline-block;
}

.badge-low {
    background-color: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.5);
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.75rem;
    display: inline-block;
}

.badge-benign {
    background-color: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.5);
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.75rem;
    display: inline-block;
}

/* Offline Demarcation Banner */
.offline-banner {
    background: linear-gradient(90deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border-left: 4px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin-bottom: 22px;
    font-size: 0.85rem;
    color: #cbd5e1;
    display: flex;
    align-items: center;
}

/* Headers */
h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

h1 {
    font-size: 2.1rem !important;
    letter-spacing: -0.02em;
}

h2 {
    font-size: 1.45rem !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    padding-bottom: 8px;
    margin-top: 24px !important;
}

/* Button Customization */
div.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.25rem !important;
    transition: all 0.2s ease-in-out !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.5) !important;
}

/* Tables & Dataframes */
div[data-testid="stDataFrame"] {
    background-color: #0f172a;
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 8px;
}
</style>
"""

import streamlit as st


def apply_cyber_css():
    """
    Injects custom dark navy & cyan cybersecurity CSS into the Streamlit app.
    Preserves modern typography (Inter, JetBrains Mono), neon cyan accents (#38bdf8),
    and dark navy container styling (#0a0f1d, #0d1527).
    """
    st.markdown(CYBER_CSS, unsafe_allow_html=True)


def get_cyber_css() -> str:
    """Returns raw cyber theme CSS string."""
    return CYBER_CSS


