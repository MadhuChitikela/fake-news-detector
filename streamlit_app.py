import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from classifier import classify_text
from fact_checker import fact_check_article
from explainer import get_highlighted_sentences
from database import init_db, save_check, get_recent, get_stats, clear_history

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Guardian AI | Fake News Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clean & High-Contrast UI Theme ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Bricolage+Grotesque:wght@500;700;800&display=swap');

/* Base Styles */
* { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; }
p, span, label, div { color: #1e293b; }

.stApp {
    background: #f8fafc;
    color: #0f172a;
}

/* Sidebar - Crisp Solid Contrast */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] h2 {
    color: #1d4ed8 !important;
}
section[data-testid="stSidebar"] h4 {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* Metric Cards - High Contrast */
.metric-container {
    padding: 1.25rem 1rem;
    border-radius: 12px;
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    text-align: center;
    margin-bottom: 0.85rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1d4ed8;
    margin-bottom: 0.25rem;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
}

/* Analysis Card */
.analysis-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Verdict Badges - Strong Readable Colors */
.badge {
    padding: 8px 20px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 14px;
    display: inline-block;
    letter-spacing: 0.02em;
}
.badge-real {
    background: #dcfce7 !important;
    color: #14532d !important;
    border: 1.5px solid #86efac !important;
}
.badge-fake {
    background: #fee2e2 !important;
    color: #7f1d1d !important;
    border: 1.5px solid #fca5a5 !important;
}
.badge-uncertain {
    background: #fef3c7 !important;
    color: #78350f !important;
    border: 1.5px solid #fcd34d !important;
}

/* Highlighted Sentences - Distinct Backgrounds with Dark Readable Text */
.sent-box {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 5px solid #94a3b8;
    background: #f1f5f9;
    font-weight: 500;
    font-size: 0.95rem;
    line-height: 1.5;
}
.sent-high {
    background: #ffe4e6 !important;
    border-left: 5px solid #e11d48 !important;
    color: #881337 !important;
}
.sent-medium {
    background: #fef3c7 !important;
    border-left: 5px solid #d97706 !important;
    color: #78350f !important;
}
.sent-low {
    background: #dcfce7 !important;
    border-left: 5px solid #16a34a !important;
    color: #14532d !important;
}

/* Custom Inputs & Buttons */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 6px 12px -1px rgba(37, 99, 235, 0.35) !important;
}

/* Sidebar Action Button */
section[data-testid="stSidebar"] .stButton > button {
    background: #f8fafc !important;
    color: #dc2626 !important;
    border: 1.5px solid #fecaca !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #fee2e2 !important;
    color: #b91c1c !important;
    border-color: #fca5a5 !important;
}

/* Tabs Styling - Bold & High Contrast */
button[data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1d4ed8 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #1d4ed8 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Footer & Dividers */
hr {
    border-color: #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}
.footer {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 4rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Init DB ──────────────────────────────────────────────────────
init_db()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#1d4ed8; margin-bottom:1.5rem;'>🛡️ Guardian Intelligence</h2>", unsafe_allow_html=True)
    
    stats = get_stats()
    
    # Custom Metric Cards in Sidebar
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{stats['total']}</div>
        <div class="metric-label">Articles Scanned</div>
    </div>
    <div class="metric-container">
        <div class="metric-value" style="color:#15803d;">{stats['real']}</div>
        <div class="metric-label">Verified Real</div>
    </div>
    <div class="metric-container">
        <div class="metric-value" style="color:#dc2626;">{stats['fake']}</div>
        <div class="metric-label">Caught Fake</div>
    </div>
    <div class="metric-container">
        <div class="metric-value" style="color:#b45309;">{stats['avg_trust']}%</div>
        <div class="metric-label">Intelligence Trust</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑️  Clear All History", use_container_width=True):
        clear_history()
        st.toast("History cleared successfully!", icon="🗑️")
        st.rerun()

    st.markdown("<h4 style='color:#0f172a; margin-top:1.5rem;'>🕐 Recent History</h4>", unsafe_allow_html=True)
    recent = get_recent(5)
    for row in recent:
        id_, ts, label, trust, verdict = row
        color = "#15803d" if label == "REAL" else "#dc2626"
        st.markdown(f"<p style='font-size:0.88rem; margin-bottom:6px; color:#1e293b;'><b>#{id_}</b> | <span style='color:{color}; font-weight:700;'>{label}</span> — <span style='font-weight:600;'>{trust}%</span></p>", unsafe_allow_html=True)

# ── Main Header ──────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 2.5rem">
    <div style="font-size:0.8rem; font-weight:800; text-transform:uppercase; color:#2563eb; letter-spacing:0.08em; margin-bottom:0.5rem">
        Intelligence Dashboard
    </div>
    <h1 style="font-size:2.8rem; margin-top:0; color:#0f172a;">
        News <span style="background: linear-gradient(135deg, #1d4ed8, #0284c7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Guardian.</span>
    </h1>
    <p style="font-size:1.05rem; color:#334155; max-width:650px; line-height:1.6;">
        Advanced fake news classification using BERT fine-tuned on GPU, multi-LLM fact verification, and 
        real-time web search integration.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Investigation Engine", "📊 Global Analytics", "💡 Simulation Cases"])

# ════════════════════════════════════════
# TAB 1 — INVESTIGATION
# ════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📄 Paste Article to Begin")

        # Quick Test Samples
        if "article_input" not in st.session_state:
            st.session_state.article_input = ""

        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#475569; margin-bottom:6px;'>⚡ Quick Test Notes / Samples:</p>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        if q1.button("🟢 Real Sample", use_container_width=True):
            st.session_state.article_input = "NASA's James Webb Space Telescope has captured deep-space imagery confirming the presence of organic carbon compounds and amino acid precursors in an interstellar molecular cloud."
        if q2.button("🔴 Fake Sample", use_container_width=True):
            st.session_state.article_input = "BREAKING: Secret government documents reveal scientists have replaced the drinking water in major cities with caffeinated energy drinks to boost factory productivity."
        if q3.button("🟡 Mixed Claim", use_container_width=True):
            st.session_state.article_input = "Global tech leaders are rumored to be finalizing a secret consortium to replace all banking networks with quantum computing protocols by next month."

        article = st.text_area(
            label="Article Input",
            value=st.session_state.article_input,
            placeholder="Enter the news headline or full text here for a deep-dive analysis...",
            height=250,
            label_visibility="collapsed"
        )
        analyze_btn = st.button("🚀 START INVESTIGATION", use_container_width=True)

    with col2:
        if analyze_btn and article.strip():
            with st.spinner("🧠 BERT Classifier scanning sequences..."):
                bert_result = classify_text(article)

            with st.spinner("🌐 Searching live web and fact-checking..."):
                fact_result = fact_check_article(article)

            sentences = get_highlighted_sentences(article, bert_result["fake_prob"] / 100)

            # Save to Audit Log
            save_check(article, bert_result["label"], bert_result["trust_score"], 
                       fact_result["overall_verdict"], fact_result["supported_count"], 
                       fact_result["verified_count"])

            st.markdown("### 📊 Investigation Report")
            
            # Gauge & Verdict Layout
            trust = bert_result["trust_score"]
            label = bert_result["label"]
            
            # Trust Dial
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=trust,
                number={"suffix": "%", "font": {"color": "#1d4ed8", "size": 36, "family": "Bricolage Grotesque"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1.5, "tickcolor": "#64748b"},
                    "bar": {"color": "#1d4ed8", "thickness": 0.8},
                    "bgcolor": "#f1f5f9",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239, 68, 68, 0.2)"},
                        {"range": [40, 75], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [75, 100], "color": "rgba(16, 185, 129, 0.2)"}
                    ],
                }
            ))
            fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # High-level Verdicts
            v1, v2 = st.columns(2)
            with v1:
                b_class = "badge-real" if label == "REAL" else "badge-fake"
                st.markdown(f"<div style='text-align:center;'><b style='color:#0f172a;'>BERT Classification</b><br><span class='badge {b_class}' style='margin-top:8px;'>{label}</span></div>", unsafe_allow_html=True)
            with v2:
                ov = fact_result['overall_verdict']
                f_class = "badge-real" if "REAL" in ov else "badge-fake" if "FAKE" in ov else "badge-uncertain"
                st.markdown(f"<div style='text-align:center;'><b style='color:#0f172a;'>Web Fact Check</b><br><span class='badge {f_class}' style='margin-top:8px;'>{ov}</span></div>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Claim Details
            st.markdown("#### 🔍 Claim Verification Matrix")
            for claim in fact_result["claims"]:
                verdict = claim["verdict"]
                icon = "🟢" if "SUP" in verdict else "🔴" if "CON" in verdict else "🟡"
                with st.expander(f"{icon} {claim['claim'][:60]}..."):
                    st.markdown(f"**Verdict:** `{claim['verdict']}`")
                    st.markdown(f"<p style='color:#1e293b; font-size:0.95rem; line-height:1.5;'>{claim['explanation']}</p>", unsafe_allow_html=True)

            # Sentence Logic
            st.markdown("#### 🎨 Contextual Risk Heatmap")
            for s in sentences:
                st.markdown(f"<div class='sent-box sent-{s['risk']}'>{s['sentence']} <span style='float:right; font-weight:700; font-size:0.8rem;'>{s['fake_prob']}% Risk</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:360px; display:flex; flex-direction:column; justify-content:center; align-items:center; background:#ffffff; border:2px dashed #cbd5e1; border-radius:16px;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🔍</div>
                <div style="font-weight:700; color:#0f172a; font-size:1.1rem;">System ready for input</div>
                <div style="font-size:0.9rem; color:#475569; margin-top:0.25rem;">Enter article text on the left to generate the investigation report</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Global Threat Intelligence")
    
    col_a, col_b = st.columns([1, 1], gap="large")
    
    with col_a:
        # Pie Chart
        fig_pie = px.pie(
            values=[stats["real"], stats["fake"]],
            names=["Verified Real", "Caught Fake"],
            color_discrete_sequence=["#16a34a", "#dc2626"],
            hole=0.6,
            title="Classification Distribution"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=True, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        st.markdown("#### 📜 Audit Log History")
        df_recent = pd.DataFrame(get_recent(15), columns=["ID", "Timestamp", "Label", "Trust", "Verdict"])
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 3 — SIMULATION
# ════════════════════════════════════════
with tab3:
    st.markdown("### 🧪 Intelligence Scenarios")
    
    cases = [
        {"title": "🟢 Medical Research (Verified)", "content": "NASA's James Webb Telescope finds amino acids in deep stellar nursery, suggesting chemical blocks for life exist everywhere in the universe."},
        {"title": "🔴 Political Misinformation (Fake)", "content": "Government secret leaked: Officials caught replacing all city water with energy drinks to increase worker productivity."},
        {"title": "🟡 Market Uncertainty (Mixed)", "content": "Reports suggest a tech giant might buy a small startup, but no official confirmation or SEC filings have been detected yet."}
    ]
    
    for c in cases:
        with st.expander(c["title"]):
            st.code(c["content"])
            st.caption("Copy this to Investigation Engine 👆")

st.markdown("<div class='footer'>AI News Guardian Engine v1.0 // Developed with BERT, LangChain & Streamlit</div>", unsafe_allow_html=True)
