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

# ── Clean & Glassy UI Theme (Front-End Developer Style) ──────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Bricolage+Grotesque:wght@400;600;700;800&display=swap');

/* Base Styles */
* { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; letter-spacing: -0.02em; }

.stApp {
    background: linear-gradient(135deg, #f8faff 0%, #ffffff 100%);
    color: #1a1c23;
}

/* Glassmorphism Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(0, 122, 255, 0.1);
}

/* Metric Cards - Modern Blue Style */
.metric-container {
    padding: 1.5rem;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(0, 122, 255, 0.08);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.02);
    text-align: center;
    margin-bottom: 1rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #007aff;
    margin-bottom: 0.2rem;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
}

/* Analysis Card */
.analysis-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05);
}

/* Verdict Badges */
.badge {
    padding: 8px 16px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 14px;
    display: inline-block;
}
.badge-real { background: #ecfdf5; color: #059669; border: 1px solid #10b981; }
.badge-fake { background: #fef2f2; color: #dc2626; border: 1px solid #ef4444; }
.badge-uncertain { background: #fffbeb; color: #d97706; border: 1px solid #f59e0b; }

/* Highlighted Sentences */
.sent-box {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    border-left: 4px solid #cbd5e1;
    background: #f8fafc;
    transition: all 0.2s ease;
}
.sent-high   { background: #fff1f2; border-left: 4px solid #f43f5e; color: #9f1239; }
.sent-medium { background: #fffcf0; border-left: 4px solid #f59e0b; color: #92400e; }
.sent-low    { background: #f0fdf4; border-left: 4px solid #10b981; color: #166534; }

/* Custom Inputs & Buttons */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    color: #1e293b !important;
}
.stTextArea textarea:focus {
    border-color: #007aff !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
}

.stButton > button {
    background: #007aff !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(0, 122, 255, 0.3) !important;
    transition: transform 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    background: #006ce0 !important;
}

/* Footer & Dividers */
hr { border-color: rgba(0, 122, 255, 0.05); }
.footer { color: #94a3b8; font-size: 0.8rem; margin-top: 4rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Init DB ──────────────────────────────────────────────────────
init_db()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#007aff; margin-bottom:1.5rem;'>🛡️ Guardian Intelligence</h2>", unsafe_allow_html=True)
    
    stats = get_stats()
    
    # Custom Metric Cards in Sidebar
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{stats['total']}</div>
        <div class="metric-label">Articles Scanned</div>
    </div>
    <div class="metric-container">
        <div class="metric-value" style="color:#10b981">{stats['real']}</div>
        <div class="metric-label">Verified Real</div>
    </div>
    <div class="metric-container">
        <div class="metric-value" style="color:#ef4444">{stats['fake']}</div>
        <div class="metric-label">Caught Fake</div>
    </div>
    <div class="metric-container">
        <div class="metric-value" style="color:#f59e0b">{stats['avg_trust']}%</div>
        <div class="metric-label">Intelligence Trust</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑️  Clear All History", use_container_width=True):
        clear_history()
        st.toast("History cleared successfully!", icon="🗑️")
        st.rerun()

    st.markdown("<h4 style='color:#475569;'>🕐 Recent History</h4>", unsafe_allow_html=True)
    recent = get_recent(5)
    for row in recent:
        id_, ts, label, trust, verdict = row
        color = "#10b981" if label == "REAL" else "#ef4444"
        st.markdown(f"<p style='font-size:0.85rem; margin-bottom:4px;'><b>#{id_}</b> | <span style='color:{color}'>{label}</span> — {trust}%</p>", unsafe_allow_html=True)

# ── Main Header ──────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 3rem">
    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#007aff; letter-spacing:0.1em; margin-bottom:0.5rem">
        Intelligence Dashboard
    </div>
    <h1 style="font-size:3rem; margin-top:0;">
        News <span style="background: linear-gradient(to right, #007aff, #00c6ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Guardian.</span>
    </h1>
    <p style="font-size:1.1rem; color:#64748b; max-width:650px;">
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
        article = st.text_area(
            label="Article Input",
            placeholder="Enter the news headline or full text here for a deep-dive analysis...",
            height=320,
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
                number={"suffix": "%", "font": {"color": "#007aff", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e2e8f0"},
                    "bar": {"color": "#007aff", "thickness": 1},
                    "bgcolor": "#f8fafc",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239, 68, 68, 0.1)"},
                        {"range": [40, 75], "color": "rgba(245, 158, 11, 0.1)"},
                        {"range": [75, 100], "color": "rgba(16, 185, 129, 0.1)"}
                    ],
                }
            ))
            fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # High-level Verdicts
            v1, v2 = st.columns(2)
            with v1:
                b_class = "badge-real" if label == "REAL" else "badge-fake"
                st.markdown(f"<div style='text-align:center;'><b>BERT Classification</b><br><span class='badge {b_class}' style='margin-top:8px;'>{label}</span></div>", unsafe_allow_html=True)
            with v2:
                ov = fact_result['overall_verdict']
                f_class = "badge-real" if "REAL" in ov else "badge-fake" if "FAKE" in ov else "badge-uncertain"
                st.markdown(f"<div style='text-align:center;'><b>Web Fact Check</b><br><span class='badge {f_class}' style='margin-top:8px;'>{ov}</span></div>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Claim Details
            st.markdown("#### 🔍 Claim Verification Matrix")
            for claim in fact_result["claims"]:
                verdict = claim["verdict"]
                icon = "🟢" if "SUP" in verdict else "🔴" if "CON" in verdict else "🟡"
                with st.expander(f"{icon} {claim['claim'][:60]}..."):
                    st.markdown(f"**Verdict:** `{claim['verdict']}`")
                    st.markdown(f"<p style='color:#64748b;'>{claim['explanation']}</p>", unsafe_allow_html=True)

            # Sentence Logic
            st.markdown("#### 🎨 Contextual Risk Heatmap")
            for s in sentences:
                st.markdown(f"<div class='sent-box sent-{s['risk']}'>{s['sentence']} <span style='float:right; opacity:0.6; font-size:0.75rem;'>{s['fake_prob']}% Risk</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:400px; display:flex; flex-direction:column; justify-content:center; align-items:center; opacity:0.4; border:2px dashed #e2e8f0; border-radius:20px;">
                <div style="font-size:3rem;">🔍</div>
                <div style="font-weight:600;">System ready for input</div>
                <div style="font-size:0.85rem;">Input article data to spawn audit log</div>
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
            color_discrete_sequence=["#10b981", "#ef4444"],
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
