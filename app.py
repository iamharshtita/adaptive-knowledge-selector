"""
Streamlit Web UI — Adaptive Knowledge Selector  (Modern Dark Theme)
Run with:  streamlit run app.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Source metadata ───────────────────────────────────────────────────────────
SOURCE_COLORS = {
    "KnowledgeGraphSource": "#38bdf8",
    "ToolAPISource":        "#4ade80",
    "LLMSource":            "#c084fc",
    "PDFKnowledgeSource":   "#fbbf24",
}
SOURCE_RGB = {
    "KnowledgeGraphSource": "56,189,248",
    "ToolAPISource":        "74,222,128",
    "LLMSource":            "192,132,252",
    "PDFKnowledgeSource":   "251,191,36",
}
SOURCE_ICONS = {
    "KnowledgeGraphSource": "🕸️",
    "ToolAPISource":        "🔧",
    "LLMSource":            "🤖",
    "PDFKnowledgeSource":   "📚",
}
SHORT_NAMES = {
    "KnowledgeGraphSource": "Knowledge Graph",
    "ToolAPISource":        "Tool / API",
    "LLMSource":            "LLM",
    "PDFKnowledgeSource":   "PDF Search",
}

def reward_emoji(r):
    if r >= 0.8:   return "🟢"
    if r >= 0.3:   return "🟡"
    if r >= 0:     return "🟠"
    return "🔴"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Knowledge Selector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, .stApp {
font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
background: #006A67 !important;
background-attachment: fixed !important;
min-height: 100vh;
}

.stApp::before {
content: '';
position: fixed;
inset: 0;
background-image: radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1px);
background-size: 32px 32px;
pointer-events: none;
z-index: 0;
}

[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }

[data-testid="stAppViewContainer"] .block-container {
padding-top: 1.5rem !important;
max-width: 1300px;
}

/* ─── HIDE CHROME ─────────────────────────────────────────────────────────── */
header[data-testid="stHeader"],
footer,
#MainMenu,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ─── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.6); }

/* ─── TYPOGRAPHY ──────────────────────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6 { color: rgba(255,255,255,0.93) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
p, li { color: rgba(255,255,255,0.72); }
code { background: rgba(255,255,255,0.07) !important; color: #a78bfa !important; border-radius: 4px !important; padding: 1px 6px !important; }
pre code { color: rgba(255,255,255,0.8) !important; }

/* ─── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
background: transparent !important;
border-radius: 0 !important;
padding: 0 !important;
border: none !important;
border-bottom: 1px solid #27272a !important;
gap: 0 !important;
justify-content: center !important;
align-items: flex-end !important;
overflow: hidden !important;
}

.stTabs [data-baseweb="tab"] {
border-radius: 0 !important;
font-weight: 500 !important;
font-size: 0.84rem !important;
letter-spacing: 0.01em !important;
padding: 12px 28px !important;
color: #a8b8cc !important;
transition: all 0.15s ease !important;
background: transparent !important;
border: none !important;
border-bottom: 2px solid transparent !important;
white-space: nowrap !important;
flex-shrink: 0 !important;
}

.stTabs [data-baseweb="tab"]:hover {
color: #a1a1aa !important;
background: rgba(39,39,42,0.4) !important;
}

.stTabs [aria-selected="true"] {
background: transparent !important;
color: #f59e0b !important;
border-bottom: 2px solid #f59e0b !important;
font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }

/* ─── BUTTON ──────────────────────────────────────────────────────────────── */
.stButton > button {
border-radius: 8px !important;
font-weight: 600 !important;
font-size: 0.88rem !important;
letter-spacing: 0.03em !important;
padding: 0.6rem 1.5rem !important;
transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"] {
background: #f59e0b !important;
color: #003161 !important;
border: none !important;
font-weight: 700 !important;
box-shadow: 0 1px 0 rgba(0,0,0,0.4), 0 2px 8px rgba(245,158,11,0.2) !important;
}

.stButton > button[kind="primary"]:hover {
background: #d97706 !important;
transform: none !important;
box-shadow: 0 2px 12px rgba(245,158,11,0.35) !important;
}

/* ─── TEXT INPUT ──────────────────────────────────────────────────────────── */
/* ─── TEXT INPUT ─────────────────────────────────────────────────────── */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"] {
background: #18181b !important;
border: 1px solid #3f3f46 !important;
border-radius: 8px !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
.stTextInput input,
[data-testid="stTextInput"] input,
.stTextInput > div > div > input,
input[aria-label] {
background: #18181b !important;
color: #fafafa !important;
caret-color: #fafafa !important;
border-radius: 8px !important;
font-size: 1rem !important;
font-weight: 400 !important;
padding: 12px 16px !important;
border: 1px solid #3f3f46 !important;
box-shadow: none !important;
transition: border-color 0.15s ease !important;
-webkit-text-fill-color: #fafafa !important;
opacity: 1 !important;
}

div[data-baseweb="input"] input::placeholder,
.stTextInput input::placeholder,
[data-testid="stTextInput"] input::placeholder {
color: #94a3b8 !important;
-webkit-text-fill-color: #94a3b8 !important;
}

div[data-baseweb="input"] input:focus,
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
border-color: #f59e0b !important;
box-shadow: 0 0 0 2px rgba(245,158,11,0.15) !important;
background: #1c1c1f !important;
-webkit-text-fill-color: #fafafa !important;
color: #fafafa !important;
outline: none !important;
}

/* ─── METRIC CARDS ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
background: #18181b !important;
border: 1px solid #27272a !important;
border-radius: 10px !important;
padding: 18px 20px !important;
transition: border-color 0.15s ease !important;
}

[data-testid="stMetric"]:hover {
border-color: #3f3f46 !important;
}

[data-testid="stMetricValue"] {
color: #fafafa !important;
font-size: 1.65rem !important;
font-weight: 700 !important;
letter-spacing: -0.03em;
line-height: 1.2 !important;
font-family: 'Space Mono', monospace !important;
}

[data-testid="stMetricLabel"] {
color: #a8b8cc !important;
font-size: 0.68rem !important;
font-weight: 600 !important;
text-transform: uppercase !important;
letter-spacing: 0.1em !important;
}

[data-testid="stMetricDelta"] {
color: #94a3b8 !important;
font-size: 0.74rem !important;
}

/* ─── DIVIDER ─────────────────────────────────────────────────────────────── */
hr {
border: none !important;
height: 1px !important;
background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
margin: 28px 0 !important;
}

/* ─── EXPANDER ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
background: #18181b !important;
border: 1px solid #27272a !important;
border-radius: 10px !important;
overflow: hidden;
}

[data-testid="stExpander"] details {
background: #18181b !important;
}

[data-testid="stExpander"] summary {
color: #a1a1aa !important;
font-weight: 600 !important;
padding: 14px 18px !important;
background: #18181b !important;
}

[data-testid="stExpander"] summary:hover {
color: #fafafa !important;
background: #1c1c1f !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] .streamlit-expanderContent {
background: #18181b !important;
color: #d4d4d8 !important;
}

/* ─── TOOLTIPS (help ? icon) ──────────────────────────────────────────────── */
[data-testid="stTooltipContent"],
div[data-baseweb="tooltip"] div,
div[data-baseweb="popover"] div[data-baseweb="typo-paragraphsmall"] {
background: #27272a !important;
color: #d4d4d8 !important;
border: 1px solid #3f3f46 !important;
border-radius: 6px !important;
}

div[data-baseweb="tooltip"],
div[data-baseweb="popover"] {
background: transparent !important;
}

div[data-baseweb="tooltip"] > div:first-child,
div[data-baseweb="popover"] > div:first-child {
background: #27272a !important;
color: #d4d4d8 !important;
border: 1px solid #3f3f46 !important;
border-radius: 6px !important;
box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
}

/* ─── DATAFRAME / TABLE ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
border-radius: 10px !important;
overflow: hidden !important;
border: 1px solid #27272a !important;
box-shadow: 0 4px 20px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.03) !important;
}

/* Force dark bg on all table/dataframe elements */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] > div {
border-radius: 10px !important;
}

.stDataFrame [data-testid="glideDataEditor"],
.stDataFrame th,
.stDataFrame td {
background: #18181b !important;
color: #d4d4d8 !important;
border-color: #27272a !important;
}

/* ─── CHECKBOX ────────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: rgba(255,255,255,0.62) !important; font-size: 0.87rem !important; }

/* ─── ALERTS ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 14px !important; backdrop-filter: blur(20px) !important; }

/* ─── CAPTION ─────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: rgba(255,255,255,0.65) !important; font-size: 0.77rem !important; }

/* ─── HERO HEADER ─────────────────────────────────────────────────────────── */
.aks-hero {
display: flex;
align-items: center;
justify-content: space-between;
flex-wrap: wrap;
gap: 20px;
padding: 28px 36px;
background: #18181b;
border-radius: 10px;
border: 1px solid #27272a;
border-left: 3px solid #f59e0b;
margin-bottom: 20px;
}

.aks-hero-left { display: flex; align-items: center; gap: 16px; }

.aks-icon {
font-size: 2.4rem;
display: inline-block;
filter: none;
animation: none;
}

.aks-title {
font-size: 1.55rem;
font-weight: 700;
letter-spacing: -0.03em;
color: #fafafa;
margin: 0 0 4px;
line-height: 1.2;
display: block;
}

.aks-sub {
color: #94a3b8;
font-size: 0.76rem;
font-weight: 400;
letter-spacing: 0.02em;
margin: 0;
display: block;
}

.aks-badges {
display: flex;
align-items: center;
gap: 6px;
flex-wrap: wrap;
}

.aks-badge {
padding: 4px 10px;
border-radius: 5px;
font-size: 0.68rem;
font-weight: 600;
letter-spacing: 0.06em;
text-transform: uppercase;
border: 1px solid;
display: inline-block;
}

/* ─── RESULT CARD ─────────────────────────────────────────────────────────── */
.aks-src-card {
border-radius: 20px;
padding: 22px 26px;
backdrop-filter: blur(30px);
position: relative;
overflow: hidden;
box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07), inset 0 -1px 0 rgba(0,0,0,0.12);
transition: all 0.28s ease;
}

.aks-src-card::before {
content: '';
position: absolute;
top: 0; left: 0; right: 0;
height: 1px;
background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent);
pointer-events: none;
}

/* ─── ANSWER CARD ─────────────────────────────────────────────────────────── */
.aks-answer-outer {
border-radius: 18px;
padding: 2px;
margin-top: 16px;
}

.aks-answer-inner {
border-radius: 16px;
padding: 22px 26px;
background: rgba(8,14,28,0.85);
backdrop-filter: blur(24px);
font-family: 'Cascadia Code','Fira Code','JetBrains Mono',monospace;
font-size: 0.865rem;
line-height: 1.78;
color: rgba(255,255,255,0.84);
white-space: pre-wrap;
word-break: break-word;
position: relative;
}

.aks-answer-inner::before {
content: '';
position: absolute;
top: 0; left: 0; right: 0;
height: 1px;
background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
pointer-events: none;
}

/* ─── MISC COMPONENTS ─────────────────────────────────────────────────────── */
.aks-section-head {
font-size: 0.67rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.14em;
color: #a8b8cc;
margin: 24px 0 12px;
display: flex;
align-items: center;
gap: 8px;
}

.aks-section-head::before {
content: '▸';
color: #f59e0b;
font-size: 0.75rem;
}

.aks-section-head::after {
content: '';
flex: 1;
height: 1px;
background: #27272a;
}

.aks-card-label {
font-size: 0.67rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.12em;
color: rgba(255,255,255,0.72);
margin-bottom: 5px;
}

.aks-qtype {
display: inline-block;
padding: 3px 11px;
border-radius: 7px;
font-size: 0.68rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.09em;
background: rgba(255,255,255,0.06);
color: rgba(255,255,255,0.5);
border: 1px solid rgba(255,255,255,0.09);
}

.aks-pill {
display: inline-flex;
align-items: center;
gap: 7px;
padding: 6px 16px;
border-radius: 100px;
font-size: 0.82rem;
font-weight: 700;
letter-spacing: 0.02em;
border: 1px solid;
}

.aks-stat {
display: inline-flex;
align-items: baseline;
gap: 4px;
padding: 4px 12px;
border-radius: 10px;
background: rgba(255,255,255,0.05);
border: 1px solid rgba(255,255,255,0.08);
font-size: 0.82rem;
color: rgba(255,255,255,0.65);
font-weight: 600;
}

.aks-stat strong {
color: rgba(255,255,255,0.92);
font-size: 0.9rem;
font-weight: 800;
}

.aks-ex-cat {
font-size: 0.68rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.1em;
color: rgba(255,255,255,0.72);
margin-bottom: 8px;
display: block;
}

.aks-ex-q {
padding: 7px 13px;
margin: 4px 0;
border-radius: 10px;
font-size: 0.8rem;
color: rgba(255,255,255,0.55);
border: 1px solid rgba(255,255,255,0.06);
background: rgba(255,255,255,0.02);
}

@keyframes aks-glow {
0%,100% { box-shadow: 0 0 14px var(--gc, rgba(99,179,237,0.4)); }
50%      { box-shadow: 0 0 32px var(--gc, rgba(99,179,237,0.7)); }
}

.aks-glow { animation: aks-glow 2.8s ease-in-out infinite; }

/* ─── ABOUT PAGE CARDS ────────────────────────────────────────────────────── */
.aks-about-card {
background: rgba(255,255,255,0.033);
border: 1px solid rgba(255,255,255,0.07);
border-radius: 18px;
padding: 24px 28px;
backdrop-filter: blur(24px);
box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
margin-bottom: 16px;
position: relative;
overflow: hidden;
}

.aks-about-card::before {
content: '';
position: absolute;
top: 0; left: 0; right: 0;
height: 1px;
background: linear-gradient(90deg, transparent, rgba(255,255,255,0.13), transparent);
pointer-events: none;
}

.aks-big-stat {
font-size: 2.4rem;
font-weight: 900;
letter-spacing: -0.04em;
background: linear-gradient(135deg, #60a5fa, #a78bfa);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
line-height: 1.1;
display: block;
}

.aks-stat-label {
font-size: 0.7rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.1em;
color: rgba(255,255,255,0.75);
margin-top: 4px;
display: block;
}

.aks-source-row {
display: flex;
align-items: flex-start;
gap: 16px;
padding: 16px 20px;
border-radius: 14px;
border: 1px solid rgba(255,255,255,0.06);
background: rgba(255,255,255,0.02);
margin-bottom: 10px;
}

.aks-source-icon-big {
font-size: 2rem;
flex-shrink: 0;
margin-top: 2px;
}

.aks-source-name {
font-size: 0.95rem;
font-weight: 700;
color: rgba(255,255,255,0.9);
margin-bottom: 4px;
}

.aks-source-desc {
font-size: 0.82rem;
color: rgba(255,255,255,0.5);
line-height: 1.5;
}

.aks-milestone {
display: flex;
align-items: center;
gap: 12px;
padding: 10px 16px;
border-radius: 10px;
background: rgba(255,255,255,0.02);
border: 1px solid rgba(255,255,255,0.05);
margin-bottom: 6px;
font-size: 0.84rem;
color: rgba(255,255,255,0.65);
}

/* ─── RESPONSIVE ──────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
.aks-hero { flex-direction: column; padding: 20px 16px; gap: 14px; }
.aks-badges { flex-wrap: wrap; gap: 5px; }
.stTabs [data-baseweb="tab"] { padding: 10px 14px !important; font-size: 0.78rem !important; }
[data-testid="stMetric"] { padding: 12px 14px !important; }
.aks-answer-inner { padding: 14px 16px; font-size: 0.8rem; }
}

@media (max-width: 480px) {
.aks-hero { padding: 16px 12px; }
.aks-title { font-size: 1.15rem !important; }
.stTabs [data-baseweb="tab"] { padding: 8px 10px !important; font-size: 0.72rem !important; }
}

/* Ensure main content doesn't overflow on small screens */
.stApp > .main > .block-container {
max-width: 100% !important;
padding-left: clamp(12px, 3vw, 40px) !important;
padding-right: clamp(12px, 3vw, 40px) !important;
}

</style>
""", unsafe_allow_html=True)

# ── JavaScript: force white text in search input (Streamlit overrides CSS) ───
st.markdown("""
<script>
(function() {
  function fixInputs() {
    document.querySelectorAll('input').forEach(function(el) {
      el.style.setProperty('color', '#e8f0ff', 'important');
      el.style.setProperty('-webkit-text-fill-color', '#e8f0ff', 'important');
      el.style.setProperty('opacity', '1', 'important');
    });
  }
  fixInputs();
  new MutationObserver(fixInputs).observe(document.documentElement, {childList:true, subtree:true});
})();
</script>
""", unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_evaluation_metrics():
    """
    Reads data/rl_selector/evaluation_metrics.json (from evaluate_model_metrics.py).
    @st.cache_data serialises the result — parsed once per session.
    """
    path = os.path.join(ROOT, "data", "rl_selector", "evaluation_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_training_log():
    """
    Reads data/rl_selector/training_log.json (from train_rl_agent.py).
    @st.cache_data serialises the result — parsed once per session.
    """
    path = os.path.join(ROOT, "data", "rl_selector", "training_log.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Initialising knowledge sources…")
def load_system():
    """
    @st.cache_resource keeps Python objects alive across reruns (no re-init).
    Loads: SentenceTransformer encoder, AdaptiveSelector DQN, TrainingSystem.
    """
    from sentence_transformers import SentenceTransformer
    from models.adaptive_selector import AdaptiveSelector
    from scripts.train_rl_agent import TrainingSystem

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    agent   = AdaptiveSelector(input_dim=384, num_sources=4)
    model_path = os.path.join(ROOT, "data", "rl_selector", "adaptive_dqn.pth")
    model_loaded = agent.load(model_path)
    system = TrainingSystem()
    return encoder, agent, system, model_loaded


@st.cache_resource(show_spinner=False)
def load_judge():
    try:
        from utils.llm_judge import LLMJudge
        return LLMJudge()
    except Exception:
        return None


# ── Chart builders ────────────────────────────────────────────────────────────
_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font={"color": "rgba(255,255,255,0.65)", "family": "Inter, sans-serif", "size": 12},
    margin={"l": 0, "r": 0, "t": 46, "b": 0},
)

def _lay(fig, **kw):
    fig.update_layout(**{**_BASE, **kw})
    return fig

_AXIS = {"gridcolor": "rgba(255,255,255,0.06)", "zerolinecolor": "rgba(255,255,255,0.08)"}


def chart_qvalues(qtable, selected):
    """Clean ranked bar chart: one row per source, value label on right, selected highlighted."""
    items = sorted(qtable.items(), key=lambda x: x[1], reverse=True)
    n = len(items)

    y_labels, bar_colors, values, customdata = [], [], [], []
    for rank, (k, v) in enumerate(items, 1):
        name   = SHORT_NAMES.get(k, k)
        icon   = SOURCE_ICONS.get(k, "")
        is_sel = k == selected
        sel_txt = " ✦ SELECTED" if is_sel else ""
        y_labels.append(f"#{rank}  {icon} {name}{sel_txt}")
        bar_colors.append(SOURCE_COLORS.get(k, "#aaa") if is_sel
                          else f"rgba({SOURCE_RGB.get(k, '136,136,136')},0.25)")
        values.append(v)
        customdata.append(name)

    # Determine axis range so labels never clip
    vmin = min(values)
    vmax = max(values)
    pad  = (vmax - vmin) * 0.35 if vmax != vmin else 0.5
    x_range = [vmin - 0.1, vmax + pad]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=y_labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
            cornerradius=4,
        ),
        text=[f" {v:+.3f}" for v in values],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(
            color="rgba(255,255,255,0.85)",
            size=12,
            family="Space Mono, monospace",
        ),
        hovertemplate="<b>%{customdata}</b><br>Q-Score: %{x:+.4f}<extra></extra>",
        customdata=customdata,
    ))

    # Zero reference line
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)

    return _lay(fig,
        title=dict(
            text="Routing Decision — Q-Score per Knowledge Source",
            font=dict(size=12, color="rgba(255,255,255,0.45)", family="Space Grotesk"),
            x=0,
        ),
        height=n * 72 + 60,
        margin=dict(l=20, r=90, t=44, b=30),
        bargap=0.35,
        xaxis=dict(
            range=x_range,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.15)",
            zeroline=True,
            showticklabels=True,
            tickfont=dict(size=10, color="rgba(255,255,255,0.35)"),
            title=dict(
                text="← worse    Q-Score    better →",
                font=dict(size=9, color="rgba(255,255,255,0.55)"),
                standoff=4,
            ),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed",
            tickfont=dict(size=12, color="rgba(255,255,255,0.82)", family="Space Grotesk"),
            automargin=True,
        ),
    )


def chart_kg_graph(query: str, answer: str):
    """Network graph visualization for Knowledge Graph query results."""
    import re, math

    # Patterns that flag section headers / query echoes — not real drug entities
    _SKIP_RE = re.compile(
        r'for\s*:|interactions?\s+(for|of)|results?\s+for|found\s+\d+|'
        r'no\s+results|source\s*:|answer\s*:|query\s*:',
        re.IGNORECASE,
    )
    query_norm = re.sub(r'[^a-z0-9 ]', '', query.lower()).strip()

    lines = [l.strip() for l in answer.split('\n') if l.strip()]
    entities = []
    for line in lines:
        clean = re.sub(r'^[-\u2022*\d.]+\s*', '', line).strip()
        if not (3 < len(clean) < 55):
            continue
        if clean.endswith(':'):
            continue
        if _SKIP_RE.search(clean):
            continue
        # Skip lines that substantially repeat the query
        clean_norm = re.sub(r'[^a-z0-9 ]', '', clean.lower()).strip()
        if clean_norm in query_norm or query_norm in clean_norm:
            continue
        if sum(1 for w in ['the ', 'this ', 'has ', 'have ', 'is a ', 'are '] if w in clean.lower()) > 0:
            continue
        entity = clean.split('(')[0].split(' - ')[0].strip()[:40]
        if entity:
            entities.append(entity)

    entities = list(dict.fromkeys(entities))[:14]
    if not entities:
        return None

    n_nodes = len(entities) + 1   # entities + center hub
    n_edges = len(entities)        # one spoke per entity

    center_label = query
    positions = {}
    for i, e in enumerate(entities):
        angle = 2 * math.pi * i / len(entities) - math.pi / 2
        positions[e] = (1.9 * math.cos(angle), 1.9 * math.sin(angle))

    fig = go.Figure()
    for e in entities:
        x1, y1 = positions[e]
        fig.add_trace(go.Scatter(
            x=[0, x1, None], y=[0, y1, None], mode='lines',
            line=dict(color='rgba(129,140,248,0.18)', width=1.2),
            hoverinfo='none', showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[positions[e][0] for e in entities],
        y=[positions[e][1] for e in entities],
        mode='markers+text',
        marker=dict(color='rgba(52,211,153,0.85)', size=13,
                    line=dict(width=1.5, color='rgba(255,255,255,0.15)')),
        text=[f'<b>{e}</b>' for e in entities], textposition='top center',
        textfont=dict(color='rgba(255,255,255,0.85)', size=10, family='Space Grotesk'),
        hovertext=entities, hoverinfo='text', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(color='#818cf8', size=30,
                    line=dict(width=2, color='rgba(165,180,252,0.5)')),
        text=[f'<b>{center_label}</b>'], textposition='bottom center',
        textfont=dict(color='#a5b4fc', size=10, family='Space Grotesk'),
        hovertext=[center_label], hoverinfo='text', showlegend=False,
    ))
    return _lay(fig,
        title=dict(
            text=f'Knowledge Graph — Entity Map  <span style="font-size:11px;color:#FDFAF6;font-weight:500;">{n_nodes} nodes · {n_edges} edges</span>',
            font=dict(size=12, color='rgba(255,255,255,0.75)', family='Space Grotesk'),
            x=0,
        ),
        height=500, showlegend=False,
        margin=dict(l=40, r=40, t=52, b=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.8, 2.8],
                   fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.8, 2.8],
                   fixedrange=True),
        annotations=[dict(
            text="Query → extracted entities from the Knowledge Graph answer",
            xref="paper", yref="paper", x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=9, color="rgba(255,255,255,0.55)", family="Space Grotesk"),
        )],
    )


def chart_confusion_matrix(cm):
    sources = list(cm.keys())
    short   = [SHORT_NAMES.get(s, s) for s in sources]
    matrix  = np.array([[cm[t].get(p, 0) for p in sources] for t in sources], dtype=float)
    z_norm  = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    text    = [[str(int(matrix[i][j])) for j in range(len(sources))] for i in range(len(sources))]
    fig = go.Figure(go.Heatmap(
        z=z_norm, x=short, y=short, text=text, texttemplate="%{text}",
        textfont={"size": 15, "color": "white"},
        colorscale=[[0.0, "rgba(29,78,216,0.1)"], [0.5, "rgba(124,58,237,0.55)"], [1.0, "rgba(192,132,252,0.95)"]],
        showscale=True, zmin=0, zmax=1,
        colorbar={"title": {"text": "Row %", "font": {"color": "rgba(255,255,255,0.5)", "size": 11}},
                  "tickformat": ".0%", "tickfont": {"color": "rgba(255,255,255,0.5)"}, "thickness": 10, "len": 0.85},
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
    ))
    return _lay(fig,
        title={"text": "Confusion Matrix  (row=true, col=predicted)", "font": {"size": 12, "color": "rgba(255,255,255,0.5)"}},
        xaxis={**_AXIS, "title": "Predicted", "side": "bottom", "tickfont": {"color": "rgba(255,255,255,0.65)"}},
        yaxis={**_AXIS, "title": "True", "autorange": "reversed", "tickfont": {"color": "rgba(255,255,255,0.65)"}},
        height=360,
    )


def chart_per_source_metrics(sm):
    sources = list(sm.keys())
    short   = [SHORT_NAMES.get(s, s) for s in sources]
    fig = go.Figure()
    for metric, color in [("precision", "#38bdf8"), ("recall", "#4ade80"), ("f1", "#fbbf24")]:
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=short,
            y=[sm[s][metric] for s in sources],
            marker={"color": color, "opacity": 0.88, "line": {"width": 0},
                     "cornerradius": 3},
            text=[f"{sm[s][metric]:.2f}" for s in sources],
            textposition="outside",
            cliponaxis=False,
            textfont={"size": 10, "color": "rgba(255,255,255,0.7)"},
        ))
    return _lay(fig,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        title={"text": "Per-Source Classification Metrics", "font": {"size": 12, "color": "rgba(255,255,255,0.75)"}},
        yaxis={**_AXIS, "title": "Score", "range": [0, 1.18], "tickformat": ".1f",
               "dtick": 0.2},
        xaxis={**_AXIS, "tickfont": {"color": "rgba(255,255,255,0.65)", "size": 11}},
        height=380,
        margin=dict(t=60, b=40),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1,
                "font": {"color": "rgba(255,255,255,0.6)", "size": 11}, "bgcolor": "rgba(0,0,0,0)"},
    )


def chart_reward_history(rewards):
    eps = list(range(1, len(rewards) + 1))
    window  = min(10, len(rewards))
    rolling = [np.mean(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=rewards, mode="lines", name="Episode Reward",
                             line={"color": "rgba(192,132,252,0.35)", "width": 1.2}))
    fig.add_trace(go.Scatter(x=eps, y=rolling, mode="lines", name=f"{window}-ep Rolling Avg",
                             line={"color": "#c084fc", "width": 2.5},
                             fill="tozeroy", fillcolor="rgba(192,132,252,0.07)"))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.12)", line_width=1)
    return _lay(fig,
        title={"text": "Training Reward per Episode", "font": {"size": 12, "color": "rgba(255,255,255,0.5)"}},
        xaxis={**_AXIS, "title": "Episode"}, yaxis={**_AXIS, "title": "Reward"}, height=270,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "font": {"color": "rgba(255,255,255,0.55)", "size": 11}, "bgcolor": "rgba(0,0,0,0)"},
    )


def chart_loss(losses):
    if not losses:
        fig = go.Figure()
        return _lay(fig, title="Training Loss", height=270,
                    annotations=[{"text": "No loss data", "showarrow": False,
                                  "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                                  "font": {"color": "rgba(255,255,255,0.6)"}}])
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(losses) + 1)), y=losses, mode="lines",
        line={"color": "#4ade80", "width": 2.2},
        fill="tozeroy", fillcolor="rgba(74,222,128,0.08)",
    ))
    return _lay(fig,
        title={"text": "Training Loss (MSE)", "font": {"size": 12, "color": "rgba(255,255,255,0.5)"}},
        xaxis={**_AXIS, "title": "Batch"}, yaxis={**_AXIS, "title": "Loss"}, height=270,
    )


def chart_source_distribution(picks):
    labels = [SHORT_NAMES.get(k, k) for k in picks]
    colors = [SOURCE_COLORS.get(k, "#aaa") for k in picks]
    fig = go.Figure(go.Pie(
        labels=labels, values=list(picks.values()), hole=0.52,
        marker={"colors": colors, "line": {"color": "rgba(8,14,28,0.8)", "width": 2}},
        textinfo="label+percent",
        textfont={"color": "rgba(255,255,255,0.75)", "size": 11},
        hovertemplate="%{label}: %{value} episodes<extra></extra>",
    ))
    return _lay(fig,
        title={"text": "Source Selection During Training", "font": {"size": 12, "color": "rgba(255,255,255,0.5)"}},
        height=270, showlegend=False,
    )


def chart_epsilon(episodes):
    eps_vals = [e["epsilon"] for e in episodes]
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(eps_vals) + 1)), y=eps_vals, mode="lines",
        line={"color": "#fbbf24", "width": 2.2},
        fill="tozeroy", fillcolor="rgba(251,191,36,0.08)",
    ))
    return _lay(fig,
        title={"text": "Epsilon (Exploration) Decay", "font": {"size": 12, "color": "rgba(255,255,255,0.5)"}},
        xaxis={**_AXIS, "title": "Episode"}, yaxis={**_AXIS, "title": "Epsilon"}, height=270,
    )


def chart_judge_scores(scores):
    dims   = ["Correctness", "Relevance", "Completeness"]
    vals   = [scores["correctness"], scores["relevance"], scores["completeness"]]
    colors = ["#38bdf8", "#4ade80", "#c084fc"]
    fig = go.Figure(go.Bar(
        x=dims, y=vals,
        marker={"color": colors, "opacity": 0.85, "line": {"width": 0}},
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        textfont={"color": "rgba(255,255,255,0.7)", "size": 12},
        cliponaxis=False,
    ))
    return _lay(fig,
        title={"text": f"LLM Judge Scores — Overall Quality: {scores['quality']:.2f}",
               "font": {"size": 12, "color": "rgba(255,255,255,0.5)"}},
        yaxis={**_AXIS, "range": [0, 1.28], "title": "Score", "tickformat": ".2f"},
        xaxis={**_AXIS, "tickfont": {"color": "rgba(255,255,255,0.65)", "size": 12}},
        height=240, showlegend=False,
    )


# ── Helper: inline section heading ───────────────────────────────────────────
def section_head(label):
    st.markdown(f'<div class="aks-section-head">{label}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER  (no 4-space indentation inside HTML — avoids markdown code block)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
'<div class="aks-hero">'
'<div class="aks-hero-left">'
'<span class="aks-icon">🧠</span>'
'<div><span class="aks-title">Adaptive Knowledge Selector</span>'
'<span class="aks-sub">Reinforcement Learning · Medical Query Routing · ASU KRR 2026</span></div>'
'</div>'
'<div class="aks-badges">'
'<span class="aks-badge" style="color:#38bdf8;border-color:rgba(56,189,248,0.3);background:rgba(56,189,248,0.06);">🕸️ Knowledge Graph</span>'
'<span class="aks-badge" style="color:#4ade80;border-color:rgba(74,222,128,0.3);background:rgba(74,222,128,0.06);">🔧 Tool / API</span>'
'<span class="aks-badge" style="color:#c084fc;border-color:rgba(192,132,252,0.3);background:rgba(192,132,252,0.06);">🤖 LLM</span>'
'<span class="aks-badge" style="color:#fbbf24;border-color:rgba(251,191,36,0.3);background:rgba(251,191,36,0.06);">📚 PDF Search</span>'
'</div>'
'</div>',
unsafe_allow_html=True,
)

tab_query, tab_metrics, tab_training, tab_about = st.tabs([
    "💬  Query Interface",
    "📊  Model Metrics",
    "📈  Training History",
    "ℹ️  About",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — QUERY INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:

    try:
        encoder, agent, system, model_loaded = load_system()
    except Exception as exc:
        st.error(f"Failed to initialise system: {exc}")
        st.stop()

    if not model_loaded:
        st.warning("⚠️ No model weights found. Run `pretrain_supervised.py` then `train_rl_agent.py` first.")

    # Input row
    q_col, btn_col = st.columns([6, 1])
    with q_col:
        raw_query = st.text_input("query_input",
            placeholder="e.g.  What drugs interact with warfarin?",
            label_visibility="collapsed")
    with btn_col:
        ask_clicked = st.button("Ask  ↵", type="primary", use_container_width=True)

    # Example queries
    with st.expander("💡  Example queries", expanded=False):
        ex_cols = st.columns(4)
        examples = [
            ("🕸️", "Drug Interactions", [
                "What drugs interact with warfarin?",
                "Can I take aspirin with ibuprofen?",
                "Does metformin interact with alcohol?",
            ]),
            ("🔧", "Calculations", [
                "Calculate BMI for 85 kg and 1.80 m",
                "CrCl: 60 yr male, 70 kg, creatinine 1.0",
                "Pediatric dose: 25 kg child, adult dose 400 mg",
            ]),
            ("🤖", "Concepts", [
                "How does insulin regulate blood sugar?",
                "Explain the mechanism of ACE inhibitors",
                "What is pharmacokinetics vs pharmacodynamics?",
            ]),
            ("📚", "Documents", [
                "What do KRR papers say about ontologies?",
                "Knowledge representation in clinical decision support",
                "Medical reasoning approaches in the documents",
            ]),
        ]
        for col, (icon, cat, qs) in zip(ex_cols, examples):
            with col:
                items = "".join(f'<div class="aks-ex-q">• {q}</div>' for q in qs)
                st.markdown(f'<span class="aks-ex-cat">{icon} {cat}</span>{items}', unsafe_allow_html=True)

    st.divider()

    # ── Process query ────────────────────────────────────────────────────────────
    if ask_clicked and raw_query.strip():
        from models.reward_evaluator import RewardEvaluator, classify_query

        prog = st.empty()
        prog.markdown(
            '<div style="background:#18181b;border:1px solid #27272a;border-radius:8px;'
            'padding:16px 20px;margin-bottom:12px;">'
            '<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;'
            'color:#f59e0b;margin-bottom:10px;">⏳ Processing query…</div>'
            + "".join([
                f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
                f'font-size:0.78rem;color:#94a3b8;">'
                f'<span style="width:7px;height:7px;border-radius:50%;background:#27272a;flex-shrink:0;"></span>'
                f'{step}</div>'
                for step in [
                    "Classifying query type…",
                    "Embedding query into 384-dim vector…",
                    "Computing Q-scores for all knowledge sources…",
                    "Selecting optimal source (greedy policy)…",
                    "Retrieving answer…",
                ]
            ])
            + "</div>",
            unsafe_allow_html=True,
        )

        qtype       = classify_query(raw_query)
        emb         = encoder.encode([raw_query])[0]
        qtable      = agent.get_q_table(emb)
        action_idx  = agent.select_action(emb, epsilon=0.0)
        source_name = agent.sources[action_idx]
        results     = system.query_source(source_name, raw_query)
        if isinstance(results, dict):
            answer     = results.get("answer", str(results))
            confidence = results.get("confidence", 0.5)
        else:
            answer     = str(results) if results else "No answer returned."
            confidence = 0.5
        rwd = RewardEvaluator.compute_reward(raw_query, source_name, results)
        prog.empty()

        st.session_state["result"] = dict(
            query=raw_query, qtype=qtype, qtable=qtable,
            source_name=source_name, answer=answer,
            confidence=confidence, reward=rwd,
        )
        st.session_state.pop("judge_result", None)

    elif ask_clicked:
        st.warning("Please enter a query first.")

    # ── Display results (ordered steps) ──────────────────────────────────────────
    if "result" in st.session_state:
        res      = st.session_state["result"]
        src_name = res["source_name"]
        clr      = SOURCE_COLORS.get(src_name, "#888")
        rgb      = SOURCE_RGB.get(src_name, "136,136,136")
        icon     = SOURCE_ICONS.get(src_name, "")
        short    = SHORT_NAMES.get(src_name, src_name)
        conf_pct = int(res["confidence"] * 100)
        rwd_clr  = "#22c55e" if res["reward"] > 0 else "#ef4444"
        best_q   = max(res["qtable"].values())

        # ── 1 · Query Type ────────────────────────────────────────────────────
        section_head("1 · Query Classification")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;padding:14px 18px;'
            f'background:#18181b;border:1px solid #27272a;border-radius:8px;margin-bottom:4px;">'
            f'<span style="padding:4px 12px;border-radius:5px;font-size:0.78rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.08em;background:#27272a;color:#e4e4e7;">'
            f'{res["qtype"]}</span>'
            f'<span style="font-size:0.8rem;color:#94a3b8;line-height:1.6;">'
            f'Classified as <strong style="color:#a1a1aa;">{res["qtype"]}</strong> — '
            f'guides which knowledge sources are likely most relevant.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── 2 · Pipeline Steps ───────────────────────────────────────────────
        section_head("2 · Routing Pipeline")
        pipeline = [
            ("💬", "Query",       res["query"],                                           "#a1a1aa"),
            ("📐", "Embedding",   "Encoded to 384-dim vector via sentence-transformer",   "#a8b8cc"),
            ("🧠", "DQN Scoring", f"Q-values computed for {len(res['qtable'])} sources","#a8b8cc"),
            ("✅",      "Selected",    f"{icon} {short}  —  Q = {best_q:+.3f}", clr),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;'
            f'{"" if i==len(pipeline)-1 else "border-bottom:1px solid #27272a;"}">'
            f'<span style="font-size:1rem;margin-top:1px;flex-shrink:0;">{ico}</span>'
            f'<div><div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.1em;color:#94a3b8;margin-bottom:2px;">{lbl}</div>'
            f'<div style="font-size:0.82rem;color:{dc};">{dtl}</div></div></div>'
            for i, (ico, lbl, dtl, dc) in enumerate(pipeline)
        )
        st.markdown(
            f'<div style="padding:0 18px;background:#18181b;border:1px solid #27272a;'
            f'border-radius:8px;margin-bottom:4px;">{rows}</div>',
            unsafe_allow_html=True,
        )

        # ── 3 · Routing Decision (Q-values chart) ────────────────────────────
        section_head("3 · Routing Decision")
        st.plotly_chart(chart_qvalues(res["qtable"], src_name),
                        use_container_width=True, config={"displayModeBar": False})

        # ── 4 · Decision Explanation ──────────────────────────────────────────
        section_head("4 · Decision Explanation")
        scores_html = " · ".join(
            f'<span style="color:{SOURCE_COLORS.get(k,"#888")};font-weight:600;">'
            f'{SHORT_NAMES.get(k,k)} <code style="font-size:0.75rem;">{v:+.3f}</code></span>'
            for k, v in sorted(res["qtable"].items(), key=lambda x: x[1], reverse=True)
        )
        st.markdown(
            f'<div style="background:#18181b;border:1px solid #27272a;border-left:3px solid #f59e0b;'
            f'border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#a1a1aa;line-height:1.8;">'
            f'<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#94a3b8;margin-bottom:8px;">How the DQN Decided</div>'
            f'The query embedding was fed through the DQN policy network, which outputs a Q-value '
            f'representing expected reward per source.<br>'
            f'<span style="color:#a8b8cc;">All scores: {scores_html}</span><br>'
            f'<strong style="color:{clr};">{icon} {short}</strong> had the highest Q-score '
            f'(<code style="color:#22c55e;background:#0f2318;padding:1px 6px;border-radius:3px;">'
            f'Q = {best_q:+.3f}</code>) — selected as optimal source.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── 5 · Answer ───────────────────────────────────────────────────────
        section_head("5 · Answer")
        st.markdown(
            f'<div class="aks-answer-outer">'
            f'<div class="aks-answer-inner">{res["answer"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── 6 · KG Visualization ─────────────────────────────────────────────
        if src_name == "KnowledgeGraphSource":
            kg_fig = chart_kg_graph(res["query"], res["answer"])
            if kg_fig:
                section_head("6 · Knowledge Graph — Entity Map")
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#cbd5e1;margin-bottom:8px;line-height:1.6;">'
                    f'Entities extracted from the KG answer for query: '
                    f'<em style="color:#e2e8f0;">"{res["query"]}"</em></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(kg_fig, use_container_width=True,
                                config={"displayModeBar": False})

        # ── 7 · LLM Judge Evaluation ──────────────────────────────────────────
        section_head("7 · LLM Judge Evaluation")
        run_judge = st.checkbox("Run LLM Judge  *(requires AWS Bedrock credentials)*",
                                value=False, key="judge_toggle")

        if run_judge:
            if "judge_result" not in st.session_state:
                judge = load_judge()
                if judge:
                    with st.spinner("Evaluating answer quality with LLM Judge…"):
                        try:
                            st.session_state["judge_result"] = judge.evaluate_quality(
                                res["query"], res["answer"], res["source_name"])
                        except Exception as exc:
                            st.error(f"LLM Judge failed: {exc}")
                else:
                    st.info("LLM Judge is unavailable — set AWS credentials in `.env`.")

            if "judge_result" in st.session_state:
                jr = st.session_state["judge_result"]
                q_score = jr["quality"]
                if q_score >= 0.7:
                    badge_color, badge_text = "green",  "✅  High Quality"
                elif q_score >= 0.4:
                    badge_color, badge_text = "orange", "⚠️  Acceptable"
                else:
                    badge_color, badge_text = "red",    "❌  Low Quality"

                mc0, mc1, mc2, mc3 = st.columns(4)
                mc0.metric("Overall Quality",  f"{q_score:.2f}")
                mc1.metric("Correctness",      f"{jr['correctness']:.2f}")
                mc2.metric("Relevance",        f"{jr['relevance']:.2f}")
                mc3.metric("Completeness",     f"{jr['completeness']:.2f}")

                if q_score > 0.0:
                    st.plotly_chart(chart_judge_scores(jr), use_container_width=True)

                st.markdown(
                    f"**Assessment:** :{badge_color}[{badge_text}]  \n"
                    f"**Reason:** {jr.get('explanation', '—')}"
                )
        else:
            st.caption("Enable the checkbox above to evaluate this answer with the LLM Judge.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    metrics = load_evaluation_metrics()

    if not metrics:
        st.warning("No evaluation metrics found. Run `python scripts/evaluate_model_metrics.py`.")
    else:
        data           = metrics.get("manual", metrics)
        accuracy       = data["accuracy"]
        source_metrics = data["source_metrics"]
        cm_data        = data["confusion_matrix"]

        total_support = sum(sm["support"] for sm in source_metrics.values())
        st.markdown(
            f'<div class="aks-about-card" style="text-align:center;background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(29,78,216,0.08));border-color:rgba(124,58,237,0.2);">'
            f'<div class="aks-card-label">Overall Model Accuracy</div>'
            f'<span class="aks-big-stat">{accuracy:.1%}</span>'
            f'<span style="color:rgba(255,255,255,0.35);font-size:0.8rem;">evaluated on {total_support} holdout queries</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        section_head("Per-Source Performance")
        st.markdown(
            '<div style="font-size:0.74rem;color:rgba(255,255,255,0.35);margin:-8px 0 14px;">'  
            '🟢 ≥ 0.90 — Excellent • '
            '🟡 0.75– 0.90 — Good • '
            '🔴 < 0.75 — Needs improvement'
            '</div>',
            unsafe_allow_html=True,
        )
        cols_src = st.columns(4)
        for col, src in zip(cols_src, ["KnowledgeGraphSource", "ToolAPISource", "LLMSource", "PDFKnowledgeSource"]):
            sm = source_metrics.get(src, {})
            f1 = sm.get('f1', 0)
            badge = '🟢' if f1 >= 0.90 else ('🟡' if f1 >= 0.75 else '🔴')
            tip = ('Excellent routing accuracy' if f1 >= 0.90
                   else ('Good — minor misrouting on ambiguous queries' if f1 >= 0.75
                         else 'Lower — open-ended queries are harder to classify'))
            col.metric(
                f"{SOURCE_ICONS.get(src,'')}  {SHORT_NAMES.get(src, src)}  F1  {badge}",
                f"{f1:.3f}",
                f"P {sm.get('precision',0):.2f} · R {sm.get('recall',0):.2f}",
                help=tip,
            )

        st.divider()

        section_head("Charts")
        st.markdown(
            '<div style="font-size:0.74rem;color:rgba(255,255,255,0.32);margin:-10px 0 10px;">'  
            'Left: Per-source F1/Precision/Recall bar chart.  '
            'Right: Confusion matrix — <strong style="color:rgba(165,180,252,0.7);">rows = Actual source</strong>, '
            '<strong style="color:rgba(52,211,153,0.7);">columns = Predicted source</strong>. '
            'On-diagonal — correct routing. Off-diagonal — misrouted queries.'
            '</div>',
            unsafe_allow_html=True,
        )
        bar_col, cm_col = st.columns(2)
        with bar_col:
            st.plotly_chart(chart_per_source_metrics(source_metrics), use_container_width=True)
        with cm_col:
            st.plotly_chart(chart_confusion_matrix(cm_data), use_container_width=True)

        section_head("Detailed Metrics")
        import pandas as pd
        rows = [{"Source": f"{SOURCE_ICONS.get(s,'')}  {SHORT_NAMES.get(s, s)}",
                 "Precision": round(sm["precision"], 4),
                 "Recall":    round(sm["recall"],    4),
                 "F1 Score":  round(sm["f1"],        4),
                 "Support":   sm["support"]}
                for s, sm in source_metrics.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("📋  Raw Confusion Matrix Values"):
            sources  = list(cm_data.keys())
            cm_rows  = [{"True \\ Predicted": f"{SOURCE_ICONS.get(t,'')} {SHORT_NAMES.get(t, t)}",
                         **{SHORT_NAMES.get(p, p): cm_data[t].get(p, 0) for p in sources}}
                        for t in sources]
            st.dataframe(pd.DataFrame(cm_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_training:
    tlog = load_training_log()

    if not tlog:
        st.warning("No training log found. Run `python scripts/train_rl_agent.py`.")
    else:
        rewards      = tlog["rewards"]
        losses       = tlog["losses"]
        episodes     = tlog["episodes"]
        source_picks = tlog["source_picks"]
        total_eps    = tlog["total_episodes"]
        elapsed      = tlog["elapsed_seconds"]

        section_head("Training Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Episodes",      total_eps)
        s2.metric("Avg Reward",    f"{np.mean(rewards):.3f}")
        s3.metric("Best Reward",   f"{max(rewards):+.3f}")
        s4.metric("Final Epsilon", f"{episodes[-1]['epsilon']:.3f}")
        s5.metric("Training Time", f"{elapsed:.1f}s")

        st.divider()

        section_head("Learning Curves")
        r_col, l_col = st.columns(2)
        with r_col:
            st.plotly_chart(chart_reward_history(rewards), use_container_width=True)
        with l_col:
            st.plotly_chart(chart_loss(losses), use_container_width=True)

        section_head("Exploration Behaviour")
        pie_col, eps_col = st.columns(2)
        with pie_col:
            st.plotly_chart(chart_source_distribution(source_picks), use_container_width=True)
        with eps_col:
            st.plotly_chart(chart_epsilon(episodes), use_container_width=True)

        with st.expander("📋  Episode-level Detail"):
            import pandas as pd
            df = pd.DataFrame(episodes)
            df["source_chosen"] = df["source_chosen"].map(
                lambda x: f"{SOURCE_ICONS.get(x,'')} {SHORT_NAMES.get(x, x)}")
            df.columns = [c.replace("_", " ").title() for c in df.columns]
            st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:

    # Project overview card
    st.markdown(
        '<div class="aks-about-card" style="background:linear-gradient(135deg,rgba(29,78,216,0.12),rgba(124,58,237,0.1));border-color:rgba(124,58,237,0.2);">'
        '<span class="aks-title" style="font-size:1.8rem;margin-bottom:10px;display:block;">Adaptive Knowledge Selector for Medical Queries</span>'
        '<p style="color:rgba(255,255,255,0.62);font-size:0.92rem;line-height:1.7;margin:0 0 12px;">An intelligent reinforcement learning system that dynamically selects the best knowledge source — Knowledge Graph, Tools/APIs, LLM, or Documents — for medical and pharmaceutical queries. Built with Deep Q-Networks (DQN) and evaluated with LLM-as-Judge quality metrics.</p>'
        '<p style="color:rgba(255,255,255,0.5);font-size:0.84rem;line-height:1.6;margin:0;">The system intelligently routes medical queries to the most appropriate knowledge source using a trained RL agent. It integrates four distinct biomedical knowledge sources and learns optimal routing decisions through reinforcement learning with automatic reward signals.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Key stats row
    section_head("Key Stats")
    ka, kb, kc, kd, ke, kf = st.columns(6)
    ka.metric("Knowledge Sources", "4")
    kb.metric("Graph Nodes",       "47,031")
    kc.metric("Graph Edges",       "2.25M")
    kd.metric("PDF Chunks",        "1,649")
    ke.metric("Training Queries",  "600")
    kf.metric("Test Queries",      "250")

    st.divider()

    # Knowledge sources
    section_head("Knowledge Sources")
    sources_info = [
        ("KnowledgeGraphSource", "Hetionet Knowledge Graph",
         "47,031 biomedical nodes (1,552 drugs, 137 diseases, 20,945 genes) · 2,250,197 edges · Drug interactions, side effects, disease associations, gene relationships",
         "Best for: drug interactions, disease-drug links, target identification"),
        ("ToolAPISource", "Tool / API",
         "OpenFDA drug labels & adverse events · RxNorm drug terminology · Medical calculators: BMI, CrCl (Cockcroft-Gault), IBW, Pediatric dosing · Confidence: 1.0 (success) / 0.0 (error)",
         "Best for: FDA labels, drug info lookup, medical calculations"),
        ("LLMSource", "LLM (AWS Bedrock Nova 2 Lite)",
         "Amazon Nova 2 Lite · ~$0.0001/query · Explanations, mechanisms, conceptual medical questions · Confidence: 0.8 (good), 0.3 (uncertain), 0.0 (error)",
         "Best for: mechanistic explanations, conceptual questions, reasoning"),
        ("PDFKnowledgeSource", "PDF Knowledge (FAISS)",
         "3 KRR medical informatics papers · 1,649 semantic chunks · FAISS vector store · Sentence Transformers (all-MiniLM-L6-v2, 384-dim) · Confidence: FAISS similarity score (0-1)",
         "Best for: document-specific queries, KRR papers, research content"),
    ]
    for src, name, desc, use in sources_info:
        clr = SOURCE_COLORS.get(src, "#888")
        rgb = SOURCE_RGB.get(src, "136,136,136")
        ico = SOURCE_ICONS.get(src, "")
        st.markdown(
            f'<div class="aks-source-row" style="border-color:rgba({rgb},0.15);background:linear-gradient(90deg,rgba({rgb},0.05),transparent);">'
            f'<div class="aks-source-icon-big">{ico}</div>'
            f'<div><div class="aks-source-name" style="color:{clr};">{name}</div>'
            f'<div class="aks-source-desc">{desc}</div>'
            f'<div style="margin-top:6px;font-size:0.76rem;color:rgba({rgb},0.8);font-weight:600;">{use}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Architecture + Training
    arch_col, train_col = st.columns(2)

    with arch_col:
        section_head("System Architecture")
        st.markdown(
            '<div class="aks-about-card">'
            '<p style="color:rgba(255,255,255,0.65);font-size:0.88rem;line-height:1.7;margin:0 0 12px;">'
            'Queries are embedded into 384-dimensional vectors using Sentence Transformers, '
            'then passed through a <strong style="color:#a78bfa;">Deep Q-Network</strong> (384→256→128→64→4) '
            'that predicts Q-values for each knowledge source. The source with the highest Q-value is selected '
            '(greedy policy, ε=0 at inference).</p>'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.45);font-family:monospace;">'
            'Query → Encoder (384-dim) → DQN → Q-values → argmax → Source → Answer → Reward'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        section_head("Reward Function")
        st.markdown(
            '<div class="aks-about-card">'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.65);font-family:monospace;line-height:1.8;">'
            'Correct source + results  →  <span style="color:#4ade80;">+1.00</span><br>'
            'Correct source, no results  →  <span style="color:#fbbf24;">-0.20</span><br>'
            'Wrong source + results  →  <span style="color:#fbbf24;">0.00</span><br>'
            'Wrong source, no results  →  <span style="color:#f87171;">-0.50</span><br>'
            'Confidence bonus  →  <span style="color:#38bdf8;">+0.2 × conf</span><br>'
            'Perfect routing bonus  →  <span style="color:#4ade80;">+0.15</span><br>'
            'Misrouting penalty  →  <span style="color:#f87171;">-0.2 to -0.5</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        section_head("Evaluation & Metrics")
        st.markdown(
            '<div class="aks-about-card">'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.65);line-height:1.8;">'
            '<strong style="color:#60a5fa;">ML Metrics (250-query test set)</strong><br>'
            '• Overall Accuracy, Per-Source Precision, Recall, F1<br>'
            '• Confusion Matrix for source routing patterns<br>'
            '• 63 KG + 63 Tool + 63 LLM + 61 PDF test queries<br><br>'
            '<strong style="color:#f472b6;">LLM-as-Judge Evaluation</strong><br>'
            '• Correctness — Is the answer factually accurate?<br>'
            '• Relevance — Does it answer the question?<br>'
            '• Completeness — Is sufficient detail provided?<br>'
            '• Cost: ~$0.0001 per evaluation (AWS Nova Lite)'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with train_col:
        section_head("Training Pipeline")
        st.markdown(
            '<div class="aks-about-card">'
            '<div style="margin-bottom:14px;">'
            '<div style="font-size:0.82rem;font-weight:700;color:#a78bfa;margin-bottom:4px;">Phase 1 — Supervised Pre-training</div>'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.6);line-height:1.6;">'
            '600 labeled queries · 80/20 train/val split · 50 epochs with early stopping · Cross-entropy loss · 95.2% train / 88.9% val accuracy'
            '</div></div>'
            '<div style="margin-bottom:14px;">'
            '<div style="font-size:0.82rem;font-weight:700;color:#38bdf8;margin-bottom:4px;">Phase 2 — RL Fine-tuning</div>'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.6);line-height:1.6;">'
            'Epsilon-greedy exploration (ε: 1.0→0.05) · Experience replay buffer (2000 capacity, batch 32) · 50 training episodes · Avg reward: 0.520'
            '</div></div>'
            '<div>'
            '<div style="font-size:0.82rem;font-weight:700;color:#4ade80;margin-bottom:4px;">Phase 3 — Online Learning</div>'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.6);line-height:1.6;">'
            'Continuous improvement during real usage · Auto-retraining every 16 queries · Incremental model updates · S3 model versioning'
            '</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        section_head("Project Structure")
        st.markdown(
            '<div class="aks-about-card">'
            '<div style="font-size:0.8rem;color:rgba(255,255,255,0.6);font-family:monospace;line-height:1.7;">'
            '<span style="color:#a78bfa;">models/</span> — DQN agent, reward evaluator, replay buffer<br>'
            '<span style="color:#38bdf8;">knowledge_sources/</span> — KG, Tool/API, LLM, PDF<br>'
            '<span style="color:#4ade80;">scripts/</span> — Training, evaluation, dashboard<br>'
            '<span style="color:#fbbf24;">utils/</span> — LLM judge, S3 sync<br>'
            '<span style="color:#f472b6;">data/</span> — Hetionet, PDF store, RL models'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        section_head("UI & Technology Stack")
        st.markdown(
            '<div class="aks-about-card">'
            '<div style="font-size:0.82rem;color:rgba(255,255,255,0.65);line-height:1.9;">'
            '<strong style="color:#38bdf8;">Streamlit</strong> — Python-native reactive web framework<br>'
            '<strong style="color:#4ade80;">Plotly</strong> — Interactive charts (bar, heatmap, scatter, pie)<br>'
            '<strong style="color:#c084fc;">PyTorch</strong> — Neural network & RL training<br>'
            '<strong style="color:#fbbf24;">FAISS</strong> — Vector similarity search<br>'
            '<strong style="color:#f87171;">Sentence Transformers</strong> — Query embeddings (all-MiniLM-L6-v2)<br>'
            '<strong style="color:#f472b6;">CSS Glassmorphism</strong> — Dark theme with blur, gradients & 3D effects<br>'
            '<strong style="color:#60a5fa;">NetworkX</strong> — Knowledge graph data structure<br>'
            '<strong style="color:#a78bfa;">AWS Bedrock</strong> — LLM inference (Nova 2 Lite)'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Key Concepts
    section_head("Key RL Concepts")
    concepts = [
        ("🧠", "#a78bfa", "Deep Q-Network (DQN)", "Value-based RL algorithm for discrete action spaces — predicts Q-values for each source"),
        ("🔄", "#38bdf8", "Experience Replay", "Stores transitions (state, action, reward, next state) for stable off-policy learning"),
        ("🎲", "#4ade80", "Epsilon-Greedy", "Balances exploration vs exploitation — decays ε from 1.0 to 0.05 during training"),
        ("⚖️", "#fbbf24", "LLM-as-Judge", "Uses LLMs to evaluate AI system outputs for correctness, relevance, and completeness"),
    ]
    c1, c2 = st.columns(2)
    for i, (ico, clr, title, desc) in enumerate(concepts):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(
                f'<div class="aks-source-row" style="border-color:rgba(255,255,255,0.06);">'
                f'<div style="font-size:1.5rem;flex-shrink:0;">{ico}</div>'
                f'<div><div style="font-size:0.88rem;font-weight:700;color:{clr};margin-bottom:3px;">{title}</div>'
                f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.5);line-height:1.5;">{desc}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Milestones
    section_head("Project Milestones")
    milestones = [
        ("✅", "#4ade80", "Multi-source knowledge integration (4 sources)"),
        ("✅", "#4ade80", "Hetionet knowledge graph (47K nodes, 2.25M edges)"),
        ("✅", "#4ade80", "FAISS semantic search for PDF corpus (1,649 chunks)"),
        ("✅", "#4ade80", "Deep Q-Network (DQN) with experience replay"),
        ("✅", "#4ade80", "Two-phase training: supervised pre-training + RL fine-tuning"),
        ("✅", "#4ade80", "Automatic reward computation (no human feedback)"),
        ("✅", "#4ade80", "LLM-as-Judge quality evaluation"),
        ("✅", "#4ade80", "ML metrics: accuracy 86.7%, KG F1 0.924, Tool F1 0.961"),
        ("✅", "#4ade80", "Online learning with auto-retraining every 16 queries"),
        ("✅", "#4ade80", "S3 model versioning and team sync"),
        ("✅", "#4ade80", "Streamlit Web UI with glassmorphism dark theme"),
        ("✅", "#4ade80", "Interactive Plotly charts & real-time query testing"),
        ("🎯", "#a78bfa", "Multi-source ensemble (query multiple, synthesize)"),
        ("🎯", "#a78bfa", "Cost-aware routing (latency & API cost signals)"),
        ("🎯", "#a78bfa", "User feedback integration (thumbs up/down)"),
        ("🎯", "#a78bfa", "REST API deployment"),
    ]
    m1, m2 = st.columns(2)
    for i, (icon, clr, text) in enumerate(milestones):
        col = m1 if i % 2 == 0 else m2
        with col:
            st.markdown(
                f'<div class="aks-milestone">'
                f'<span style="font-size:1rem;">{icon}</span>'
                f'<span style="color:rgba(255,255,255,0.65);font-size:0.83rem;">{text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Course & Team
    section_head("Course & Team")
    st.markdown(
        '<div class="aks-about-card" style="text-align:center;">'
        '<div style="font-size:0.88rem;color:rgba(255,255,255,0.7);line-height:1.8;">'
        '<strong style="color:#a78bfa;">CSE 579</strong> — Knowledge Representation & Reasoning<br>'
        '<strong style="color:#38bdf8;">Arizona State University</strong> · Spring 2026<br>'
        '<span style="color:rgba(255,255,255,0.45);">Students: Rashi Sharma, Harsh Tita, Harpreet Kaur Brar, Shashwat Dwivedi, Sarthak Singh</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown(
        '<div style="text-align:center;padding:20px 0 8px;">'
        '<span style="color:rgba(255,255,255,0.25);font-size:0.78rem;">'
        
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )
