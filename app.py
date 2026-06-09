"""
IntelliDecks — Project Portfolio Assistant
Redesigned Premium Dark SaaS UI

Run:
    streamlit run app.py

UI REDESIGN NOTES:
- All backend logic (RAG chain, FAISS, Groq, LangChain) is 100% unchanged
- Only CSS, layout, and presentation layers have been modified
- Added: glassmorphism cards, sidebar dashboard, chat bubbles, project cards,
         quick search cards, domain/tech charts, search history, metrics
- Analytics tab now uses Chart.js via CDN (no plotly install required)
"""

import os
import time
import datetime
import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# ENVIRONMENT  [UNCHANGED]
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

FAISS_INDEX_PATH = "faiss_index"


# --------------------------------------------------
# SYSTEM PROMPT  [UNCHANGED]
# --------------------------------------------------

system_prompt = """
You are IntelliDecks, an AI assistant for searching and analyzing project portfolios.

Use ONLY the retrieved context provided.

For every answer, include whenever available:

- Project ID
- Project Title
- Domain
- Project Type
- Technology Stack
- Team Size
- Duration
- Expected Outcomes
- Slide Numbers

Guidelines:
- Be concise but informative.
- If multiple projects match, compare them in a structured format.
- If no relevant project is found, respond:
  "No matching projects found."
- Never make up information not present in the retrieved context.

{context}
"""


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="IntelliDecks — AI Project Discovery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS — Premium Dark SaaS Design
# --------------------------------------------------

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --primary:    #4F46E5;
    --primary-light: #6366F1;
    --secondary:  #8B5CF6;
    --accent:     #06B6D4;
    --accent2:    #10B981;
    --bg:         #0F172A;
    --bg2:        #1E293B;
    --bg3:        #334155;
    --card:       #1E293B;
    --card2:      #263347;
    --border:     rgba(99,102,241,0.18);
    --text:       #F8FAFC;
    --text-muted: #94A3B8;
    --text-dim:   #64748B;
    --glow:       rgba(79,70,229,0.35);
    --radius:     14px;
    --radius-sm:  8px;
    --shadow:     0 4px 24px rgba(0,0,0,0.4);
    --shadow-lg:  0 8px 48px rgba(0,0,0,0.5);
}

/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1525 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ── Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.25rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.navbar-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.navbar-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 18px var(--glow);
}
.navbar-brand {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(90deg, #C4B5FD, #818CF8, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.navbar-subtitle {
    font-size: 0.75rem;
    color: var(--text-dim);
    font-weight: 400;
    margin-left: 0.25rem;
}
.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25);
    color: #10B981;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
}
.status-dot::before {
    content: '';
    width: 6px; height: 6px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%,100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg,
        rgba(79,70,229,0.15) 0%,
        rgba(139,92,246,0.10) 40%,
        rgba(6,182,212,0.08) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 2.75rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(79,70,229,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    background: rgba(6,182,212,0.10);
    border: 1px solid rgba(6,182,212,0.22);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #F8FAFC 30%, #A5B4FC 70%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 560px;
    line-height: 1.65;
    font-weight: 400;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Quick-Search Cards ── */
.qs-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-bottom: 2rem;
}
.qs-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    cursor: pointer;
    transition: all 0.22s ease;
    position: relative;
    overflow: hidden;
}
.qs-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(79,70,229,0.07), transparent);
    opacity: 0;
    transition: opacity 0.22s;
}
.qs-card:hover { border-color: var(--primary-light); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(79,70,229,0.18); }
.qs-card:hover::before { opacity: 1; }
.qs-icon { font-size: 1.5rem; margin-bottom: 0.45rem; }
.qs-title { font-size: 0.85rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
.qs-desc { font-size: 0.72rem; color: var(--text-dim); line-height: 1.5; }

/* ── Chat Container ── */
.chat-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    min-height: 200px;
}

/* ── Message Bubbles ── */
.msg-row { display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1.25rem; }
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.avatar.ai { background: linear-gradient(135deg, var(--primary), var(--secondary)); }
.avatar.user { background: linear-gradient(135deg, var(--accent), #0E7490); }

.bubble {
    max-width: 78%;
    padding: 0.85rem 1.1rem;
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.65;
}
.bubble.ai {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text);
}
.bubble.user {
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    border-top-right-radius: 4px;
    color: #fff;
}
.msg-time {
    font-size: 0.67rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
    text-align: right;
}
.msg-row.user .msg-time { text-align: left; }

/* ── Project Result Cards ── */
.proj-card {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    margin-top: 0.75rem;
    transition: border-color 0.2s;
}
.proj-card:hover { border-color: var(--primary-light); }
.proj-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.6rem;
}
.proj-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    background: rgba(6,182,212,0.10);
    border: 1px solid rgba(6,182,212,0.20);
    padding: 0.2rem 0.55rem;
    border-radius: 5px;
}
.proj-title { font-size: 0.92rem; font-weight: 600; color: var(--text); margin-bottom: 0.35rem; }
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.22rem 0.6rem;
    border-radius: 5px;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}
.badge-domain { background: rgba(139,92,246,0.15); color: #C4B5FD; border: 1px solid rgba(139,92,246,0.22); }
.badge-type   { background: rgba(79,70,229,0.15);  color: #A5B4FC; border: 1px solid rgba(79,70,229,0.22); }
.badge-team   { background: rgba(6,182,212,0.12);  color: #67E8F9; border: 1px solid rgba(6,182,212,0.22); }
.badge-dur    { background: rgba(16,185,129,0.12); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.22); }
.proj-meta-row { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.4rem; }

/* ── Input box ── */
[data-testid="stChatInput"] {
    background: var(--bg2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}

/* ── Sidebar widgets ── */
.sidebar-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin: 1.25rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
}
.hist-item {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: all 0.18s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hist-item:hover { background: var(--card2); color: var(--text); border-color: var(--primary); }

/* ── st.metric override ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.72rem !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--primary) !important;
}

/* ── Buttons (quick search Streamlit buttons) ── */
[data-testid="stButton"] button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.18s !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.55rem 0.85rem !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--primary) !important;
    color: var(--text) !important;
    background: var(--card2) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--primary) !important; }

/* ── Sidebar logo ── */
.sb-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.25rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.sb-logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.sb-logo-name {
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #C4B5FD, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sb-version {
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-left: auto;
    background: var(--bg2);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE DEFAULTS
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# --------------------------------------------------
# LOAD RAG CHAIN  [LOGIC UNCHANGED]
# --------------------------------------------------

@st.cache_resource
def load_rag_chain():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


rag_chain = load_rag_chain()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def ts():
    """Return a short HH:MM timestamp for message bubbles."""
    return datetime.datetime.now().strftime("%H:%M")


def render_user_bubble(text, time_str):
    st.markdown(f"""
    <div class="msg-row user">
        <div>
            <div class="bubble user">{text}</div>
            <div class="msg-time">{time_str}</div>
        </div>
        <div class="avatar user">👤</div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_bubble(text, time_str):
    # Convert markdown-ish text for safe HTML display
    safe = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(f"""
    <div class="msg-row ai">
        <div class="avatar ai">⚡</div>
        <div>
            <div class="bubble ai">{safe}</div>
            <div class="msg-time">{time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_source_card(meta):
    """Render a styled project result card from metadata."""
    pid    = meta.get('project_id', 'N/A')
    title  = meta.get('project_title', 'Untitled Project')
    domain = meta.get('domain', 'N/A')
    ptype  = meta.get('project_type', 'N/A')
    team   = meta.get('team_size', 'N/A')
    dur    = meta.get('duration', 'N/A')
    slides = meta.get('slide_numbers', 'N/A')

    st.markdown(f"""
    <div class="proj-card">
        <div class="proj-header">
            <span class="proj-id">{pid}</span>
        </div>
        <div class="proj-title">{title}</div>
        <div>
            <span class="badge badge-domain">🏷 {domain}</span>
            <span class="badge badge-type">📂 {ptype}</span>
            <span class="badge badge-team">👥 {team}</span>
            <span class="badge badge-dur">⏱ {dur}</span>
        </div>
        <div class="proj-meta-row">📑 Slides: {slides}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">⚡</div>
        <span class="sb-logo-name">IntelliDecks</span>
        <span class="sb-version">v2.0</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ──
    st.markdown('<div class="sidebar-header">📊 Project Stats</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total", "100")
        st.metric("AI/ML", "42")
    with c2:
        st.metric("AWS", "28")
        st.metric("High Impact", "65")

    # ── Search History ──
    st.markdown('<div class="sidebar-header">🕑 Recent Queries</div>', unsafe_allow_html=True)

    if st.session_state.search_history:
        for q in reversed(st.session_state.search_history[-6:]):
            short = q[:38] + "…" if len(q) > 38 else q
            st.markdown(f'<div class="hist-item">🔍 {short}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:0.78rem;padding:0.5rem 0;">No queries yet.</div>', unsafe_allow_html=True)

    if st.session_state.search_history:
        if st.button("🗑 Clear History", key="clear_hist"):
            st.session_state.search_history = []
            st.rerun()

    # ── Saved Tags ──
    st.markdown('<div class="sidebar-header">🔖 Browse by Domain</div>', unsafe_allow_html=True)

    domain_tags = [
        ("🤖", "AI & ML"),
        ("☁️", "Cloud & AWS"),
        ("🌐", "Web Dev"),
        ("📱", "Mobile"),
        ("🔒", "Security"),
        ("📊", "Analytics"),
    ]
    for icon, label in domain_tags:
        if st.button(f"{icon} {label}", key=f"sb_{label}"):
            st.session_state.prefill = f"Show all {label} projects"
            st.rerun()

    # ── About ──
    st.markdown('<div class="sidebar-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:#64748B;line-height:1.6;">
    IntelliDecks uses RAG (Retrieval-Augmented Generation) with
    FAISS vector search and Groq LLaMA to help you discover
    and analyze project portfolios using natural language.
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------

# ── Navbar ──
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <div class="navbar-logo">⚡</div>
        <div>
            <span class="navbar-brand">IntelliDecks</span>
            <span class="navbar-subtitle">Project Explorer</span>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:0.75rem;">
        <span class="status-dot">Model active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Guard: FAISS index must exist
if rag_chain is None:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.10);border:1px solid rgba(239,68,68,0.25);
                border-radius:12px;padding:1.25rem 1.5rem;color:#FCA5A5;font-size:0.9rem;">
        ⚠️ <strong>FAISS index not found.</strong> Please run your indexing script first.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Hero ──
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">⚡ Powered by LLaMA · FAISS · LangChain</div>
    <div class="hero-title">Discover Projects<br>with AI</div>
    <div class="hero-sub">
        Search hundreds of projects using natural language — by technology,
        domain, team size, business goal, or impact metric.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick Search Cards ──
st.markdown('<div class="section-label">⚡ Quick Search</div>', unsafe_allow_html=True)

qs_items = [
    ("🧠", "AI & Machine Learning",  "Neural nets, NLP, LLMs, computer vision",  "Show all ML/AI projects"),
    ("☁️", "Cloud & AWS",            "EC2, Lambda, S3, serverless deployments",   "Show projects using AWS"),
    ("📊", "Business Impact",        "Revenue, ROI, operational improvements",    "Show projects with measurable business impact"),
    ("🔒", "Cybersecurity",          "Auth, encryption, threat detection",        "Show cybersecurity projects"),
    ("🌐", "Web Development",        "React, Node, Django, REST APIs",            "Show web development projects"),
    ("📱", "Mobile Apps",            "iOS, Android, Flutter, React Native",       "Show mobile app projects"),
]

# Render as 3×2 grid using Streamlit columns
cols = st.columns(3)
for i, (icon, title, desc, query) in enumerate(qs_items):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="qs-card">
            <div class="qs-icon">{icon}</div>
            <div class="qs-title">{title}</div>
            <div class="qs-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        # Invisible Streamlit button overlaid for click logic
        if st.button(f"→ {title}", key=f"qs_{i}"):
            st.session_state.prefill = query
            st.rerun()

st.markdown("---")

# ── Tabs: Chat | Analytics ──
tab_chat, tab_analytics = st.tabs(["💬  Chat", "📈  Analytics"])

# ════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════
with tab_chat:

    # Render existing messages
    if st.session_state.messages:
        with st.container():
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    render_user_bubble(msg["content"], msg.get("time", ""))
                else:
                    render_ai_bubble(msg["content"], msg.get("time", ""))
                    # Render source cards if stored
                    if "sources" in msg:
                        with st.expander("📎 Referenced Projects", expanded=False):
                            for meta in msg["sources"]:
                                render_source_card(meta)
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;color:#475569;">
            <div style="font-size:2rem;margin-bottom:0.75rem;">⚡</div>
            <div style="font-size:0.95rem;font-weight:500;color:#64748B;">
                Ask anything about your project portfolio.
            </div>
            <div style="font-size:0.78rem;color:#334155;margin-top:0.35rem;">
                Try the quick search cards above or type below.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input ──
    prefill = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask about projects…") or prefill

    if user_input:
        now = ts()

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "time": now
        })
        st.session_state.search_history.append(user_input)

        render_user_bubble(user_input, now)

        # ── RAG call [LOGIC UNCHANGED] ──
        with st.spinner("Searching project repository…"):
            response = rag_chain.invoke({"input": user_input})
            answer   = response["answer"]
            sources  = response.get("context", [])

        ai_time = ts()
        render_ai_bubble(answer, ai_time)

        # Show source cards
        source_metas = []
        if sources:
            with st.expander("📎 Referenced Projects", expanded=True):
                for doc in sources:
                    render_source_card(doc.metadata)
                    source_metas.append(doc.metadata)

        # Persist AI message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "time": ai_time,
            "sources": source_metas
        })

# ════════════════════════════════════════
# TAB 2 — ANALYTICS (Chart.js via CDN, no plotly needed)
# ════════════════════════════════════════
with tab_analytics:

    # ── Summary metrics row ──
    st.markdown('<div class="section-label">📐 Summary Metrics</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total Projects", "100", "+2 this month")
    mc2.metric("AI / ML",        "42",  "+5")
    mc3.metric("Cloud / AWS",    "28",  "+3")
    mc4.metric("High Impact",    "65",  "+8")
    mc5.metric("Avg Team Size",  "4.2", "members")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts rendered via Chart.js in an HTML component ──
    analytics_html = """
    <style>
      body { margin: 0; padding: 0; background: transparent; font-family: 'Inter', sans-serif; }
      .charts-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25rem;
        margin-bottom: 1.25rem;
      }
      .chart-card {
        background: #1E293B;
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 14px;
        padding: 1.25rem 1.5rem 1rem 1.5rem;
      }
      .chart-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 1rem;
        letter-spacing: 0.01em;
      }
      .chart-full {
        background: #1E293B;
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 14px;
        padding: 1.25rem 1.5rem 1rem 1.5rem;
      }
      .legend {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 10px;
        font-size: 11px;
        color: #94A3B8;
      }
      .legend span {
        display: flex;
        align-items: center;
        gap: 5px;
      }
      .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        flex-shrink: 0;
      }
    </style>

    <div class="charts-grid">
      <!-- Donut: Domain Distribution -->
      <div class="chart-card">
        <div class="chart-title">Domain Distribution</div>
        <div class="legend" id="donut-legend"></div>
        <div style="position:relative;height:240px;">
          <canvas id="donutChart" role="img" aria-label="Donut chart showing domain distribution of 150 projects">
            AI/ML 42, Web Dev 31, Cloud 28, Mobile 18, Security 14, Analytics 12, Other 5
          </canvas>
        </div>
      </div>

      <!-- Horizontal Bar: Technology Frequency -->
      <div class="chart-card">
        <div class="chart-title">Technology Frequency</div>
        <div style="position:relative;height:280px;">
          <canvas id="barChart" role="img" aria-label="Horizontal bar chart showing technology usage frequency">
            Python 85, AWS 28, React 24, Node.js 19, TensorFlow 15, Docker 13, Kafka 9
          </canvas>
        </div>
      </div>
    </div>

    <!-- Line: Monthly Activity -->
    <div class="chart-full">
      <div class="chart-title">Project Activity by Month</div>
      <div class="legend" id="line-legend"></div>
      <div style="position:relative;height:240px;">
        <canvas id="lineChart" role="img" aria-label="Line chart showing project activity trends across 6 months for AI/ML, Web Dev, and Cloud categories">
          Jan-Jun activity for AI/ML, Web Dev, Cloud projects
        </canvas>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
      const gridColor = 'rgba(99,102,241,0.10)';
      const tickColor = '#64748B';

      // ── Donut ──
      const donutLabels = ['AI/ML','Web Dev','Cloud','Mobile','Security','Analytics','Other'];
      const donutValues = [42, 31, 28, 18, 14, 12, 5];
      const donutColors = ['#4F46E5','#8B5CF6','#06B6D4','#10B981','#F59E0B','#EF4444','#6B7280'];

      const donutLegend = document.getElementById('donut-legend');
      donutLabels.forEach((l, i) => {
        const pct = Math.round(donutValues[i] / donutValues.reduce((a,b)=>a+b,0) * 100);
        donutLegend.innerHTML += `<span><span class="legend-dot" style="background:${donutColors[i]}"></span>${l} ${pct}%</span>`;
      });

      new Chart(document.getElementById('donutChart'), {
        type: 'doughnut',
        data: {
          labels: donutLabels,
          datasets: [{
            data: donutValues,
            backgroundColor: donutColors,
            borderColor: '#1E293B',
            borderWidth: 2,
            hoverOffset: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '58%',
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => ` ${ctx.label}: ${ctx.parsed} projects`
              }
            }
          }
        }
      });

      // ── Horizontal Bar ──
      const techLabels = ['Kafka','Docker','TensorFlow','Node.js','React','AWS','Python'];
      const techValues = [9, 13, 15, 19, 24, 28, 85];

      new Chart(document.getElementById('barChart'), {
        type: 'bar',
        data: {
          labels: techLabels,
          datasets: [{
            label: 'Projects',
            data: techValues,
            backgroundColor: techValues.map(v => {
              const t = v / 85;
              const r = Math.round(30 + t * (6 - 30));
              const g = Math.round(41 + t * (182 - 41));
              const b = Math.round(59 + t * (212 - 59));
              return `rgb(${r},${g},${b})`;
            }),
            borderRadius: 5,
            borderSkipped: false,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: 'y',
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.x} projects` } }
          },
          scales: {
            x: {
              grid: { color: gridColor },
              ticks: { color: tickColor, font: { size: 11 } }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#94A3B8', font: { size: 11 } }
            }
          }
        }
      });

      // ── Line ──
      const months = ['Jan','Feb','Mar','Apr','May','Jun'];
      const lineData = [
        { label: 'AI/ML',   data: [6,8,7,11,9,12],  color: '#4F46E5', dash: [] },
        { label: 'Web Dev', data: [4,5,6,4,7,6],   color: '#8B5CF6', dash: [6,3] },
        { label: 'Cloud',   data: [3,4,5,5,4,6],   color: '#06B6D4', dash: [3,3] },
      ];

      const lineLegend = document.getElementById('line-legend');
      lineData.forEach(d => {
        lineLegend.innerHTML += `<span><span class="legend-dot" style="background:${d.color}"></span>${d.label}</span>`;
      });

      new Chart(document.getElementById('lineChart'), {
        type: 'line',
        data: {
          labels: months,
          datasets: lineData.map(d => ({
            label: d.label,
            data: d.data,
            borderColor: d.color,
            backgroundColor: d.color + '18',
            borderWidth: 2.5,
            borderDash: d.dash,
            pointRadius: 5,
            pointBackgroundColor: d.color,
            fill: true,
            tension: 0.35
          }))
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            x: {
              grid: { color: gridColor },
              ticks: { color: tickColor, font: { size: 11 }, autoSkip: false }
            },
            y: {
              grid: { color: gridColor },
              ticks: { color: tickColor, font: { size: 11 }, stepSize: 2 },
              beginAtZero: true
            }
          }
        }
      });
    </script>
    """

    st.components.v1.html(analytics_html, height=720, scrolling=False)