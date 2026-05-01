"""
custom_css.py
-------------
Drop-in visual overhaul for the Grade 9 Linear Equations Tutor.

Usage in app.py:
    from custom_css import CUSTOM_CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

Replace the old CUSTOM_CSS string and st.markdown call entirely with the above.
"""

CUSTOM_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── CSS variables ───────────────────────────────────────── */
:root {
    --c-bg:        #F4F6F8;
    --c-surface:   #FFFFFF;
    --c-primary:   #2F4858;
    --c-primary2:  #3E6374;
    --c-accent:    #4CAF82;
    --c-accent2:   #86A873;
    --c-warn:      #E8825A;
    --c-muted:     #6B7280;
    --c-soft:      #DDE3EA;
    --c-card-border: rgba(47,72,88,0.10);
    --c-shadow:    0 2px 12px rgba(47,72,88,0.08);
    --c-shadow-lg: 0 8px 32px rgba(47,72,88,0.13);
    --radius:      14px;
    --radius-sm:   8px;
}

/* ── Base ────────────────────────────────────────────────── */
html, body, [class*="stApp"] {
    background: var(--c-bg) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: var(--c-primary);
}
h1,h2,h3,h4 { color: var(--c-primary) !important; letter-spacing:-0.02em; }

/* ── Animated hero ───────────────────────────────────────── */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #1e3a4c 0%, #2F4858 50%, #3E6374 100%);
    background-size: 200% 200%;
    animation: heroShift 8s ease infinite;
    color: white;
    padding: 32px 36px;
    border-radius: 20px;
    margin-bottom: 22px;
    box-shadow: var(--c-shadow-lg);
}
@keyframes heroShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* floating math symbols */
.hero::before {
    content: "y  m  x  b  +  =  Δ  ∫  √";
    position: absolute;
    top: -10px; left: 0; right: 0;
    font-size: 4rem;
    color: rgba(255,255,255,0.04);
    letter-spacing: 18px;
    white-space: nowrap;
    animation: floatSymbols 18s linear infinite;
    pointer-events: none;
    font-weight: 700;
}
@keyframes floatSymbols {
    from { transform: translateX(0); }
    to   { transform: translateX(-40%); }
}

.hero h1 {
    color: white !important;
    margin: 0 0 6px;
    font-size: 1.8rem;
    font-weight: 700;
    position: relative;
}
.hero p  {
    color: rgba(216,226,232,0.92);
    margin: 0;
    font-size: 1rem;
    position: relative;
}

/* ── Page load fade-in ───────────────────────────────────── */
section.main > div:first-child {
    animation: fadeUp 0.45s ease both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Metric cards ────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--c-surface);
    border: 1px solid var(--c-card-border);
    border-top: 3px solid var(--c-accent);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--c-shadow);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--c-shadow-lg);
}
div[data-testid="stMetricLabel"] { color: var(--c-muted) !important; font-size: 0.82rem; font-weight:500; text-transform:uppercase; letter-spacing:0.04em; }
div[data-testid="stMetricValue"] { color: var(--c-primary) !important; font-weight: 700; font-size: 1.6rem; }

/* Accent colour varies by column position (nth-child cycles) */
div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] { border-top-color: #4CAF82; }
div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] { border-top-color: #378ADD; }
div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] { border-top-color: #E8825A; }
div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"] { border-top-color: #9B6DD6; }

/* ── Card borders ────────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-card-border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--c-shadow);
    transition: box-shadow 0.18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--c-shadow-lg);
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--c-primary) 0%, var(--c-primary2) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
    box-shadow: 0 2px 8px rgba(47,72,88,0.18) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(47,72,88,0.25) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active {
    transform: translateY(0px) scale(0.97) !important;
    box-shadow: 0 1px 4px rgba(47,72,88,0.15) !important;
}

/* ── Submit button — green accent ────────────────────────── */
/* targets the first button in 3-col layout (Submit) */
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, #3A9E6A 0%, #4CAF82 100%) !important;
}

/* ── Hint button — amber accent ──────────────────────────── */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: linear-gradient(135deg, #C07830 0%, #E09050 100%) !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [role="tablist"] {
    border-bottom: 2px solid var(--c-soft);
    gap: 4px;
}
.stTabs [role="tablist"] button {
    color: var(--c-muted) !important;
    font-weight: 500;
    border-radius: 8px 8px 0 0;
    transition: color 0.15s, background 0.15s;
    padding: 8px 18px;
}
.stTabs [role="tablist"] button:hover {
    background: rgba(47,72,88,0.06);
    color: var(--c-primary) !important;
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    color: var(--c-primary) !important;
    font-weight: 700;
    border-bottom: 3px solid var(--c-accent) !important;
    background: rgba(76,175,130,0.07);
}

/* ── Alerts / feedback ───────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 500;
}

/* ── Text input ──────────────────────────────────────────── */
input[type="text"], textarea {
    border-radius: 10px !important;
    border: 1.5px solid var(--c-soft) !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--c-accent) !important;
    box-shadow: 0 0 0 3px rgba(76,175,130,0.15) !important;
}

/* ── Select boxes ────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a4c 0%, #2F4858 100%) !important;
    border-right: none !important;
    box-shadow: 2px 0 20px rgba(0,0,0,0.12);
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label * {
    color: #D8E8F0 !important;
}
section[data-testid="stSidebar"] h3 {
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* Sidebar nav radio buttons */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 9px;
    padding: 10px 14px;
    margin-bottom: 4px;
    display: block;
    transition: background 0.15s;
    cursor: pointer;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.10) !important;
}
/* Selected nav item */
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(76,175,130,0.22) !important;
    border-color: rgba(76,175,130,0.5) !important;
    color: #A8DFBE !important;
    font-weight: 600;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stButton > button * {
    background: rgba(255,255,255,0.10) !important;
    color: #D8E8F0 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
}

/* ── Topic chip badges ───────────────────────────────────── */
.topic-chip {
    display: inline-block;
    background: rgba(47,72,88,0.08);
    color: var(--c-primary);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    border: 1px solid var(--c-card-border);
    letter-spacing: 0.02em;
}

/* ── Expander ────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid var(--c-card-border) !important;
    border-radius: var(--radius) !important;
    background: var(--c-surface) !important;
}

/* ── Answer feedback animations ─────────────────────────── */
/* Injected via JS in sounds.py — these classes are toggled dynamically */
@keyframes correctPulse {
    0%   { box-shadow: 0 0 0 0 rgba(76,175,130,0.5); }
    50%  { box-shadow: 0 0 0 14px rgba(76,175,130,0); }
    100% { box-shadow: 0 0 0 0 rgba(76,175,130,0); }
}
@keyframes wrongShake {
    0%,100% { transform: translateX(0); }
    20%     { transform: translateX(-7px); }
    40%     { transform: translateX(7px); }
    60%     { transform: translateX(-5px); }
    80%     { transform: translateX(5px); }
}
.answer-correct {
    animation: correctPulse 0.7s ease !important;
}
.answer-wrong {
    animation: wrongShake 0.45s ease !important;
}

/* ── Confetti canvas (injected by sounds.py) ─────────────── */
#confetti-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 9999;
}

/* ── Progress bar in mastery strip ──────────────────────── */
.mastery-bar-wrap {
    background: var(--c-soft);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.mastery-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4CAF82, #86A873);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── General helper ──────────────────────────────────────── */
.small-muted { color: var(--c-muted); font-size: 0.85rem; }

/* ── Dataframe / table ───────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
    box-shadow: var(--c-shadow);
}

/* ── Smooth divider ──────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--c-soft) !important;
    margin: 1rem 0 !important;
}
</style>
"""
