"""
app.py — Grade 9 Linear Equations Tutor
========================================

What this app does
------------------
A friendly, dashboard-style Streamlit tutor for Grade 9 math focused on
linear equations in the form y = mx + b: slope, y-intercept, graphing,
and word problems. It generates practice problems, offers scaffolded
hints, shows step-by-step solutions, and tracks mastery using a
scikit-learn Random Forest classifier.

How to install
--------------
    pip install -r requirements.txt

How to run locally
------------------
    streamlit run app.py

How to deploy on Streamlit Community Cloud
------------------------------------------
1. Push this folder (app.py, requirements.txt, content.py, ml_model.py,
   utils.py) to a public GitHub repo.
2. On https://share.streamlit.io, click "New app" and point it at your
   repo and `app.py`.
3. (Optional) In the app's "Secrets" settings, paste:

       PERPLEXITY_API_KEY = "your_key_here"

   The app also works without any API key: it automatically falls back
   to built-in local explanations, hints, and practice problems.

Secrets example
---------------
    PERPLEXITY_API_KEY = "your_key_here"
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

import db
from content import TOPICS, generate_problem, get_explanation
from ml_model import (
    CLASS_LABELS,
    compute_features,
    recommend_next_action,
    train_mastery_model,
)
from bkt import BKTTracker
from utils import check_answer, get_api_key, perplexity_chat, plot_line, plot_system
from custom_css import CUSTOM_CSS
from sounds import play_correct, play_wrong, play_hint, play_click, play_new_problem


# ---------------------------------------------------------------------------
# EQAO (level_1..level_4) → friendly (low/medium/high) adapter
# Keeps the existing UI, CSS, and Supabase schema unchanged while the new
# MasteryPipeline outputs the Ontario EQAO 4-level rubric internally.
# ---------------------------------------------------------------------------
EQAO_TO_FRIENDLY: Dict[str, str] = {
    "level_1": "low",
    "level_2": "low",
    "level_3": "medium",
    "level_4": "high",
}

# Friendly bands for the dashboard probability bars
FRIENDLY_BANDS = ["low", "medium", "high"]


def eqao_probs_to_friendly(probs: Dict[str, float]) -> Dict[str, float]:
    """Aggregate level_1..level_4 probabilities into low/medium/high buckets."""
    out = {b: 0.0 for b in FRIENDLY_BANDS}
    for lvl, p in probs.items():
        band = EQAO_TO_FRIENDLY.get(lvl)
        if band:
            out[band] += float(p)
    return out


# ---------------------------------------------------------------------------
# Page config + global styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OpenMath AI",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization (safe, idempotent)
# ---------------------------------------------------------------------------

DEFAULT_STATE: Dict[str, Any] = {
    "current_topic": "slope",
    "current_difficulty": "easy",
    "current_problem": None,          # Problem object
    "problem_started_at": None,       # time.time() when shown
    "hints_revealed": 0,              # for current problem
    "feedback": None,                 # last feedback string
    "feedback_correct": None,         # bool/None
    "attempt_history": [],            # list of dicts per attempt
    "total_attempts": 0,
    "correct_attempts": 0,
    "incorrect_attempts": 0,
    "hint_count": 0,                  # total hints across session
    "response_times": [],             # seconds per attempt
    "recent_correctness": 0.0,        # cached fraction over last N
    "mastery_prediction": "low",      # friendly label (low/medium/high) shown in UI
    "mastery_probs": {"low": 1.0, "medium": 0.0, "high": 0.0},  # friendly aggregated
    "mastery_probs_eqao": {"level_1": 1.0, "level_2": 0.0, "level_3": 0.0, "level_4": 0.0},
    "bkt_state": "",                  # JSON-serialised BKTTracker state
    "recommended_next_action": "Try a couple of easy warm-up problems to get started.",
    "answer_input_key": 0,            # used to reset the answer input widget
    "free_response_answer": "",       # last free-response tutor reply
    "free_response_source": "",       # "api" | "local" | ""
}


def init_state() -> None:
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------------
# Cloud sync helpers (no-ops if Supabase isn't configured or user not signed in)
# ---------------------------------------------------------------------------

PERSIST_KEYS = (
    "attempt_history",
    "total_attempts",
    "correct_attempts",
    "incorrect_attempts",
    "hint_count",
    "response_times",
    "mastery_prediction",
    "mastery_probs",
    "bkt_state",
)


def _current_user_id() -> str:
    u = db.current_user()
    return u["id"] if u else ""


def hydrate_from_cloud() -> None:
    """Load a signed-in user's saved progress into session_state once per session."""
    if st.session_state.get("_hydrated"):
        return
    uid = _current_user_id()
    if not uid:
        return
    row = db.load_progress(uid)
    for k in PERSIST_KEYS:
        if k in row:
            st.session_state[k] = row[k]
    # Rehydrate the in-memory BKT tracker from the JSON blob, if present.
    bkt_json = st.session_state.get("bkt_state") or ""
    try:
        if bkt_json:
            st.session_state["_bkt_tracker"] = BKTTracker.from_json(bkt_json)
        else:
            st.session_state["_bkt_tracker"] = BKTTracker()
    except Exception:
        st.session_state["_bkt_tracker"] = BKTTracker()
    st.session_state["_hydrated"] = True


def sync_to_cloud() -> None:
    """Upsert current progress to Supabase. Silent on failure."""
    uid = _current_user_id()
    if not uid:
        return
    payload = {k: st.session_state.get(k) for k in PERSIST_KEYS}
    db.save_progress(uid, payload)


# ---------------------------------------------------------------------------
# Auth gate — shown before the main app when Supabase is configured
# ---------------------------------------------------------------------------

def render_auth_gate() -> None:
    """Show a sign-in / sign-up form. Called only when no user is signed in."""
    st.markdown(
        """
        <div class="hero">
          <h1>📐 OpenMath AI</h1>
          <p>Your personal linear equations tutor. Create a free account to save your mastery progress and problem history across sessions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])

        with tab_in:
            with st.form("signin_form", clear_on_submit=False):
                email = st.text_input("Email", key="signin_email")
                password = st.text_input("Password", type="password", key="signin_pw")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Enter your email and password.")
                else:
                    ok, msg = db.sign_in(email.strip(), password)
                    if ok:
                        st.session_state.pop("_hydrated", None)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_up:
            with st.form("signup_form", clear_on_submit=False):
                email_u = st.text_input("Email", key="signup_email")
                name_u = st.text_input("Display name (optional)", key="signup_name")
                password_u = st.text_input(
                    "Password (6+ characters)", type="password", key="signup_pw"
                )
                submitted_u = st.form_submit_button(
                    "Create account", use_container_width=True
                )
            if submitted_u:
                if not email_u or not password_u:
                    st.error("Email and password are required.")
                elif len(password_u) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = db.sign_up(
                        email_u.strip(), password_u, name_u.strip()
                    )
                    if ok:
                        st.session_state.pop("_hydrated", None)
                        st.success(msg)
                        if db.current_user():
                            st.rerun()
                    else:
                        st.error(msg)

        st.caption(
            "Your email is only used to sign you in and sync your progress. "
            "Nothing is shared."
        )


# ---------------------------------------------------------------------------
# Model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    return train_mastery_model()


MODEL = load_model()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_bkt_tracker() -> BKTTracker:
    """Lazy accessor; rebuilds from JSON if missing (e.g. after a reset)."""
    tr = st.session_state.get("_bkt_tracker")
    if isinstance(tr, BKTTracker):
        return tr
    blob = st.session_state.get("bkt_state") or ""
    try:
        tr = BKTTracker.from_json(blob) if blob else BKTTracker()
    except Exception:
        tr = BKTTracker()
    st.session_state["_bkt_tracker"] = tr
    return tr


def refresh_mastery() -> None:
    feats = compute_features(
        st.session_state["attempt_history"],
        st.session_state["hint_count"],
        st.session_state["response_times"],
    )
    eqao_label, eqao_probs = MODEL.predict(feats)  # e.g. "level_3", {level_1..level_4: p}
    friendly_label = EQAO_TO_FRIENDLY.get(eqao_label, "low")
    friendly_probs = eqao_probs_to_friendly(eqao_probs)

    st.session_state["recent_correctness"] = feats["recent_correctness"]
    st.session_state["mastery_prediction"] = friendly_label
    st.session_state["mastery_probs"] = friendly_probs
    st.session_state["mastery_probs_eqao"] = eqao_probs

    # Warm-start BKT priors from the RF output (once per topic per session).
    tracker = get_bkt_tracker()
    current_topic = st.session_state.get("current_topic") or "slope"
    tracker.warm_start_from_rf(current_topic, eqao_probs)
    st.session_state["bkt_state"] = tracker.to_json()

    st.session_state["recommended_next_action"] = recommend_next_action(eqao_label, feats)


def new_problem(topic: str, difficulty: str) -> None:
    st.session_state["current_topic"] = topic
    st.session_state["current_difficulty"] = difficulty
    st.session_state["current_problem"] = generate_problem(topic, difficulty)
    st.session_state["problem_started_at"] = time.time()
    st.session_state["hints_revealed"] = 0
    st.session_state["feedback"] = None
    st.session_state["feedback_correct"] = None
    st.session_state["answer_input_key"] += 1  # reset input widget


def record_attempt(correct: bool) -> None:
    started = st.session_state.get("problem_started_at") or time.time()
    elapsed = max(0.5, time.time() - started)
    prob = st.session_state["current_problem"]
    topic = prob.topic if prob is not None else st.session_state["current_topic"]
    difficulty = prob.difficulty if prob is not None else st.session_state["current_difficulty"]

    st.session_state["attempt_history"].append({
        "topic": topic,
        "difficulty": difficulty,
        "correct": bool(correct),
        "hints_used": st.session_state["hints_revealed"],
        "seconds": round(elapsed, 1),
        "timestamp": time.strftime("%H:%M:%S"),
    })
    st.session_state["response_times"].append(elapsed)
    st.session_state["total_attempts"] += 1
    if correct:
        st.session_state["correct_attempts"] += 1
    else:
        st.session_state["incorrect_attempts"] += 1
    refresh_mastery()
    # BKT hidden-Markov update for this topic (RF warm-start happens in refresh_mastery).
    tracker = get_bkt_tracker()
    tracker.update(topic, bool(correct))
    st.session_state["bkt_state"] = tracker.to_json()
    sync_to_cloud()


def reveal_hint() -> None:
    prob = st.session_state["current_problem"]
    if prob is None:
        return
    if st.session_state["hints_revealed"] < len(prob.hints):
        st.session_state["hints_revealed"] += 1
        st.session_state["hint_count"] += 1
        refresh_mastery()
        sync_to_cloud()


def reset_session() -> None:
    # Preserve auth keys so a reset doesn't sign the user out.
    keep = {k: st.session_state[k] for k in (
        "_supabase_client", "_supabase_session", "_supabase_user", "_hydrated"
    ) if k in st.session_state}
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.session_state["answer_input_key"] += 1
    # Fresh BKT tracker on reset
    st.session_state["_bkt_tracker"] = BKTTracker()
    st.session_state["bkt_state"] = st.session_state["_bkt_tracker"].to_json()
    for k, v in keep.items():
        st.session_state[k] = v
    sync_to_cloud()  # persist the reset so it sticks across devices


# ---------------------------------------------------------------------------
# Auth gate — runs before anything else when Supabase is configured
# ---------------------------------------------------------------------------

if db.is_enabled() and db.current_user() is None:
    render_auth_gate()
    st.stop()

# Once signed in (or if auth isn't configured), pull saved progress in.
hydrate_from_cloud()


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📐 OpenMath AI")
    st.caption("Grade 9 · y = mx + b")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📚 Learn & Practice", "💬 Ask a Question", "📊 Dashboard", "👩‍🏫 Teacher View"],
        label_visibility="collapsed",
    )

    st.divider()

    # Account panel — only shown when Supabase is configured.
    _user = db.current_user()
    if _user is not None:
        _name = _user.get("display_name") or _user.get("email") or "Student"
        st.markdown(f"**Signed in as**")
        st.markdown(f"{_name}")
        if st.button("Sign out", use_container_width=True):
            db.sign_out()
            st.session_state.pop("_hydrated", None)
            st.rerun()
        st.divider()

    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>📐 OpenMath AI</h1>
      <p>Your personal linear equations tutor. Master <strong>y = mx + b</strong> —
         slope, intercepts, graphing, systems of equations, and word problems —
         with adaptive practice, scaffolded hints, and mastery tracking powered by machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Shared top-of-page progress strip
# ---------------------------------------------------------------------------

def progress_strip() -> None:
    c1, c2, c3, c4 = st.columns(4)
    total = st.session_state["total_attempts"]
    correct = st.session_state["correct_attempts"]
    acc = (correct / total * 100.0) if total else 0.0

    # BKT overall readiness drives the headline mastery number.
    try:
        _tracker = get_bkt_tracker()
        bkt_overall = _tracker.overall_readiness_score(threshold=0.75)
    except Exception:
        bkt_overall = 0.0

    c1.metric("Attempts", total)
    c2.metric("Correct", correct)
    c3.metric("Accuracy", f"{acc:.0f}%")
    c4.metric("Mastery", f"{bkt_overall:.0%}")

    # Animated bar driven by BKT overall readiness (proportion of topics
    # where P(mastered) >= 0.75). This is the headline progress signal.
    bar_pct = int(round(min(max(bkt_overall, 0.0), 1.0) * 100))
    st.markdown(
        f"""
        <div class="mastery-bar-wrap">
          <div class="mastery-bar-fill" style="width:{bar_pct}%"></div>
        </div>
        <p style="text-align:right;font-size:0.78rem;color:#6B7280;margin:2px 0 0;">
          Linear equations mastery (Bayesian Knowledge Tracing)
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def page_home() -> None:
    progress_strip()
    st.write("")

    left, right = st.columns([1.4, 1])

    with left:
        with st.container(border=True):
            st.subheader("Welcome 👋")
            st.write(
                "This tutor helps you understand **linear equations** — lines written in the form "
                "**y = mx + b**. Pick a topic on the right to read a short lesson, then head to "
                "**Learn & Practice** to try problems with hints and step-by-step solutions."
            )
            st.markdown(
                "- 📈 **Slope** — what *m* means and how to calculate it\n"
                "- 🎯 **y-intercept** — what *b* means on a graph and in an equation\n"
                "- 📐 **Graphing** — drawing a line from slope-intercept form\n"
                "- ❌ **Systems of equations** — finding where two lines meet\n"
                "- 📝 **Word problems** — turning real situations into equations"
            )
            st.caption(
                "Your progress is tracked only for this session (nothing is saved after you close the tab)."
            )

        with st.container(border=True):
            st.subheader("What's next for you")
            st.info(st.session_state["recommended_next_action"], icon="🎯")

        # Live BKT readiness snapshot — same data as the dashboard panel,
        # surfaced on the home page so the BKT layer is visible without
        # needing to navigate to the dashboard.
        try:
            _tracker_home = get_bkt_tracker()
            _readiness_home = _tracker_home.eqao_readiness(threshold=0.75)
            _overall_home = _tracker_home.overall_readiness_score(threshold=0.75)
            with st.container(border=True):
                st.subheader("🧠 BKT mastery snapshot")
                st.caption(
                    "Bayesian Knowledge Tracing tracks a hidden P(mastered) per topic, "
                    "warm-started from the Random Forest and updated after every attempt."
                )
                _cols = st.columns(len(_readiness_home))
                for _i, (_topic_key, _ready) in enumerate(_readiness_home.items()):
                    _p = _tracker_home.p_mastered(_topic_key)
                    _title = TOPICS.get(_topic_key, {}).get("title", _topic_key)
                    _emoji = TOPICS.get(_topic_key, {}).get("emoji", "•")
                    _cols[_i].metric(
                        f"{_emoji} {_title}",
                        f"{_p:.0%}",
                        "mastered" if _ready else "keep practicing",
                        delta_color="normal" if _ready else "off",
                    )
                st.progress(
                    min(max(_overall_home, 0.0), 1.0),
                    text=f"Overall linear equations mastery: {_overall_home:.0%}",
                )
        except Exception:
            pass

    with right:
        with st.container(border=True):
            st.subheader("Pick a topic")
            for key, meta in TOPICS.items():
                if st.button(
                    f"{meta['emoji']}  {meta['title']}",
                    key=f"home_topic_{key}",
                    use_container_width=True,
                ):
                    st.session_state["current_topic"] = key
                    new_problem(key, st.session_state["current_difficulty"])
                    play_click()         # subtle click
                    st.toast(f"Topic set: {meta['title']}", icon="📘")

        with st.container(border=True):
            st.subheader("About mastery tracking")
            st.caption(
                "**Bayesian Knowledge Tracing (BKT)** is the primary mastery model. It maintains a "
                "hidden-state P(mastered) per topic and updates it after every attempt using the "
                "observed correctness, slip probability, and guess probability — the same family of "
                "models used by Carnegie Mellon's Cognitive Tutor. A topic counts as **mastered** "
                "once P(mastered) ≥ 0.75, matching the Ontario Level 3 (provincial standard) threshold."
            )
            with st.expander("Random Forest classifier (warm-start · advanced)"):
                st.caption(
                    "A scikit-learn **Random Forest** (supervised, not reinforcement learning) trained "
                    "on 13 behavioral features — recent accuracy, streaks, normalized response latency, "
                    "hint utilization, and temporal trajectory — classifies you into Ontario achievement "
                    "levels 1–4. Its probability output seeds the BKT prior so BKT doesn't have to "
                    "start cold. Adaptive next-step suggestions are separate rule-based logic on top."
                )


# ---------------------------------------------------------------------------
# Page: Learn & Practice
# ---------------------------------------------------------------------------

def page_learn() -> None:
    progress_strip()
    st.write("")

    # Topic + difficulty selectors
    top_left, top_right = st.columns([2, 1])
    with top_left:
        topic_keys = list(TOPICS.keys())
        topic_labels = [f"{TOPICS[k]['emoji']}  {TOPICS[k]['title']}" for k in topic_keys]
        current_idx = topic_keys.index(st.session_state["current_topic"]) \
            if st.session_state["current_topic"] in topic_keys else 0
        chosen_label = st.selectbox(
            "Topic",
            topic_labels,
            index=current_idx,
            key="topic_select",
        )
        chosen_topic = topic_keys[topic_labels.index(chosen_label)]

    with top_right:
        diff_options = ["easy", "medium", "hard"]
        diff_idx = diff_options.index(st.session_state["current_difficulty"])
        chosen_diff = st.selectbox("Difficulty", diff_options, index=diff_idx, key="diff_select")

    if (chosen_topic != st.session_state["current_topic"]
            or chosen_diff != st.session_state["current_difficulty"]
            or st.session_state["current_problem"] is None):
        new_problem(chosen_topic, chosen_diff)

    tab_lesson, tab_practice = st.tabs(["📖 Lesson", "✏️ Practice"])

    with tab_lesson:
        with st.container(border=True):
            st.markdown(get_explanation(chosen_topic))

        # Live graph preview for graphing / intercept / slope topics
        if chosen_topic in ("graphing", "intercept", "slope"):
            with st.container(border=True):
                st.subheader("Try the graph")
                cA, cB = st.columns(2)
                m_val = cA.slider("Slope (m)", -5.0, 5.0, 2.0, 0.5, key="lesson_m")
                b_val = cB.slider("y-intercept (b)", -8.0, 8.0, 1.0, 1.0, key="lesson_b")
                st.pyplot(plot_line(m_val, b_val), use_container_width=True)

        # Live two-line demo for the systems topic.
        if chosen_topic == "systems":
            with st.container(border=True):
                st.subheader("Try it: where do these lines meet?")
                st.caption(
                    "Change the slopes and intercepts. The green dot marks the intersection — "
                    "the (x, y) point that makes both equations true."
                )
                c1, c2 = st.columns(2)
                m1_val = c1.slider("Line 1 slope (m₁)", -5.0, 5.0, 2.0, 0.5, key="sys_m1")
                b1_val = c1.slider("Line 1 intercept (b₁)", -8.0, 8.0, 1.0, 1.0, key="sys_b1")
                m2_val = c2.slider("Line 2 slope (m₂)", -5.0, 5.0, -1.0, 0.5, key="sys_m2")
                b2_val = c2.slider("Line 2 intercept (b₂)", -8.0, 8.0, 4.0, 1.0, key="sys_b2")
                if abs(m1_val - m2_val) < 1e-9:
                    if abs(b1_val - b2_val) < 1e-9:
                        st.info(
                            "Both lines are the same — every point on the line is a solution."
                        )
                    else:
                        st.info(
                            "These lines are parallel (same slope, different intercepts) — they "
                            "never cross, so the system has no solution."
                        )
                st.pyplot(
                    plot_system(m1_val, b1_val, m2_val, b2_val), use_container_width=True
                )

    with tab_practice:
        problem = st.session_state["current_problem"]
        if problem is None:
            new_problem(chosen_topic, chosen_diff)
            problem = st.session_state["current_problem"]

        with st.container(border=True):
            st.markdown(
                f"<span class='topic-chip'>{TOPICS[problem.topic]['emoji']} "
                f"{TOPICS[problem.topic]['title']}</span>"
                f"<span class='topic-chip'>Difficulty: {problem.difficulty}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"### Problem\n{problem.question}")

            answer_key = f"answer_input_{st.session_state['answer_input_key']}"
            student_answer = st.text_input(
                "Your answer",
                key=answer_key,
                placeholder="Type your answer here…",
            )

            col1, col2, col3 = st.columns([1, 1, 1])
            submit_clicked = col1.button("✅ Submit", use_container_width=True)
            hint_clicked = col2.button("💡 Hint", use_container_width=True)
            new_clicked = col3.button("🔄 New problem", use_container_width=True)

            if submit_clicked:
                correct, msg = check_answer(student_answer, problem)
                st.session_state["feedback"] = msg
                st.session_state["feedback_correct"] = correct
                # Only count a real attempt when the input parses (msg isn't the "type a number" style issue)
                if student_answer.strip() != "":
                    record_attempt(correct)
                    if correct:
                        play_correct()   # chime + confetti
                    else:
                        play_wrong()     # thud + shake

            if hint_clicked:
                reveal_hint()
                play_hint()              # soft whoosh

            if new_clicked:
                new_problem(chosen_topic, chosen_diff)
                play_new_problem()       # rising cue
                st.rerun()

            # Feedback
            fb = st.session_state["feedback"]
            if fb is not None:
                if st.session_state["feedback_correct"]:
                    st.success(fb, icon="🎉")
                else:
                    st.warning(fb, icon="🧐")

            # Hints revealed so far
            revealed = st.session_state["hints_revealed"]
            if revealed > 0:
                with st.container(border=True):
                    st.markdown("#### Hints")
                    for i, hint in enumerate(problem.hints[:revealed], start=1):
                        st.markdown(f"**Hint {i}.** {hint}")
                    if revealed < len(problem.hints):
                        st.caption(f"{len(problem.hints) - revealed} more hint(s) available.")

            # Step-by-step solution
            with st.expander("📝 Show step-by-step solution"):
                for i, step in enumerate(problem.solution_steps, start=1):
                    st.markdown(f"**Step {i}.** {step}")
                # Visualize the solution depending on the problem kind.
                if problem.answer_kind == "point" and getattr(problem, "lines", None):
                    m1_t, b1_t, m2_t, b2_t = problem.lines
                    st.pyplot(
                        plot_system(m1_t, b1_t, m2_t, b2_t),
                        use_container_width=True,
                    )
                elif problem.answer_equation is not None and problem.answer_kind == "equation_mb":
                    m_t, b_t = problem.answer_equation
                    st.pyplot(plot_line(m_t, b_t), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Free-response Q&A
# ---------------------------------------------------------------------------

def page_ask() -> None:
    progress_strip()
    st.write("")

    with st.container(border=True):
        st.subheader("Ask a question")
        st.caption(
            "Ask anything about slope, y-intercept, graphing, systems of equations, or word problems "
            "and the tutor will walk you through it step by step."
        )

        question = st.text_area(
            "Your question",
            placeholder="e.g., How do I find the slope between (2, 3) and (5, 12)?",
            key="free_response_q",
            height=120,
        )
        col1, col2 = st.columns([1, 3])
        ask_clicked = col1.button("Ask tutor", use_container_width=True)
        if col2.button("Clear", use_container_width=True):
            st.session_state["free_response_answer"] = ""
            st.session_state["free_response_source"] = ""

        if ask_clicked and question.strip():
            api_key = get_api_key()
            api_reply = None
            if api_key:
                system = (
                    "You are a friendly Grade 9 math tutor. Focus on linear equations "
                    "in slope-intercept form (y = mx + b). Explain step by step, keep "
                    "the tone encouraging, and use short sentences. Use plain text only."
                )
                with st.spinner("Thinking…"):
                    api_reply = perplexity_chat(question.strip(), system=system)

            if api_reply:
                st.session_state["free_response_answer"] = api_reply
                st.session_state["free_response_source"] = "api"
            else:
                # Local fallback: show the explanation most relevant to the question
                q_low = question.lower()
                if "system" in q_low or "intersect" in q_low or "two lines" in q_low:
                    topic = "systems"
                elif "intercept" in q_low:
                    topic = "intercept"
                elif "graph" in q_low or "plot" in q_low or "draw" in q_low:
                    topic = "graphing"
                elif "word" in q_low or "cost" in q_low or "save" in q_low or "pool" in q_low:
                    topic = "word_problems"
                else:
                    topic = "slope"

                st.session_state["free_response_answer"] = (
                    f"Here's a local explanation that should help:\n\n{get_explanation(topic)}"
                )
                st.session_state["free_response_source"] = "local"

        reply = st.session_state["free_response_answer"]
        if reply:
            source = st.session_state["free_response_source"]
            if source == "api":
                st.success("Tutor reply", icon="🤖")
            else:
                st.info("Tutor reply", icon="📚")
            st.markdown(reply)
        else:
            st.caption("Type a question above and click **Ask tutor** to get started.")


# ---------------------------------------------------------------------------
# Page: Student dashboard
# ---------------------------------------------------------------------------

def page_dashboard() -> None:
    progress_strip()
    st.write("")

    total = st.session_state["total_attempts"]
    if total == 0:
        with st.container(border=True):
            st.subheader("No attempts yet")
            st.caption(
                "Head to **Learn & Practice** and solve a problem or two. "
                "Your mastery dashboard fills in automatically."
            )
        return

    left, right = st.columns([1.3, 1])

    with left:
        # ----- BKT (primary mastery model) ----------------------------------
        try:
            _tracker = get_bkt_tracker()
            _readiness = _tracker.eqao_readiness(threshold=0.75)
            _overall = _tracker.overall_readiness_score(threshold=0.75)
        except Exception:
            _tracker, _readiness, _overall = None, {}, 0.0

        with st.container(border=True):
            st.subheader("🧠 BKT mastery by topic")
            st.caption(
                "Bayesian Knowledge Tracing maintains a hidden-state P(mastered) per topic. "
                "It updates after every attempt using observed correctness plus slip/guess "
                "probabilities calibrated for Grade 9 linear equations content."
            )
            if _tracker is not None and _readiness:
                st.metric(
                    "Overall linear equations mastery",
                    f"{_overall:.0%}",
                    f"{sum(_readiness.values())}/{len(_readiness)} topics mastered",
                )
                st.progress(min(max(_overall, 0.0), 1.0))

                # Per-topic probability bars
                import matplotlib.pyplot as _plt
                _topic_keys = list(_readiness.keys())
                _topic_names = [TOPICS.get(k, {}).get("title", k) for k in _topic_keys]
                _topic_pmast = [_tracker.p_mastered(k) for k in _topic_keys]
                _bar_colors = ["#86A873" if p >= 0.75 else "#A7C098" if p >= 0.5 else "#C9D7C1"
                               for p in _topic_pmast]
                _fig, _ax = _plt.subplots(figsize=(5.6, 3.0), dpi=120)
                _bars = _ax.barh(_topic_names, _topic_pmast, color=_bar_colors, edgecolor="#4F6B44")
                _ax.axvline(0.75, color="#E76F51", linestyle="--", linewidth=1, label="Mastery threshold (0.75)")
                _ax.set_xlim(0, 1.0)
                _ax.set_xlabel("P(mastered)", color="#2F4858", fontsize=9)
                _ax.invert_yaxis()
                _ax.grid(axis="x", color="#E3E8EE", linewidth=0.6)
                for _s in _ax.spines.values():
                    _s.set_color("#CBD2D9")
                for _bar, _v in zip(_bars, _topic_pmast):
                    _ax.text(_v + 0.02, _bar.get_y() + _bar.get_height() / 2,
                             f"{_v:.0%}", va="center", fontsize=8, color="#2F4858")
                _ax.legend(fontsize=8, loc="lower right", frameon=False)
                _fig.tight_layout()
                st.pyplot(_fig, use_container_width=True)
            else:
                st.info("Solve a problem in Learn & Practice to start populating BKT.", icon="📝")

        with st.container(border=True):
            st.subheader("Next action")
            st.info(st.session_state["recommended_next_action"], icon="🧭")

        # ----- RF classifier (demoted to expander) --------------------------
        with st.expander("🌲 Random Forest classifier details (advanced)"):
            label = st.session_state["mastery_prediction"]
            probs = st.session_state["mastery_probs"]
            st.caption(
                "The Random Forest produces a one-shot classification from your 13 behavioral "
                "features. Its output also warm-starts the BKT prior above so BKT doesn't begin "
                "from a cold default."
            )
            st.metric("RF current level", label.title())
            import matplotlib.pyplot as _plt2
            _fig2, _ax2 = _plt2.subplots(figsize=(5.2, 2.4), dpi=120)
            _labels2 = [c.title() for c in FRIENDLY_BANDS]
            _values2 = [probs.get(c, 0.0) for c in FRIENDLY_BANDS]
            _colors2 = ["#C9D7C1", "#A7C098", "#86A873"]
            _bars2 = _ax2.bar(_labels2, _values2, color=_colors2, edgecolor="#4F6B44")
            _ax2.set_ylim(0, 1.05)
            _ax2.set_ylabel("Probability", color="#2F4858", fontsize=9)
            _ax2.grid(axis="y", color="#E3E8EE", linewidth=0.6)
            for _s in _ax2.spines.values():
                _s.set_color("#CBD2D9")
            for _bar, _v in zip(_bars2, _values2):
                _ax2.text(_bar.get_x() + _bar.get_width() / 2, _v + 0.03,
                          f"{_v:.0%}", ha="center", fontsize=9, color="#2F4858")
            _fig2.tight_layout()
            st.pyplot(_fig2, use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Session stats")
            correct = st.session_state["correct_attempts"]
            incorrect = st.session_state["incorrect_attempts"]
            hints = st.session_state["hint_count"]
            avg_time = (
                sum(st.session_state["response_times"]) / len(st.session_state["response_times"])
                if st.session_state["response_times"] else 0.0
            )
            st.metric("Correct", correct)
            st.metric("Incorrect", incorrect)
            st.metric("Hints used", hints)
            st.metric("Avg. time / problem", f"{avg_time:.1f}s")

    # Accuracy by topic
    hist = st.session_state["attempt_history"]
    if hist:
        df = pd.DataFrame(hist)
        with st.container(border=True):
            st.subheader("Accuracy by topic")
            acc = (
                df.groupby("topic")["correct"]
                .agg(["sum", "count"])
                .reset_index()
            )
            acc["accuracy"] = (acc["sum"] / acc["count"] * 100).round(1)
            acc["topic"] = acc["topic"].map(lambda k: TOPICS.get(k, {}).get("title", k))
            acc = acc.rename(columns={"sum": "correct", "count": "attempts"})
            st.dataframe(acc, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.subheader("Recent attempts")
            show = df.tail(8).iloc[::-1].copy()
            show["topic"] = show["topic"].map(lambda k: TOPICS.get(k, {}).get("title", k))
            show["correct"] = show["correct"].map(lambda b: "✅" if b else "❌")
            show = show.rename(columns={
                "topic": "Topic",
                "difficulty": "Difficulty",
                "correct": "Correct",
                "hints_used": "Hints",
                "seconds": "Time (s)",
                "timestamp": "Time",
            })
            st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Teacher view
# ---------------------------------------------------------------------------

def page_teacher() -> None:
    st.subheader("👩‍🏫 Teacher view · current session")
    st.caption(
        "Snapshot of this student's activity in the current browser session. "
        "Data is not persisted after the session ends."
    )

    total = st.session_state["total_attempts"]
    correct = st.session_state["correct_attempts"]
    acc = (correct / total * 100.0) if total else 0.0

    try:
        _tracker_t = get_bkt_tracker()
        _bkt_overall_t = _tracker_t.overall_readiness_score(threshold=0.75)
    except Exception:
        _bkt_overall_t = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attempts", total)
    c2.metric("Accuracy", f"{acc:.0f}%")
    c3.metric("Hints used", st.session_state["hint_count"])
    c4.metric("Mastery", f"{_bkt_overall_t:.0%}")

    st.write("")

    if total == 0:
        with st.container(border=True):
            st.info("No attempts yet this session.", icon="📭")
        return

    df = pd.DataFrame(st.session_state["attempt_history"])

    with st.container(border=True):
        st.markdown("#### Per-topic performance")
        by_topic = (
            df.groupby(["topic", "difficulty"])["correct"]
            .agg(["sum", "count"])
            .reset_index()
        )
        by_topic["accuracy %"] = (by_topic["sum"] / by_topic["count"] * 100).round(1)
        by_topic["topic"] = by_topic["topic"].map(lambda k: TOPICS.get(k, {}).get("title", k))
        by_topic = by_topic.rename(columns={
            "topic": "Topic",
            "difficulty": "Difficulty",
            "sum": "Correct",
            "count": "Attempts",
        })
        st.dataframe(by_topic, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("#### Attempt log")
        log = df.iloc[::-1].copy()
        log["topic"] = log["topic"].map(lambda k: TOPICS.get(k, {}).get("title", k))
        log["correct"] = log["correct"].map(lambda b: "✅" if b else "❌")
        log = log.rename(columns={
            "topic": "Topic",
            "difficulty": "Difficulty",
            "correct": "Correct",
            "hints_used": "Hints",
            "seconds": "Time (s)",
            "timestamp": "Time",
        })
        st.dataframe(log, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download session CSV",
            data=csv,
            file_name="session_attempts.csv",
            mime="text/csv",
            use_container_width=False,
        )

    with st.container(border=True):
        st.markdown("#### Coaching suggestion")
        st.info(st.session_state["recommended_next_action"], icon="🧭")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page.startswith("🏠"):
    page_home()
elif page.startswith("📚"):
    page_learn()
elif page.startswith("💬"):
    page_ask()
elif page.startswith("📊"):
    page_dashboard()
elif page.startswith("👩"):
    page_teacher()
else:
    page_home()
