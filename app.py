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

from content import TOPICS, generate_problem, get_explanation
from ml_model import (
    CLASS_LABELS,
    compute_features,
    recommend_next_action,
    train_mastery_model,
)
from utils import check_answer, get_api_key, perplexity_chat, plot_line


# ---------------------------------------------------------------------------
# Page config + global styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Linear Equations Tutor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Calm academic palette:
#   Background: #F7F5F0 (warm off-white)
#   Primary:    #2F4858 (deep slate)
#   Accent:     #86A873 (sage)
#   Soft:       #E3E8EE (mist)
CUSTOM_CSS = """
<style>
:root {
    --lt-bg: #F7F5F0;
    --lt-surface: #FFFFFF;
    --lt-primary: #2F4858;
    --lt-accent: #86A873;
    --lt-muted: #6B7280;
    --lt-soft: #E3E8EE;
}

html, body, [class*="stApp"] {
    background-color: var(--lt-bg);
    color: var(--lt-primary);
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}

h1, h2, h3, h4 {
    color: var(--lt-primary) !important;
    letter-spacing: -0.01em;
}

/* Card-like containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--lt-surface);
    border-radius: 14px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--lt-surface);
    border: 1px solid var(--lt-soft);
    border-radius: 12px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] {
    color: var(--lt-muted) !important;
    font-size: 0.85rem;
}
div[data-testid="stMetricValue"] {
    color: var(--lt-primary) !important;
}

/* Buttons */
.stButton > button {
    background: var(--lt-primary);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: transform 0.05s ease-in-out, background 0.15s;
}
.stButton > button:hover {
    background: #1F313C;
    color: white;
}
.stButton > button:active {
    transform: translateY(1px);
}

/* Secondary / form buttons */
button[kind="secondary"] {
    background: var(--lt-surface) !important;
    color: var(--lt-primary) !important;
    border: 1px solid var(--lt-soft) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #EEF1EC;
    border-right: 1px solid var(--lt-soft);
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: var(--lt-primary) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {
    color: var(--lt-primary) !important;
    background: rgba(47, 72, 88, 0.08) !important;
}

/* Tabs */
.stTabs [role="tablist"] button {
    color: var(--lt-muted);
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    color: var(--lt-primary);
    border-bottom: 2px solid var(--lt-accent) !important;
}

/* Info/success/warn blocks use softer tones */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

.hero {
    background: linear-gradient(135deg, #2F4858 0%, #3E6374 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 18px;
}
.hero h1 { color: white !important; margin: 0 0 6px 0; font-size: 1.7rem; }
.hero p  { color: #D8E2E8; margin: 0; font-size: 0.98rem; }

.topic-chip {
    display: inline-block;
    background: var(--lt-soft);
    color: var(--lt-primary);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-right: 6px;
}

.small-muted { color: var(--lt-muted); font-size: 0.85rem; }
</style>
"""
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
    "mastery_prediction": "low",      # label
    "mastery_probs": {"low": 1.0, "medium": 0.0, "high": 0.0},
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
# Model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    return train_mastery_model()


MODEL = load_model()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def refresh_mastery() -> None:
    feats = compute_features(
        st.session_state["attempt_history"],
        st.session_state["hint_count"],
        st.session_state["response_times"],
    )
    label, probs = MODEL.predict(feats)
    st.session_state["recent_correctness"] = feats["recent_correctness"]
    st.session_state["mastery_prediction"] = label
    st.session_state["mastery_probs"] = probs
    st.session_state["recommended_next_action"] = recommend_next_action(label, feats)


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


def reveal_hint() -> None:
    prob = st.session_state["current_problem"]
    if prob is None:
        return
    if st.session_state["hints_revealed"] < len(prob.hints):
        st.session_state["hints_revealed"] += 1
        st.session_state["hint_count"] += 1
        refresh_mastery()


def reset_session() -> None:
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.session_state["answer_input_key"] += 1


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📐 Linear Tutor")
    st.caption("Grade 9 · y = mx + b")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📚 Learn & Practice", "💬 Ask a Question", "📊 Dashboard", "👩‍🏫 Teacher View"],
        label_visibility="collapsed",
    )

    st.divider()

    api_ok = get_api_key() is not None
    if api_ok:
        st.success("Perplexity API: connected", icon="✅")
    else:
        st.info("Running in offline mode (local content only).", icon="🧭")

    st.caption("Add `PERPLEXITY_API_KEY` in Streamlit secrets to unlock AI tutor replies.")

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
      <h1>Linear Equations Tutor</h1>
      <p>Master y = mx + b — slope, intercepts, graphing, and word problems — with practice, hints, and mastery tracking.</p>
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
    c1.metric("Attempts", total)
    c2.metric("Correct", correct)
    c3.metric("Accuracy", f"{acc:.0f}%")
    c4.metric("Mastery", st.session_state["mastery_prediction"].title())


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
                "- 📝 **Word problems** — turning real situations into equations"
            )
            st.caption(
                "Your progress is tracked only for this session (nothing is saved after you close the tab)."
            )

        with st.container(border=True):
            st.subheader("What's next for you")
            st.info(st.session_state["recommended_next_action"], icon="🎯")

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
                    st.toast(f"Topic set: {meta['title']}", icon="📘")

        with st.container(border=True):
            st.subheader("About mastery tracking")
            st.caption(
                "A scikit-learn **Random Forest classifier** (supervised, not reinforcement learning) "
                "predicts your mastery as **low / medium / high** from your recent accuracy, "
                "number of attempts, hint usage, and response time. "
                "Adaptive next-step suggestions are separate rule-based logic on top of that prediction."
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

            if hint_clicked:
                reveal_hint()

            if new_clicked:
                new_problem(chosen_topic, chosen_diff)
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
                if problem.answer_equation is not None:
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
            "Ask anything about slope, y-intercept, graphing, or word problems. "
            "When an API key is set, the tutor uses Perplexity. Otherwise, you'll get a "
            "local summary from the built-in lessons."
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
                if "intercept" in q_low:
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
                st.success("Tutor reply (Perplexity)", icon="🤖")
            else:
                st.info("Tutor reply (local fallback)", icon="📚")
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
        with st.container(border=True):
            st.subheader("Mastery prediction")
            label = st.session_state["mastery_prediction"]
            probs = st.session_state["mastery_probs"]
            st.metric("Current level", label.title())
            # Render the probability bars in pedagogical order (Low → Medium → High).
            import matplotlib.pyplot as _plt  # local import keeps top clean
            _fig, _ax = _plt.subplots(figsize=(5.2, 2.6), dpi=120)
            _labels = [c.title() for c in CLASS_LABELS]
            _values = [probs.get(c, 0.0) for c in CLASS_LABELS]
            _colors = ["#C9D7C1", "#A7C098", "#86A873"]
            _bars = _ax.bar(_labels, _values, color=_colors, edgecolor="#4F6B44")
            _ax.set_ylim(0, 1.05)
            _ax.set_ylabel("Probability", color="#2F4858", fontsize=9)
            _ax.grid(axis="y", color="#E3E8EE", linewidth=0.6)
            for _s in _ax.spines.values():
                _s.set_color("#CBD2D9")
            for _bar, _v in zip(_bars, _values):
                _ax.text(_bar.get_x() + _bar.get_width() / 2, _v + 0.03,
                         f"{_v:.0%}", ha="center", fontsize=9, color="#2F4858")
            _fig.tight_layout()
            st.pyplot(_fig, use_container_width=True)
            st.caption(
                "Predicted by a scikit-learn Random Forest using your recent correctness, "
                "attempt count, hint usage, and average response time."
            )

        with st.container(border=True):
            st.subheader("Next action")
            st.info(st.session_state["recommended_next_action"], icon="🧭")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attempts", total)
    c2.metric("Accuracy", f"{acc:.0f}%")
    c3.metric("Hints used", st.session_state["hint_count"])
    c4.metric("Mastery", st.session_state["mastery_prediction"].title())

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
