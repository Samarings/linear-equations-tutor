"""
utils.py
--------
Helpers for answer checking, graphing, and the optional Perplexity API.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import streamlit as st  # for st.secrets
except ImportError:  # pragma: no cover
    st = None  # type: ignore

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# ---------------------------------------------------------------------------
# API key + Perplexity call (safe, optional)
# ---------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Read the Perplexity API key from st.secrets or environment. Never hardcode."""
    key: Optional[str] = None
    if st is not None:
        try:
            key = st.secrets.get("PERPLEXITY_API_KEY", None)  # type: ignore[attr-defined]
        except Exception:
            key = None
    if not key:
        key = os.environ.get("PERPLEXITY_API_KEY")
    if key and isinstance(key, str) and key.strip():
        return key.strip()
    return None


def perplexity_chat(prompt: str, system: str = "", timeout: int = 20) -> Optional[str]:
    """Call Perplexity's chat completions API. Returns None on any failure."""
    key = get_api_key()
    if not key or requests is None:
        return None
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": "sonar",
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = (choice.get("message") or {}).get("content")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _normalize_number(text: str) -> Optional[float]:
    text = text.strip().replace(" ", "")
    # Support simple fractions like 3/4
    if re.fullmatch(r"-?\d+/\d+", text):
        a, b = text.split("/")
        try:
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(text)
    except Exception:
        return None


_EQ_PATTERN = re.compile(
    r"^y\s*=\s*(?P<m>-?\d*\.?\d*)\s*x\s*(?P<sign>[+-])\s*(?P<b>\d+\.?\d*)$"
)
_EQ_PATTERN_NO_B = re.compile(r"^y\s*=\s*(?P<m>-?\d*\.?\d*)\s*x$")
_EQ_PATTERN_CONST = re.compile(r"^y\s*=\s*(?P<b>-?\d+\.?\d*)$")


def parse_line_equation(text: str) -> Optional[Tuple[float, float]]:
    """Parse a string like 'y=2x+3' or 'y = -x - 4' into (m, b)."""
    if not text:
        return None
    t = text.lower().replace(" ", "")
    m1 = _EQ_PATTERN.match(t)
    if m1:
        m_raw = m1.group("m")
        if m_raw in ("", "+"):
            m = 1.0
        elif m_raw == "-":
            m = -1.0
        else:
            try:
                m = float(m_raw)
            except ValueError:
                return None
        sign = m1.group("sign")
        try:
            b = float(m1.group("b"))
        except ValueError:
            return None
        if sign == "-":
            b = -b
        return m, b
    m2 = _EQ_PATTERN_NO_B.match(t)
    if m2:
        m_raw = m2.group("m")
        if m_raw in ("", "+"):
            m = 1.0
        elif m_raw == "-":
            m = -1.0
        else:
            try:
                m = float(m_raw)
            except ValueError:
                return None
        return m, 0.0
    m3 = _EQ_PATTERN_CONST.match(t)
    if m3:
        try:
            return 0.0, float(m3.group("b"))
        except ValueError:
            return None
    return None


def check_answer(student_input: str, problem) -> Tuple[bool, str]:
    """Check the student's answer against the problem.

    Returns (is_correct, feedback_message).
    """
    if student_input is None or str(student_input).strip() == "":
        return False, "Please type an answer before submitting."

    kind = getattr(problem, "answer_kind", "number")

    if kind in ("number", "slope", "intercept"):
        val = _normalize_number(student_input)
        if val is None:
            return False, "That doesn't look like a number. Try entering just a number like 2 or -0.5."
        target = problem.answer_value
        if target is None:
            return False, "Problem is missing a numeric answer. Please skip this one."
        if abs(val - target) < 1e-6:
            return True, "✅ Correct! Nice work."
        return False, f"Not quite. Your answer: {val}. Think about the steps again and peek at a hint."

    if kind == "equation_mb":
        parsed = parse_line_equation(student_input)
        if parsed is None:
            return False, "Please enter the equation in the form y=mx+b (for example: y=2x+3)."
        m_stu, b_stu = parsed
        m_t, b_t = problem.answer_equation or (0.0, 0.0)
        if abs(m_stu - m_t) < 1e-6 and abs(b_stu - b_t) < 1e-6:
            return True, "✅ Correct! Equation matches."
        return False, "Close — double-check the slope (m) and the y-intercept (b)."

    return False, "Unsupported answer type."


# ---------------------------------------------------------------------------
# Graphing
# ---------------------------------------------------------------------------

def plot_line(
    m: float,
    b: float,
    x_range: Tuple[int, int] = (-10, 10),
    y_range: Tuple[int, int] = (-10, 10),
    highlight_intercept: bool = True,
):
    """Return a matplotlib Figure showing y = mx + b on a tidy grid.

    The y-axis range is fixed (default ±10) and the axes are kept at equal
    scale so slope changes actually look steeper/shallower — otherwise
    matplotlib auto-scales the y-axis and a line with slope 5 looks the same
    as a line with slope 1.
    """
    # Square figure + equal aspect makes slope visually accurate.
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=120)
    xs = np.linspace(x_range[0], x_range[1], 200)
    ys = m * xs + b
    ax.plot(xs, ys, color="#2F4858", linewidth=2.4, label=f"y = {m:g}x + {b:g}")

    if highlight_intercept and y_range[0] <= b <= y_range[1]:
        ax.scatter([0], [b], color="#86A873", zorder=5, s=70, label=f"y-intercept (0, {b:g})")

    # Axes
    ax.axhline(0, color="#9AA5B1", linewidth=0.8)
    ax.axvline(0, color="#9AA5B1", linewidth=0.8)
    ax.grid(True, color="#E3E8EE", linewidth=0.6)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    # Equal aspect ratio — one unit of x equals one unit of y on screen.
    # This is what makes a slope of 2 look twice as steep as a slope of 1.
    ax.set_aspect("equal", adjustable="box")

    # Integer tick marks every 2 units — easier for students to read than 2.5s.
    ax.set_xticks(np.arange(x_range[0], x_range[1] + 1, 2))
    ax.set_yticks(np.arange(y_range[0], y_range[1] + 1, 2))

    ax.set_xlabel("x", fontsize=10, color="#2F4858")
    ax.set_ylabel("y", fontsize=10, color="#2F4858")
    ax.set_title("Line graph", fontsize=12, color="#2F4858", pad=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    for spine in ax.spines.values():
        spine.set_color("#CBD2D9")

    fig.tight_layout()
    return fig
