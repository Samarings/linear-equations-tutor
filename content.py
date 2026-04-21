"""
content.py
----------
Built-in local content for the Grade 9 Linear Equations tutor.

This module contains:
- Topic definitions and subtopics
- Local, teacher-written explanations for each subtopic
- A practice-problem generator for each subtopic and difficulty
- Scaffolded hints and step-by-step solutions

Everything here is pure Python + stdlib so the app runs fully offline
when no API key is available.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Topic catalog
# ---------------------------------------------------------------------------

TOPICS: Dict[str, Dict[str, str]] = {
    "slope": {
        "title": "Understanding Slope (m)",
        "emoji": "📈",
        "summary": "What slope means, how to read it, and how to calculate it from two points.",
    },
    "intercept": {
        "title": "Understanding the y-Intercept (b)",
        "emoji": "🎯",
        "summary": "What b means in y = mx + b and how to spot it on a graph or in an equation.",
    },
    "graphing": {
        "title": "Graphing y = mx + b",
        "emoji": "📐",
        "summary": "How to graph a line starting from the y-intercept and using the slope as rise/run.",
    },
    "word_problems": {
        "title": "Linear Word Problems",
        "emoji": "📝",
        "summary": "Turning real-life situations into y = mx + b and answering questions about them.",
    },
}


# ---------------------------------------------------------------------------
# Local explanations (used when no API key is available)
# ---------------------------------------------------------------------------

EXPLANATIONS: Dict[str, str] = {
    "slope": """
### What is slope?

In the equation **y = mx + b**, the letter **m** is the **slope**.
Slope tells you **how steep** the line is and **which direction** it goes.

**Rise over run:**

$$ m = \\frac{\\text{rise}}{\\text{run}} = \\frac{y_2 - y_1}{x_2 - x_1} $$

- If **m > 0** → the line goes **up** from left to right.
- If **m < 0** → the line goes **down** from left to right.
- If **m = 0** → the line is **flat** (horizontal).
- A bigger |m| means a **steeper** line.

**Quick example:** Through (1, 2) and (4, 8):
- rise = 8 − 2 = 6
- run = 4 − 1 = 3
- m = 6 ÷ 3 = **2**

So the line goes up 2 for every 1 step to the right.
""",
    "intercept": """
### What is the y-intercept?

In **y = mx + b**, the letter **b** is the **y-intercept**.
It is the **y-value where the line crosses the y-axis** — the point (0, b).

**How to find b:**
- From an equation like y = 3x + 5 → **b = 5**.
- From a graph → look at where the line hits the vertical y-axis.
- From a table → find the row where x = 0; the y-value is b.

**Real-life meaning:** b is often the **starting amount** before anything happens.
For example, if a taxi charges $4 plus $2 per km, then b = 4 (the cost before you move).
""",
    "graphing": """
### Graphing y = mx + b

Follow these three simple steps:

1. **Plot the y-intercept** (0, b). This is your starting point on the y-axis.
2. **Use the slope** m = rise/run to step to the next point:
   - Go **up** (if rise is positive) or **down** (if negative)
   - Then go **right** by the run
3. **Draw a straight line** through your points. Extend it in both directions.

**Example:** y = 2x + 1
- Start at (0, 1).
- Slope = 2 = 2/1 → up 2, right 1 → next point (1, 3).
- Another step → (2, 5). Draw the line.

✅ Tip: If m is a fraction like 3/4, rise = 3, run = 4.
""",
    "word_problems": """
### Linear word problems

Many real-life problems follow **y = mx + b**:

- **b** = the starting value (what you have before anything changes)
- **m** = the rate of change (how much it grows or shrinks per unit)
- **x** = the input (time, distance, number of items…)
- **y** = the output (total cost, total distance, money saved…)

**Strategy:**
1. Find the starting amount → that's **b**.
2. Find the rate → that's **m**.
3. Write **y = mx + b**.
4. Plug in the x they ask about to get y (or set y and solve for x).

**Example:** A pool has 20 L of water and fills at 5 L per minute.
- b = 20, m = 5 → y = 5x + 20
- After 7 minutes: y = 5(7) + 20 = **55 L**
""",
}


# ---------------------------------------------------------------------------
# Problem generator
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    topic: str
    difficulty: str
    question: str
    answer: str                 # canonical string form, e.g. "2" or "y=2x+3"
    answer_value: Optional[float] = None
    answer_equation: Optional[Tuple[float, float]] = None  # (m, b) if applicable
    hints: List[str] = field(default_factory=list)
    solution_steps: List[str] = field(default_factory=list)
    # Parsing hint: "number" | "equation_mb" | "slope" | "intercept"
    answer_kind: str = "number"


def _rand_nonzero(lo: int, hi: int, rng: random.Random) -> int:
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def _fmt_line(m: float, b: float) -> str:
    """Pretty-format y = mx + b as a canonical string."""
    m_i = int(m) if float(m).is_integer() else m
    b_i = int(b) if float(b).is_integer() else b
    if m_i == 0:
        return f"y={b_i}"
    if m_i == 1:
        m_str = ""
    elif m_i == -1:
        m_str = "-"
    else:
        m_str = f"{m_i}"
    if b_i == 0:
        return f"y={m_str}x"
    sign = "+" if b_i > 0 else "-"
    return f"y={m_str}x{sign}{abs(b_i)}"


# ---- Slope problems -------------------------------------------------------

def _gen_slope(difficulty: str, rng: random.Random) -> Problem:
    if difficulty == "easy":
        m = rng.choice([1, 2, 3, -1, -2])
        x1 = rng.randint(-3, 3)
        y1 = rng.randint(-3, 3)
    elif difficulty == "medium":
        m = rng.choice([2, 3, 4, -2, -3, -4])
        x1 = rng.randint(-5, 5)
        y1 = rng.randint(-5, 5)
    else:  # hard
        m = rng.choice([-5, -4, -3, 3, 4, 5])
        x1 = rng.randint(-8, 8)
        y1 = rng.randint(-8, 8)

    run = _rand_nonzero(1, 4, rng)
    x2 = x1 + run
    y2 = y1 + m * run
    question = (
        f"Find the slope of the line that passes through "
        f"the points ({x1}, {y1}) and ({x2}, {y2}). "
        f"Enter your answer as a number (for example: 2 or -0.5)."
    )
    hints = [
        "Slope is rise over run: (y₂ − y₁) / (x₂ − x₁).",
        f"Compute the rise: {y2} − {y1} = {y2 - y1}.",
        f"Compute the run: {x2} − {x1} = {x2 - x1}. Now divide rise by run.",
    ]
    steps = [
        "Recall the slope formula: m = (y₂ − y₁) / (x₂ − x₁).",
        f"Substitute the points: m = ({y2} − {y1}) / ({x2} − {x1}).",
        f"Simplify the top and bottom: m = {y2 - y1} / {x2 - x1}.",
        f"Divide to finish: m = {m}.",
    ]
    return Problem(
        topic="slope",
        difficulty=difficulty,
        question=question,
        answer=str(m),
        answer_value=float(m),
        hints=hints,
        solution_steps=steps,
        answer_kind="slope",
    )


# ---- Intercept problems ---------------------------------------------------

def _gen_intercept(difficulty: str, rng: random.Random) -> Problem:
    m = _rand_nonzero(-5, 5, rng)
    if difficulty == "easy":
        b = rng.randint(-5, 5)
        pretty = _fmt_line(m, b)
        question = f"What is the y-intercept of the line {pretty}? Enter just the number."
        hints = [
            "The y-intercept is the constant term b in y = mx + b.",
            "It is the y-value when x = 0.",
        ]
        steps = [
            "Compare y = mx + b with the given equation.",
            f"The constant term is {b}, so b = {b}.",
            f"The y-intercept is {b}. The line crosses the y-axis at (0, {b}).",
        ]
    elif difficulty == "medium":
        b = rng.randint(-8, 8)
        x0 = _rand_nonzero(-4, 4, rng)
        y0 = m * x0 + b
        question = (
            f"A line has slope {m} and passes through the point ({x0}, {y0}). "
            f"What is its y-intercept? Enter just the number."
        )
        hints = [
            "Use y = mx + b and plug in the point to solve for b.",
            f"Substitute: {y0} = ({m})({x0}) + b.",
            "Isolate b by subtracting m·x from both sides.",
        ]
        steps = [
            "Start with y = mx + b.",
            f"Plug in the point ({x0}, {y0}) and m = {m}: {y0} = {m}·{x0} + b.",
            f"Compute {m}·{x0} = {m*x0}, so {y0} = {m*x0} + b.",
            f"Subtract {m*x0} from both sides: b = {y0 - m*x0}.",
        ]
        b = y0 - m * x0  # ensures int
    else:  # hard
        b = rng.randint(-9, 9)
        x1 = _rand_nonzero(-4, 4, rng)
        x2 = x1 + _rand_nonzero(1, 4, rng)
        y1 = m * x1 + b
        y2 = m * x2 + b
        question = (
            f"A line passes through ({x1}, {y1}) and ({x2}, {y2}). "
            f"What is its y-intercept? Enter just the number."
        )
        hints = [
            "First find the slope using rise/run.",
            f"Then substitute one point into y = mx + b to solve for b.",
            "Double-check by plugging the other point in.",
        ]
        steps = [
            f"Slope m = ({y2} − {y1}) / ({x2} − {x1}) = {m}.",
            f"Use y = mx + b with point ({x1}, {y1}): {y1} = {m}·{x1} + b.",
            f"So b = {y1} − {m*x1} = {b}.",
        ]
    return Problem(
        topic="intercept",
        difficulty=difficulty,
        question=question,
        answer=str(b),
        answer_value=float(b),
        hints=hints,
        solution_steps=steps,
        answer_kind="intercept",
    )


# ---- Graphing problems ----------------------------------------------------

def _gen_graphing(difficulty: str, rng: random.Random) -> Problem:
    if difficulty == "easy":
        m = rng.choice([1, 2, -1, -2])
        b = rng.randint(-3, 3)
    elif difficulty == "medium":
        m = rng.choice([2, 3, -2, -3])
        b = rng.randint(-5, 5)
    else:
        m = rng.choice([-4, -3, 3, 4])
        b = rng.randint(-7, 7)

    # Ask student for the equation given slope & intercept — graphing companion exercise.
    question = (
        f"Write the equation of a line with slope **{m}** and y-intercept **{b}**. "
        f"Enter it in the form y=mx+b (for example: y=2x+3 or y=-x-4)."
    )
    hints = [
        "The slope-intercept form is y = mx + b.",
        f"Plug in m = {m} and b = {b}.",
        "If b is negative, write it with a minus sign (e.g. y=2x-3).",
    ]
    steps = [
        "Start from the slope-intercept form y = mx + b.",
        f"Substitute m = {m} and b = {b}.",
        f"The equation is {_fmt_line(m, b)}.",
        f"Graphing tip: start at (0, {b}), then use slope {m} as rise/run to plot the next point.",
    ]
    return Problem(
        topic="graphing",
        difficulty=difficulty,
        question=question,
        answer=_fmt_line(m, b),
        answer_equation=(float(m), float(b)),
        hints=hints,
        solution_steps=steps,
        answer_kind="equation_mb",
    )


# ---- Word problems --------------------------------------------------------

_WORD_TEMPLATES = [
    {
        "scenario": "A swimming pool already contains {b} litres of water. A hose fills it at {m} litres per minute.",
        "unit_x": "minutes",
        "unit_y": "litres",
        "subject": "water in the pool",
    },
    {
        "scenario": "Maya has ${b} saved. She saves another ${m} each week.",
        "unit_x": "weeks",
        "unit_y": "dollars",
        "subject": "total savings",
    },
    {
        "scenario": "A taxi costs ${b} as a base fare plus ${m} per kilometre.",
        "unit_x": "kilometres",
        "unit_y": "dollars",
        "subject": "total taxi cost",
    },
    {
        "scenario": "A candle starts at {b} cm tall and burns down {m_abs} cm per hour.",
        "unit_x": "hours",
        "unit_y": "cm",
        "subject": "candle height",
        "force_negative_m": True,
    },
]


def _gen_word(difficulty: str, rng: random.Random) -> Problem:
    tpl = rng.choice(_WORD_TEMPLATES)
    if difficulty == "easy":
        b = rng.randint(2, 10)
        m_mag = rng.randint(1, 5)
        x_query = rng.randint(2, 6)
    elif difficulty == "medium":
        b = rng.randint(5, 25)
        m_mag = rng.randint(2, 8)
        x_query = rng.randint(3, 10)
    else:
        b = rng.randint(10, 50)
        m_mag = rng.randint(3, 12)
        x_query = rng.randint(4, 15)

    m = -m_mag if tpl.get("force_negative_m") else m_mag
    scenario = tpl["scenario"].format(b=b, m=m, m_abs=abs(m))
    y_answer = m * x_query + b

    question = (
        f"{scenario} "
        f"After {x_query} {tpl['unit_x']}, what is the {tpl['subject']} "
        f"in {tpl['unit_y']}? Enter just the number."
    )
    hints = [
        f"Find the starting value (b) and the rate of change (m).",
        f"Write the equation y = mx + b with the numbers from the problem.",
        f"Plug in x = {x_query} and compute y.",
    ]
    steps = [
        f"Starting value b = {b}, rate m = {m}.",
        f"Equation: y = {m}x + {b}.",
        f"Substitute x = {x_query}: y = {m}·{x_query} + {b}.",
        f"Compute: y = {m*x_query} + {b} = {y_answer}.",
        f"Answer: {y_answer} {tpl['unit_y']}.",
    ]
    return Problem(
        topic="word_problems",
        difficulty=difficulty,
        question=question,
        answer=str(y_answer),
        answer_value=float(y_answer),
        hints=hints,
        solution_steps=steps,
        answer_kind="number",
    )


_GENERATORS: Dict[str, Callable[[str, random.Random], Problem]] = {
    "slope": _gen_slope,
    "intercept": _gen_intercept,
    "graphing": _gen_graphing,
    "word_problems": _gen_word,
}


def generate_problem(topic: str, difficulty: str, seed: Optional[int] = None) -> Problem:
    """Generate a fresh problem for the given topic and difficulty."""
    rng = random.Random(seed)
    if topic not in _GENERATORS:
        topic = "slope"
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "easy"
    return _GENERATORS[topic](difficulty, rng)


def get_explanation(topic: str) -> str:
    """Return the local markdown explanation for a topic."""
    return EXPLANATIONS.get(topic, "No explanation available for this topic yet.")
