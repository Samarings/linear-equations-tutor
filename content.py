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
    "systems": {
        "title": "Systems of Equations",
        "emoji": "❌",
        "summary": "Finding the point where two lines meet — by graphing, substitution, or elimination.",
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
    "systems": """
### Systems of two linear equations

A **system of equations** is just **two lines at the same time**.
The **solution** is the single point **(x, y)** that makes **both** equations true —
on a graph, this is the point where the two lines **cross**.

**Three ways to solve:**

**1. Graphing** — draw both lines and read off where they intersect.
Good for checking, but only exact when the crossing lands on grid points.

**2. Substitution** — when one equation is already solved for *y* (like y = mx + b),
plug that expression into the other equation.
- Example: y = 2x + 1 and y = −x + 7
- Since both equal y, set them equal: 2x + 1 = −x + 7
- Solve: 3x = 6 → **x = 2**
- Plug back in: y = 2(2) + 1 = **5**. Intersection: **(2, 5)**.

**3. Elimination** — add or subtract the two equations to cancel a variable.
Best when both equations are in the form Ax + By = C.

**Three possible outcomes:**
- **Different slopes** → exactly one intersection point (most common).
- **Same slope, different intercepts** → parallel lines, **no solution**.
- **Same slope, same intercept** → same line, **infinite solutions**.

**Tip:** When both equations are in y = mx + b form, set the right sides equal
to find *x*, then plug *x* back into either equation to find *y*.
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
    # Parsing hint: "number" | "equation_mb" | "slope" | "intercept" | "point"
    answer_kind: str = "number"
    # Extra data for rendering solutions (e.g. the two lines in a system problem).
    # For systems problems: (m1, b1, m2, b2).
    lines: Optional[Tuple[float, float, float, float]] = None


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


# ---- Systems of equations -------------------------------------------------

def _gen_systems(difficulty: str, rng: random.Random) -> Problem:
    """Generate two lines that intersect at a nice integer point.

    Strategy: pick the intersection point (x, y) first, then build two lines
    that both pass through it. This guarantees integer answers every time.
    """
    if difficulty == "easy":
        x = rng.randint(-4, 4)
        y = rng.randint(-4, 4)
        m1 = rng.choice([1, 2, -1, -2])
        m2 = rng.choice([v for v in [1, 2, 3, -1, -2, -3] if v != m1])
    elif difficulty == "medium":
        x = rng.randint(-5, 5)
        y = rng.randint(-6, 6)
        m1 = rng.choice([1, 2, 3, -1, -2, -3])
        m2 = rng.choice([v for v in [1, 2, 3, 4, -1, -2, -3, -4] if v != m1])
    else:  # hard — include negative slopes, larger intercepts, and sometimes opposite-sign slopes
        x = rng.randint(-6, 6)
        y = rng.randint(-8, 8)
        m1 = rng.choice([2, 3, 4, -2, -3, -4])
        # Force the second slope to have opposite sign sometimes for a trickier crossing
        pool = [v for v in [2, 3, 4, 5, -2, -3, -4, -5] if v != m1]
        m2 = rng.choice(pool)

    # Build b so each line passes through (x, y): y = m*x + b  →  b = y - m*x
    b1 = y - m1 * x
    b2 = y - m2 * x

    eq1 = _fmt_line(m1, b1)
    eq2 = _fmt_line(m2, b2)

    # Pretty-print "m·x + b" so signs read naturally (e.g. "x − 10" instead of "1x + -10").
    def _rhs(m: int, b: int) -> str:
        mx = "x" if m == 1 else ("-x" if m == -1 else f"{m}x")
        if b == 0:
            return mx
        sign = "+" if b > 0 else "−"
        return f"{mx} {sign} {abs(b)}"

    rhs1 = _rhs(m1, b1)
    rhs2 = _rhs(m2, b2)
    dx = m1 - m2
    db = b2 - b1

    question = (
        f"Find the point where these two lines intersect:\n\n"
        f"**Line 1:** {eq1}\n\n"
        f"**Line 2:** {eq2}\n\n"
        f"Enter your answer as an ordered pair in the form (x, y) — for example: (2, 5) or (-1, 3)."
    )
    hints = [
        "At the intersection point, both equations give the same y for the same x.",
        f"Since both right-hand sides equal y, set them equal: {rhs1} = {rhs2}.",
        "Solve that equation for x, then plug x into either line to get y.",
    ]
    steps = [
        f"Both lines share a point (x, y), so set the right sides equal: {rhs1} = {rhs2}.",
        f"Collect x-terms on one side: {dx}x = {db}.",
        f"Divide both sides by {dx}: x = {db} ÷ {dx} = {x}.",
        f"Plug x = {x} into {eq1}: y = {rhs1.replace('x', f'({x})')} = {y}.",
        f"The two lines intersect at **({x}, {y})**.",
    ]
    return Problem(
        topic="systems",
        difficulty=difficulty,
        question=question,
        answer=f"({x}, {y})",
        answer_equation=(float(x), float(y)),  # (re)used to store the point for graphing
        hints=hints,
        solution_steps=steps,
        answer_kind="point",
        lines=(float(m1), float(b1), float(m2), float(b2)),
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


# Systems-of-equations word-problem templates.
# Each describes two linear scenarios that meet at a single (x, y) point. The
# generator below picks the crossing point first, then fills in the numbers.
_SYSTEM_WORD_TEMPLATES = [
    {
        "scenario": (
            "Gym A charges a ${b1} sign-up fee plus ${m1} per visit. "
            "Gym B charges a ${b2} sign-up fee plus ${m2} per visit."
        ),
        "question_tail": (
            "After how many visits do the two gyms cost the same, and what is that cost? "
            "Enter your answer as a point (visits, cost) — for example: (4, 60)."
        ),
        "x_label": "visits",
        "y_label": "total cost ($)",
        "require_positive_x": True,
    },
    {
        "scenario": (
            "Phone plan A costs ${b1} per month plus ${m1} per GB of data. "
            "Phone plan B costs ${b2} per month plus ${m2} per GB."
        ),
        "question_tail": (
            "At how many GB do both plans cost the same, and what is that monthly cost? "
            "Enter your answer as a point (GB, cost) — for example: (3, 42)."
        ),
        "x_label": "GB",
        "y_label": "monthly cost ($)",
        "require_positive_x": True,
    },
    {
        "scenario": (
            "Liam starts with ${b1} in savings and adds ${m1} each week. "
            "Noor starts with ${b2} and adds ${m2} each week."
        ),
        "question_tail": (
            "After how many weeks will they have the same amount, and how much will each have? "
            "Enter your answer as a point (weeks, dollars) — for example: (5, 120)."
        ),
        "x_label": "weeks",
        "y_label": "savings ($)",
        "require_positive_x": True,
    },
    {
        "scenario": (
            "Pool A starts with {b1} litres and fills at {m1} L per minute. "
            "Pool B starts with {b2} litres and fills at {m2} L per minute."
        ),
        "question_tail": (
            "After how many minutes will both pools contain the same amount of water, "
            "and how much water will that be? Enter your answer as a point (minutes, litres) — "
            "for example: (6, 90)."
        ),
        "x_label": "minutes",
        "y_label": "water (L)",
        "require_positive_x": True,
        "no_dollar": True,
    },
]


def _gen_word_system(difficulty: str, rng: random.Random) -> Problem:
    """Two-line word problem — the student finds the crossing point.

    Numbers are chosen so the answer is a clean integer (x, y) with positive x
    (you can't have -4 weeks).
    """
    if difficulty == "easy":
        x = rng.randint(2, 6)
        y_extra = rng.randint(10, 30)
        m1 = rng.randint(2, 5)
        m2 = rng.choice([v for v in range(1, 8) if v != m1])
    elif difficulty == "medium":
        x = rng.randint(3, 10)
        y_extra = rng.randint(20, 60)
        m1 = rng.randint(2, 8)
        m2 = rng.choice([v for v in range(1, 12) if v != m1])
    else:
        x = rng.randint(4, 14)
        y_extra = rng.randint(30, 90)
        m1 = rng.randint(3, 12)
        m2 = rng.choice([v for v in range(2, 16) if v != m1])

    y = m1 * x + y_extra - m1 * x  # fallback — will be overwritten below
    # Pick starting values so each line passes through (x, y) with b ≥ 0.
    # Bigger-slope plan should have a smaller starting value so they actually cross at x > 0.
    if m1 < m2:
        m_small, m_big = m1, m2
    else:
        m_small, m_big = m2, m1
    b_small = rng.randint(5, 30) + (m_big - m_small) * x  # larger starting value, smaller per-unit rate
    b_big = b_small - (m_big - m_small) * x               # smaller starting value, larger per-unit rate
    y = m_small * x + b_small  # same as m_big * x + b_big by construction

    # Assign back to m1/b1, m2/b2 so the template text reads naturally.
    # Keep the order the template was designed with (line 1 = smaller rate, line 2 = bigger rate).
    m1, b1 = m_small, b_small
    m2, b2 = m_big, b_big

    tpl = rng.choice(_SYSTEM_WORD_TEMPLATES)
    scenario = tpl["scenario"].format(b1=b1, m1=m1, b2=b2, m2=m2)
    question = f"{scenario} {tpl['question_tail']}"

    # Pretty-print right-hand sides for readable hints/steps.
    def _rhs(m: int, b: int) -> str:
        mx = "x" if m == 1 else ("-x" if m == -1 else f"{m}x")
        if b == 0:
            return mx
        sign = "+" if b > 0 else "−"
        return f"{mx} {sign} {abs(b)}"

    rhs1 = _rhs(m1, b1)
    rhs2 = _rhs(m2, b2)
    dx = m1 - m2
    db = b2 - b1

    hints = [
        "Write an equation for each scenario in the form y = (rate)·x + (starting value).",
        f"Line 1: y = {rhs1}. Line 2: y = {rhs2}. Set them equal to find x.",
        f"Solve {rhs1} = {rhs2} for x, then plug x back in to get y.",
    ]
    steps = [
        f"Line 1 equation: y = {rhs1}.",
        f"Line 2 equation: y = {rhs2}.",
        f"At the crossing point both equations share the same y, so: {rhs1} = {rhs2}.",
        f"Collect x-terms: {dx}x = {db}.",
        f"Divide: x = {db} ÷ {dx} = {x} {tpl['x_label']}.",
        f"Plug x = {x} into Line 1: y = {rhs1.replace('x', f'({x})')} = {y} {tpl['y_label']}.",
        f"Answer: ({x}, {y}).",
    ]
    return Problem(
        topic="word_problems",
        difficulty=difficulty,
        question=question,
        answer=f"({x}, {y})",
        answer_equation=(float(x), float(y)),
        hints=hints,
        solution_steps=steps,
        answer_kind="point",
        lines=(float(m1), float(b1), float(m2), float(b2)),
    )


def _gen_word(difficulty: str, rng: random.Random) -> Problem:
    # Mix in systems-of-equations word problems ~40% of the time so students
    # see real variety instead of the same y = mx + b pattern every time.
    if rng.random() < 0.4:
        return _gen_word_system(difficulty, rng)
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
    "systems": _gen_systems,
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
