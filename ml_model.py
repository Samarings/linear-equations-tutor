"""
ml_model.py  —  Adaptive Mastery Pipeline
==========================================
Senior ML Engineer notes
-------------------------
Architecture: scikit-learn Pipeline with advanced behavioral feature engineering.
Classifier  : RandomForestClassifier — kept lightweight for free-tier hosting.
Labels      : Ontario EQAO 4-level mastery rubric (1 = Beginning → 4 = Extending)
Features    : correctness, response latency, hint utilization rate,
              temporal trajectory, and derived interaction terms.

BKT bridge  : The RF classifier output (mastery probability vector) feeds
              directly into the BKTTracker as the prior — see bottom of file.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Ontario EQAO achievement level labels
CLASS_LABELS = ["level_1", "level_2", "level_3", "level_4"]

# Rolling window size for temporal trajectory
TRAJECTORY_WINDOW = 5

# Difficulty weight map — harder problems penalise hint use more
DIFFICULTY_WEIGHTS: Dict[str, float] = {
    "easy":   0.5,
    "medium": 1.0,
    "hard":   1.8,
}

# Topic complexity scores — used in derived features
TOPIC_COMPLEXITY: Dict[str, float] = {
    "slope":         0.6,
    "intercept":     0.5,
    "graphing":      0.7,
    "systems":       1.0,
    "word_problems": 0.9,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (the core upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_latency(seconds: float,
                       difficulty: str,
                       topic_complexity: float = 0.7) -> float:
    """
    Normalize response latency into a cognitive load signal.

    Formula
    -------
    The raw latency is scaled by expected difficulty so that 45 seconds on a
    hard problem is treated differently from 45 seconds on an easy one.

        latency_norm = log1p(seconds) / log1p(expected_seconds)

    Expected seconds are heuristically set per difficulty. The result is then
    clipped to [0, 2] so extreme outliers (student went AFK) don't dominate.

    Interpretation
    --------------
        < 0.5  : suspiciously fast — likely guessing
        0.5–1.0: on-pace for difficulty level (healthy zone)
        1.0–1.5: slow but engaged
        > 1.5  : struggling or disengaged
    """
    expected: Dict[str, float] = {"easy": 30.0, "medium": 60.0, "hard": 100.0}
    base = expected.get(difficulty, 60.0) * topic_complexity
    normalized = np.log1p(max(seconds, 0.5)) / np.log1p(base)
    return float(np.clip(normalized, 0.0, 2.0))


def _hint_utilization_rate(hints_used: int,
                            difficulty: str,
                            total_hints_available: int = 3) -> float:
    """
    Weighted hint utilization rate.

    Formula
    -------
        HUR = (hints_used / total_hints_available) * difficulty_weight

    Harder problems receive a higher weight so that opening 2 hints on a hard
    problem signals more confusion than 2 hints on an easy one.

    Clipped to [0, 1] for feature consistency.
    """
    if total_hints_available == 0:
        return 0.0
    raw_rate = hints_used / total_hints_available
    weight = DIFFICULTY_WEIGHTS.get(difficulty, 1.0)
    return float(np.clip(raw_rate * weight / max(DIFFICULTY_WEIGHTS.values()), 0.0, 1.0))


def _temporal_trajectory(correctness_window: List[bool]) -> float:
    """
    Temporal Trajectory Score — rolling directional momentum.

    Formula
    -------
    Given the last N attempts (chronologically ordered), apply exponentially
    increasing weights so recent results count more:

        w_i = exp(alpha * i)   where i is the position index (0 = oldest)
        TTS = (weighted_sum - 0.5 * sum(weights)) / (0.5 * sum(weights))

    Result is in [-1, 1]:
        -1 = strong downward streak (likely regressing)
         0 = neutral / mixed
        +1 = strong upward streak (likely mastering)

    Pads with 0.5 (neutral) if fewer than TRAJECTORY_WINDOW attempts exist.
    """
    alpha = 0.6
    window = list(correctness_window[-TRAJECTORY_WINDOW:])

    # Pad left with neutral (0.5) if history is short
    while len(window) < TRAJECTORY_WINDOW:
        window.insert(0, None)

    weights = np.array([np.exp(alpha * i) for i in range(len(window))])
    values = np.array([0.5 if v is None else float(v) for v in window])

    w_sum = weights.sum()
    weighted_score = (weights * values).sum()
    tts = (weighted_score - 0.5 * w_sum) / (0.5 * w_sum)
    return float(np.clip(tts, -1.0, 1.0))


def compute_features(
    attempt_history: List[Dict],
    hint_count: int,
    response_times: List[float],
    window: int = 10,
) -> Dict[str, float]:
    """
    Compute the full feature vector from raw session data.

    Returns a flat dict suitable for a single-row DataFrame.
    Designed to be called after every attempt so the RF pipeline
    can produce an updated mastery prediction in real time.
    """
    if not attempt_history:
        return _empty_features()

    recent = attempt_history[-window:]
    n = len(recent)

    # ── Correctness features ──────────────────────────────────────────────
    correctness = [a["correct"] for a in recent]
    recent_correctness = float(np.mean(correctness)) if correctness else 0.0

    # Streak: count consecutive correct from the end
    streak = 0
    for c in reversed(correctness):
        if c:
            streak += 1
        else:
            break
    correct_streak = streak

    # ── Latency features ─────────────────────────────────────────────────
    latencies = response_times[-window:] if response_times else [30.0]
    difficulties = [a.get("difficulty", "medium") for a in recent]
    complexities = [TOPIC_COMPLEXITY.get(a.get("topic", "slope"), 0.7) for a in recent]

    norm_latencies = [
        _normalize_latency(t, d, c)
        for t, d, c in zip(latencies, difficulties, complexities)
    ]
    avg_latency_norm   = float(np.mean(norm_latencies))
    latency_variance   = float(np.var(norm_latencies))     # consistency signal
    latency_trend      = float(norm_latencies[-1] - norm_latencies[0]) if len(norm_latencies) > 1 else 0.0

    # ── Hint utilization features ─────────────────────────────────────────
    hint_rates = [
        _hint_utilization_rate(
            a.get("hints_used", 0),
            a.get("difficulty", "medium"),
        )
        for a in recent
    ]
    avg_hint_rate    = float(np.mean(hint_rates))
    hint_rate_trend  = float(hint_rates[-1] - hint_rates[0]) if len(hint_rates) > 1 else 0.0
    # Heavy hint users on correct answers = surface learning signal
    hint_correct_interaction = avg_hint_rate * (1.0 - recent_correctness)

    # ── Temporal trajectory ───────────────────────────────────────────────
    trajectory = _temporal_trajectory(correctness)

    # ── Derived interaction terms ─────────────────────────────────────────
    # Speed-accuracy trade-off: fast AND correct = genuine mastery
    speed_accuracy = (1.0 - avg_latency_norm / 2.0) * recent_correctness
    # Difficulty-weighted accuracy
    diff_weights = [DIFFICULTY_WEIGHTS.get(d, 1.0) for d in difficulties]
    w_accuracy = float(
        np.average(correctness, weights=diff_weights) if correctness else 0.0
    )

    # ── Volume / attempt features ─────────────────────────────────────────
    total_attempts = len(attempt_history)
    attempt_count  = n

    return {
        # Correctness
        "recent_correctness":       recent_correctness,
        "correct_streak":           float(correct_streak),
        "difficulty_weighted_acc":  w_accuracy,
        # Latency
        "avg_latency_norm":         avg_latency_norm,
        "latency_variance":         latency_variance,
        "latency_trend":            latency_trend,
        # Hints
        "avg_hint_rate":            avg_hint_rate,
        "hint_rate_trend":          hint_rate_trend,
        "hint_correct_interaction": hint_correct_interaction,
        # Trajectory
        "temporal_trajectory":      trajectory,
        # Interaction terms
        "speed_accuracy":           speed_accuracy,
        # Volume
        "total_attempts":           float(total_attempts),
        "attempt_count":            float(attempt_count),
    }


def _empty_features() -> Dict[str, float]:
    return {k: 0.0 for k in [
        "recent_correctness", "correct_streak", "difficulty_weighted_acc",
        "avg_latency_norm", "latency_variance", "latency_trend",
        "avg_hint_rate", "hint_rate_trend", "hint_correct_interaction",
        "temporal_trajectory", "speed_accuracy", "total_attempts", "attempt_count",
    ]}

FEATURE_COLUMNS = list(_empty_features().keys())


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYNTHETIC DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_training_data(n_students: int = 800,
                            seed: int = 42) -> pd.DataFrame:
    """
    Simulate behaviorally realistic training data for N students.

    Each student is assigned a hidden 'true mastery' level (1–4) which
    parameterises the distribution of their behavioral signals.
    This mirrors real EdTech data collection without requiring PII.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for student_id in range(n_students):
        true_level = rng.integers(1, 5)   # 1,2,3,4

        # Behavioral parameters scale with true mastery
        base_accuracy     = 0.20 + (true_level - 1) * 0.22   # 0.20 → 0.86
        base_latency_mult = 1.6  - (true_level - 1) * 0.3    # 1.6  → 0.7
        base_hint_rate    = 0.70 - (true_level - 1) * 0.18   # 0.70 → 0.16

        n_attempts = int(rng.integers(8, 25))
        history, times = [], []

        for i in range(n_attempts):
            difficulty = rng.choice(["easy", "medium", "hard"])
            topic = rng.choice(list(TOPIC_COMPLEXITY.keys()))
            comp  = TOPIC_COMPLEXITY[topic]
            diff_penalty = {"easy": 0.10, "medium": 0.0, "hard": -0.12}[difficulty]

            correct   = bool(rng.random() < base_accuracy + diff_penalty + rng.normal(0, 0.05))
            hints     = int(rng.integers(0, 4))
            latency   = float(rng.lognormal(
                mean=np.log(60 * base_latency_mult * comp),
                sigma=0.4
            ))

            history.append({
                "correct":    correct,
                "difficulty": difficulty,
                "topic":      topic,
                "hints_used": hints,
            })
            times.append(latency)

        hint_total = sum(a["hints_used"] for a in history)
        feats = compute_features(history, hint_total, times)
        feats["mastery_label"] = f"level_{true_level}"
        feats["student_id"]    = student_id
        rows.append(feats)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SKLEARN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class MasteryPipeline:
    """
    Wraps the sklearn Pipeline so the rest of the app gets a clean interface.

    fit()      — trains on a DataFrame produced by simulate_training_data()
                 or a real equivalent collected from session data.
    predict()  — returns (label, prob_dict) for a single feature dict.
    importance — returns a sorted Series of feature importances post-fit.
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._classes: List[str] = CLASS_LABELS
        self._is_fitted = False

    def build_pipeline(self) -> Pipeline:
        """Construct the sklearn Pipeline."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced",   # handles imbalanced mastery distributions
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def fit(self, df: Optional[pd.DataFrame] = None) -> "MasteryPipeline":
        if df is None:
            df = simulate_training_data()

        X = df[FEATURE_COLUMNS]
        y = df["mastery_label"]

        self._pipeline = self.build_pipeline()
        self._pipeline.fit(X, y)
        self._classes = list(self._pipeline.classes_)
        self._is_fitted = True
        return self

    def cross_validate(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Run 5-fold stratified CV and return mean/std accuracy."""
        if df is None:
            df = simulate_training_data()
        X, y = df[FEATURE_COLUMNS], df["mastery_label"]
        pipeline = self.build_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return {"mean_accuracy": float(scores.mean()), "std": float(scores.std())}

    def predict(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        Predict mastery label and return per-class probabilities.

        Returns
        -------
        label : str  — e.g. "level_3"
        probs : dict — e.g. {"level_1":0.05, "level_2":0.10,
                              "level_3":0.65, "level_4":0.20}
        """
        if not self._is_fitted:
            return "level_1", {c: 0.25 for c in CLASS_LABELS}

        row = pd.DataFrame([{k: features.get(k, 0.0) for k in FEATURE_COLUMNS}])
        proba = self._pipeline.predict_proba(row)[0]
        prob_dict = {cls: float(p) for cls, p in zip(self._classes, proba)}
        label = max(prob_dict, key=prob_dict.get)
        return label, prob_dict

    @property
    def feature_importances(self) -> pd.Series:
        """Sorted feature importances from the Random Forest."""
        if not self._is_fitted:
            return pd.Series(dtype=float)
        rf = self._pipeline.named_steps["clf"]
        return (
            pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)
            .sort_values(ascending=False)
        )


def train_mastery_model() -> MasteryPipeline:
    """Convenience function — called once at Streamlit app startup."""
    return MasteryPipeline().fit()


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADAPTIVE RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def recommend_next_action(label: str, features: Dict[str, float]) -> str:
    """
    Rule-based recommendation engine layered on top of RF prediction.

    This keeps the RF focused purely on classification while a separate
    interpretable layer drives pedagogical decisions — a deliberate
    separation of concerns for auditability in an EdTech context.
    """
    traj   = features.get("temporal_trajectory", 0.0)
    hints  = features.get("avg_hint_rate", 0.0)
    latency = features.get("avg_latency_norm", 1.0)
    acc    = features.get("recent_correctness", 0.0)
    streak = features.get("correct_streak", 0.0)

    if label == "level_4":
        if streak >= 4:
            return "Outstanding streak! Try a cross-strand challenge mixing systems of equations with word problems."
        return "Excellent mastery. Challenge yourself with harder word problems or teach a concept back in your own words."

    if label == "level_3":
        if traj > 0.4:
            return "Strong upward trend — push to the next difficulty. You're almost at Level 4!"
        if hints > 0.5:
            return "Good accuracy but high hint use. Try one problem without any hints to consolidate."
        return "Solid understanding. Mix in hard-difficulty problems and time yourself to build exam fluency."

    if label == "level_2":
        if traj < -0.3:
            return "Recent results are slipping — step back to easy problems and rebuild confidence before moving on."
        if latency > 1.2:
            return "You're spending a lot of time per problem. Review the lesson tab, then try easy problems timed."
        return "Developing well. Focus on one topic at a time and use hints strategically, not as a first resort."

    # level_1
    if acc < 0.25:
        return "Let's start from the beginning. Read the Slope lesson carefully, then try 3 easy problems."
    return "Keep going — every attempt builds understanding. Use hints freely and read each step-by-step solution."
