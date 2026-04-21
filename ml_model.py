"""
ml_model.py
-----------
Mastery classifier for the Grade 9 Linear Equations tutor.

How it works (short, plain-English summary):

    We use a scikit-learn RandomForestClassifier to predict a student's
    current mastery level as one of three classes: "low", "medium", "high".

    A Random Forest is an ensemble of many small decision trees. Each tree
    looks at a random subset of the data and features and votes on a class;
    the forest returns the majority vote. This makes it robust and gives
    us a quick probability estimate for each mastery level.

    Features (all computed live from the current session):
        - recent_correctness   : fraction correct over the last N attempts (0..1)
        - total_attempts       : how many problems the student has tried
        - hint_ratio           : hints used per attempt
        - avg_response_time    : average seconds per attempt
        - correct_streak       : longest recent run of correct answers

    Training data is synthetic and generated in code at import time, so the
    app works immediately on first run with no external files.

    NOTE: Random Forest here is **supervised classification**, not
    reinforcement learning. Any adaptive behavior (choosing the next
    action for the student) lives in `recommend_next_action` below as
    plain rule-based logic driven by the classifier's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURE_NAMES: List[str] = [
    "recent_correctness",
    "total_attempts",
    "hint_ratio",
    "avg_response_time",
    "correct_streak",
]

CLASS_LABELS: List[str] = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n_per_class: int = 400, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create a small, well-behaved synthetic dataset for the mastery classes.

    The distributions are designed so the classifier learns sensible decision
    boundaries:
        low    -> low correctness, lots of hints, slow
        medium -> mid correctness, some hints, average speed
        high   -> high correctness, few hints, fast
    """
    rng = np.random.default_rng(seed)
    rows: List[List[float]] = []
    labels: List[int] = []

    for cls_idx, label in enumerate(CLASS_LABELS):
        for _ in range(n_per_class):
            if label == "low":
                recent = float(np.clip(rng.normal(0.25, 0.12), 0.0, 1.0))
                attempts = int(np.clip(rng.normal(5, 3), 1, 40))
                hint_ratio = float(np.clip(rng.normal(1.3, 0.5), 0.0, 3.0))
                avg_time = float(np.clip(rng.normal(55, 15), 5, 180))
                streak = int(np.clip(rng.normal(0.5, 0.8), 0, 10))
            elif label == "medium":
                recent = float(np.clip(rng.normal(0.6, 0.1), 0.0, 1.0))
                attempts = int(np.clip(rng.normal(10, 4), 1, 50))
                hint_ratio = float(np.clip(rng.normal(0.6, 0.3), 0.0, 2.0))
                avg_time = float(np.clip(rng.normal(32, 10), 5, 120))
                streak = int(np.clip(rng.normal(2, 1.2), 0, 10))
            else:  # high
                recent = float(np.clip(rng.normal(0.9, 0.07), 0.0, 1.0))
                attempts = int(np.clip(rng.normal(14, 5), 1, 60))
                hint_ratio = float(np.clip(rng.normal(0.15, 0.15), 0.0, 1.0))
                avg_time = float(np.clip(rng.normal(18, 6), 3, 80))
                streak = int(np.clip(rng.normal(5, 1.8), 0, 15))
            rows.append([recent, attempts, hint_ratio, avg_time, streak])
            labels.append(cls_idx)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@dataclass
class MasteryModel:
    clf: RandomForestClassifier
    train_accuracy: float

    def predict(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Predict mastery label and return a dict of class probabilities."""
        x = np.array([[features.get(name, 0.0) for name in FEATURE_NAMES]], dtype=float)
        probs = self.clf.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        label = CLASS_LABELS[idx]
        prob_map = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
        return label, prob_map


def train_mastery_model(random_state: int = 42) -> MasteryModel:
    """Train the Random Forest on synthetic data and return a MasteryModel."""
    X, y = _make_synthetic_dataset()
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=1,
    )
    clf.fit(X, y)
    train_acc = float(clf.score(X, y))
    return MasteryModel(clf=clf, train_accuracy=train_acc)


# ---------------------------------------------------------------------------
# Feature extraction from session history
# ---------------------------------------------------------------------------

def compute_features(
    attempt_history: List[dict],
    hint_count: int,
    response_times: List[float],
    window: int = 5,
) -> Dict[str, float]:
    """Compute mastery features from the current session state.

    `attempt_history` is a list of dicts with at least:
        {"correct": bool, ...}
    """
    total = len(attempt_history)
    if total == 0:
        return {
            "recent_correctness": 0.0,
            "total_attempts": 0.0,
            "hint_ratio": 0.0,
            "avg_response_time": 0.0,
            "correct_streak": 0.0,
        }

    recent = attempt_history[-window:]
    recent_correct = sum(1 for a in recent if a.get("correct")) / max(len(recent), 1)

    # Longest trailing correct streak
    streak = 0
    for a in reversed(attempt_history):
        if a.get("correct"):
            streak += 1
        else:
            break

    hint_ratio = (hint_count / total) if total > 0 else 0.0
    avg_time = float(np.mean(response_times)) if response_times else 0.0

    return {
        "recent_correctness": float(recent_correct),
        "total_attempts": float(total),
        "hint_ratio": float(hint_ratio),
        "avg_response_time": float(avg_time),
        "correct_streak": float(streak),
    }


# ---------------------------------------------------------------------------
# Rule-based adaptive policy (NOT reinforcement learning)
# ---------------------------------------------------------------------------

def recommend_next_action(mastery: str, features: Dict[str, float]) -> str:
    """Choose the next tutoring action from mastery + features.

    Pure rule-based logic that reads the classifier's prediction.
    """
    recent = features.get("recent_correctness", 0.0)
    hint_ratio = features.get("hint_ratio", 0.0)
    attempts = features.get("total_attempts", 0.0)

    if attempts < 2:
        return "Try a couple of easy warm-up problems to get started."

    if mastery == "low":
        if hint_ratio > 0.8:
            return "Slow down and review the topic explanation, then try an easy problem with hints off."
        return "Work on more easy-level problems and read the step-by-step solution after each one."

    if mastery == "medium":
        if recent >= 0.75:
            return "You're ready to step up — try a medium-level problem next."
        return "Keep practicing easy/medium problems and use hints only when stuck."

    # high
    if recent >= 0.9:
        return "Great mastery — try a hard-level or word problem to stretch yourself."
    return "You're doing well. Try another medium or hard problem to lock in mastery."
