"""
bkt.py  —  Bayesian Knowledge Tracing (BKT) Bridge
====================================================
Senior ML Architect notes
--------------------------
BKT models mastery as a HIDDEN MARKOV PROCESS.
At each timestep the student is either in "mastered" or "not mastered" state.
Four parameters govern the model:

    p_init  : P(mastered at start)      — prior, seeded from RF output
    p_learn : P(learn | not mastered)   — transition probability
    p_forget: P(forget | mastered)      — usually near 0 for short sessions
    p_slip  : P(incorrect | mastered)   — random mistakes even when mastered
    p_guess : P(correct | not mastered) — lucky guesses

The RF pipeline feeds its mastery probability directly into p_init, creating a
warm-start BKT that combines the RF's rich behavioral features with BKT's
temporal update semantics.

Architecture diagram
--------------------

  Session attempt history
          │
          ▼
  ┌───────────────────┐
  │  compute_features │  (behavioral feature engineering)
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  MasteryPipeline  │  RandomForest → (label, prob_dict)
  │  .predict()       │
  └────────┬──────────┘
           │  prob_dict["level_3"] + prob_dict["level_4"]
           │  ──────────────────────────────────────────▶ p_init (warm prior)
           ▼
  ┌───────────────────┐
  │   BKTTracker      │  per-topic, per-student hidden Markov updates
  │   .update(correct)│
  └────────┬──────────┘
           │
           ▼
  p_mastered(t)  ─▶  UI mastery bar, adaptive routing, EQAO readiness flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


# ─────────────────────────────────────────────────────────────────────────────
# 1. BKT PARAMETERS
#    These are learned from data in production BKT systems (EM / gradient
#    descent). Here we use calibrated defaults suitable for a Grade 9 topic.
#    Each topic can have independent parameters — students learn systems of
#    equations differently from slope.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BKTParams:
    p_init:   float = 0.30   # prior probability of mastery at session start
    p_learn:  float = 0.18   # probability of transitioning to mastered per correct attempt
    p_forget: float = 0.02   # very low — Grade 9 concepts are relatively stable
    p_slip:   float = 0.10   # probability of getting it wrong even if mastered
    p_guess:  float = 0.22   # probability of getting it right even if not mastered


# Per-topic parameter sets — harder topics have lower p_learn and p_init
TOPIC_BKT_PARAMS: Dict[str, BKTParams] = {
    "slope":         BKTParams(p_init=0.35, p_learn=0.22, p_slip=0.08, p_guess=0.25),
    "intercept":     BKTParams(p_init=0.35, p_learn=0.22, p_slip=0.09, p_guess=0.24),
    "graphing":      BKTParams(p_init=0.28, p_learn=0.18, p_slip=0.10, p_guess=0.20),
    "systems":       BKTParams(p_init=0.20, p_learn=0.14, p_slip=0.12, p_guess=0.18),
    "word_problems": BKTParams(p_init=0.22, p_learn=0.15, p_slip=0.13, p_guess=0.19),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE BKT UPDATE EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def bkt_update(p_mastered: float,
               correct: bool,
               params: BKTParams) -> float:
    """
    One step of the BKT hidden Markov update.

    Step 1 — Observation update (Bayes theorem):
    ─────────────────────────────────────────────
    If the student answered correctly:

        P(obs=correct | mastered)    = 1 - p_slip
        P(obs=correct | not mastered) = p_guess

        P(mastered | obs=correct) =
            P(obs|mastered) * P(mastered)
            ─────────────────────────────────────────────────────
            P(obs|mastered)*P(mastered) + P(obs|¬mastered)*P(¬mastered)

    If the student answered incorrectly:

        P(obs=wrong | mastered)    = p_slip
        P(obs=wrong | not mastered) = 1 - p_guess

    Step 2 — Transition update:
    ────────────────────────────
        P(mastered at t+1) =
            P(mastered | obs) * (1 - p_forget)
            + P(¬mastered | obs) * p_learn

    Parameters
    ----------
    p_mastered : current P(mastered) before this observation
    correct    : whether the student answered correctly
    params     : BKTParams for this topic

    Returns
    -------
    Updated P(mastered) after this observation.
    """
    L = p_mastered
    s = params.p_slip
    g = params.p_guess

    # Step 1: Observation update
    if correct:
        p_obs_given_L  = 1.0 - s
        p_obs_given_nL = g
    else:
        p_obs_given_L  = s
        p_obs_given_nL = 1.0 - g

    numerator   = p_obs_given_L * L
    denominator = numerator + p_obs_given_nL * (1.0 - L)

    if denominator < 1e-10:
        p_L_given_obs = L
    else:
        p_L_given_obs = numerator / denominator

    # Step 2: Transition update
    p_L_next = (p_L_given_obs * (1.0 - params.p_forget)
                + (1.0 - p_L_given_obs) * params.p_learn)

    return float(min(max(p_L_next, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. BKT TRACKER  —  per-topic, per-student state machine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicState:
    """Tracks BKT state for a single topic."""
    p_mastered: float = 0.0
    history:    List[float] = field(default_factory=list)   # p_mastered over time
    n_attempts: int = 0
    n_correct:  int = 0
    rf_seeded:  bool = False   # whether RF has warmed this prior yet

    def snapshot(self) -> Dict:
        return {
            "p_mastered": round(self.p_mastered, 4),
            "n_attempts": self.n_attempts,
            "n_correct":  self.n_correct,
            "trajectory": self.history[-5:],
        }


class BKTTracker:
    """
    Multi-topic BKT tracker for a single student session.

    RF Bridge
    ---------
    Call warm_start_from_rf(topic, rf_prob_dict) once you have an RF
    prediction. This sets the BKT prior to the RF's combined probability
    of level_3 + level_4 (i.e. "high mastery"), giving BKT a behaviorally
    informed starting point instead of the cold default.

    Usage
    -----
        tracker = BKTTracker()
        tracker.warm_start_from_rf("slope", rf_probs)
        tracker.update("slope", correct=True)
        p = tracker.p_mastered("slope")
        print(tracker.eqao_readiness())
    """

    def __init__(self) -> None:
        self._states: Dict[str, TopicState] = {
            t: TopicState(p_mastered=TOPIC_BKT_PARAMS[t].p_init)
            for t in TOPIC_BKT_PARAMS
        }

    def warm_start_from_rf(self,
                            topic: str,
                            rf_prob_dict: Dict[str, float]) -> None:
        """
        Seed BKT prior from the RF probability output.

        Maps RF class probabilities to a single P(mastered) by treating
        level_3 and level_4 as the "mastered" states:

            p_init_bkt = P(level_3) * 0.6 + P(level_4) * 1.0

        The 0.6 coefficient acknowledges that level_3 is partial mastery —
        consistent with Ontario EQAO rubric semantics.
        """
        if topic not in self._states:
            return
        p3 = rf_prob_dict.get("level_3", 0.0)
        p4 = rf_prob_dict.get("level_4", 0.0)
        prior = p3 * 0.6 + p4 * 1.0
        prior = float(min(max(prior, 0.05), 0.95))   # avoid degenerate priors

        state = self._states[topic]
        if not state.rf_seeded:
            state.p_mastered = prior
            state.history.append(prior)
            state.rf_seeded = True

    def update(self, topic: str, correct: bool) -> float:
        """
        Process one attempt and return the updated P(mastered).

        This is called after every problem submission — O(1) time.
        """
        if topic not in self._states:
            topic = "slope"  # fallback
        params = TOPIC_BKT_PARAMS[topic]
        state  = self._states[topic]

        state.p_mastered = bkt_update(state.p_mastered, correct, params)
        state.history.append(state.p_mastered)
        state.n_attempts += 1
        if correct:
            state.n_correct += 1
        return state.p_mastered

    def p_mastered(self, topic: str) -> float:
        """Current P(mastered) for a topic."""
        return self._states.get(topic, TopicState()).p_mastered

    def all_states(self) -> Dict[str, Dict]:
        return {t: s.snapshot() for t, s in self._states.items()}

    def eqao_readiness(self, threshold: float = 0.75) -> Dict[str, bool]:
        """
        Returns a per-topic EQAO readiness flag.

        A student is considered 'ready' on a topic when BKT P(mastered) >= threshold.
        This threshold of 0.75 corresponds roughly to achieving Level 3 consistently
        on the EQAO rubric (the provincial standard).
        """
        return {
            topic: state.p_mastered >= threshold
            for topic, state in self._states.items()
        }

    def overall_readiness_score(self, threshold: float = 0.75) -> float:
        """Proportion of topics where the student is EQAO-ready."""
        flags = self.eqao_readiness(threshold)
        return sum(flags.values()) / len(flags)

    def to_json(self) -> str:
        """Serialise full tracker state — for Supabase persistence."""
        return json.dumps({
            topic: {
                "p_mastered": s.p_mastered,
                "history":    s.history,
                "n_attempts": s.n_attempts,
                "n_correct":  s.n_correct,
                "rf_seeded":  s.rf_seeded,
            }
            for topic, s in self._states.items()
        })

    @classmethod
    def from_json(cls, data: str) -> "BKTTracker":
        """Deserialise from Supabase — restores full temporal history."""
        tracker = cls()
        parsed = json.loads(data)
        for topic, vals in parsed.items():
            if topic in tracker._states:
                s = tracker._states[topic]
                s.p_mastered = vals["p_mastered"]
                s.history    = vals["history"]
                s.n_attempts = vals["n_attempts"]
                s.n_correct  = vals["n_correct"]
                s.rf_seeded  = vals["rf_seeded"]
        return tracker
