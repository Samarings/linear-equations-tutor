"""
db.py
-----
Supabase authentication + per-user progress storage.

Reads credentials from Streamlit secrets or environment variables:
    SUPABASE_URL
    SUPABASE_ANON_KEY   (the publishable / anon key)

If either is missing, `is_enabled()` returns False and the app falls back
to session-only mode. All calls here are wrapped in try/except so a
network or auth failure never crashes the app.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

try:
    from supabase import Client, create_client  # type: ignore
except ImportError:  # pragma: no cover
    Client = None  # type: ignore
    create_client = None  # type: ignore


TABLE = "student_progress"


# ---------------------------------------------------------------------------
# Credentials + client
# ---------------------------------------------------------------------------

def _read_secret(name: str) -> Optional[str]:
    val: Optional[str] = None
    if st is not None:
        try:
            val = st.secrets.get(name, None)  # type: ignore[attr-defined]
        except Exception:
            val = None
    if not val:
        val = os.environ.get(name)
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return None


def is_enabled() -> bool:
    """True if Supabase credentials + library are available."""
    if create_client is None:
        return False
    return bool(_read_secret("SUPABASE_URL") and _read_secret("SUPABASE_ANON_KEY"))


def get_client() -> Optional["Client"]:
    """Return a cached Supabase client, or None if not configured."""
    if not is_enabled() or st is None:
        return None
    if "_supabase_client" in st.session_state:
        return st.session_state["_supabase_client"]
    url = _read_secret("SUPABASE_URL")
    key = _read_secret("SUPABASE_ANON_KEY")
    try:
        client = create_client(url, key)  # type: ignore[arg-type]
        st.session_state["_supabase_client"] = client
        return client
    except Exception:
        return None


def _apply_session_to_client(client) -> None:
    """Restore an auth session stored in st.session_state on the client."""
    if st is None:
        return
    sess = st.session_state.get("_supabase_session")
    if not sess:
        return
    try:
        client.auth.set_session(sess.get("access_token"), sess.get("refresh_token"))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def sign_up(email: str, password: str, display_name: str = "") -> Tuple[bool, str]:
    client = get_client()
    if client is None:
        return False, "Accounts are not configured for this app."
    try:
        resp = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"display_name": display_name}} if display_name else {},
        })
        if getattr(resp, "user", None) is None:
            return False, "Sign-up did not return a user. Please try again."
        # If email confirmations are on, session may be None until confirmed.
        if getattr(resp, "session", None):
            _store_session(resp.session, resp.user, display_name)
            return True, "Account created. You're signed in."
        return True, (
            "Account created. Check your inbox to confirm your email, then sign in."
        )
    except Exception as exc:  # pragma: no cover
        return False, _friendly_error(exc, "sign up")


def sign_in(email: str, password: str) -> Tuple[bool, str]:
    client = get_client()
    if client is None:
        return False, "Accounts are not configured for this app."
    try:
        resp = client.auth.sign_in_with_password({"email": email, "password": password})
        if getattr(resp, "session", None) is None or getattr(resp, "user", None) is None:
            return False, "Invalid email or password."
        _store_session(resp.session, resp.user)
        return True, "Welcome back."
    except Exception as exc:
        return False, _friendly_error(exc, "sign in")


def sign_out() -> None:
    client = get_client()
    if client is not None:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    if st is not None:
        for k in ("_supabase_session", "_supabase_user"):
            st.session_state.pop(k, None)


def current_user() -> Optional[Dict[str, Any]]:
    if st is None:
        return None
    return st.session_state.get("_supabase_user")


def _store_session(session, user, display_name: str = "") -> None:
    if st is None:
        return
    try:
        st.session_state["_supabase_session"] = {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        }
    except Exception:
        st.session_state["_supabase_session"] = None
    try:
        meta = getattr(user, "user_metadata", {}) or {}
        st.session_state["_supabase_user"] = {
            "id": user.id,
            "email": user.email,
            "display_name": display_name or meta.get("display_name") or user.email,
        }
    except Exception:
        st.session_state["_supabase_user"] = None


def _friendly_error(exc: Exception, action: str) -> str:
    msg = str(exc)
    low = msg.lower()
    if "invalid login" in low or "invalid credentials" in low:
        return "Invalid email or password."
    if "already registered" in low or "user already" in low:
        return "That email is already registered. Try signing in instead."
    if "password should" in low or "weak password" in low:
        return "Password is too weak. Use at least 6 characters."
    if "email" in low and "valid" in low:
        return "Please enter a valid email address."
    return f"Couldn't {action}: {msg[:160]}"


# ---------------------------------------------------------------------------
# Progress load / save
# ---------------------------------------------------------------------------

DEFAULT_ROW: Dict[str, Any] = {
    "attempt_history": [],
    "total_attempts": 0,
    "correct_attempts": 0,
    "incorrect_attempts": 0,
    "hint_count": 0,
    "response_times": [],
    "mastery_prediction": "low",
    "mastery_probs": {"low": 1.0, "medium": 0.0, "high": 0.0},
}


def load_progress(user_id: str) -> Dict[str, Any]:
    """Fetch a user's progress row. Returns DEFAULT_ROW if nothing saved yet."""
    client = get_client()
    if client is None:
        return dict(DEFAULT_ROW)
    _apply_session_to_client(client)
    try:
        resp = client.table(TABLE).select("*").eq("user_id", user_id).limit(1).execute()
        rows = getattr(resp, "data", None) or []
        if not rows:
            return dict(DEFAULT_ROW)
        row = rows[0]
        merged = dict(DEFAULT_ROW)
        for k in merged:
            if k in row and row[k] is not None:
                merged[k] = row[k]
        return merged
    except Exception:
        return dict(DEFAULT_ROW)


def save_progress(user_id: str, state: Dict[str, Any]) -> bool:
    """Upsert the user's progress row. Returns True on success."""
    client = get_client()
    if client is None:
        return False
    _apply_session_to_client(client)
    payload = {
        "user_id": user_id,
        "attempt_history": state.get("attempt_history", []),
        "total_attempts": int(state.get("total_attempts", 0)),
        "correct_attempts": int(state.get("correct_attempts", 0)),
        "incorrect_attempts": int(state.get("incorrect_attempts", 0)),
        "hint_count": int(state.get("hint_count", 0)),
        "response_times": [float(t) for t in state.get("response_times", [])],
        "mastery_prediction": str(state.get("mastery_prediction", "low")),
        "mastery_probs": state.get("mastery_probs", {"low": 1.0, "medium": 0.0, "high": 0.0}),
    }
    try:
        client.table(TABLE).upsert(payload, on_conflict="user_id").execute()
        return True
    except Exception:
        return False
