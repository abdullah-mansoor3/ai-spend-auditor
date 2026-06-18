"""
rate_limiter.py — Session-based rate limiting for AI Spend Auditor.

Uses Streamlit session_state exclusively — no Redis, no persistent storage,
no database.  Each browser session gets its own independent counter.

Configuration constants at the top of this file can be adjusted to tune
behaviour without touching the logic.
"""

import time
import logging

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — adjust these to tune rate-limiting behaviour
# ---------------------------------------------------------------------------
MAX_ANALYSES_PER_SESSION: int = 5     # analyses allowed per browser session
ANALYSIS_COOLDOWN_SECONDS: int = 30   # minimum gap between consecutive analyses


class SessionRateLimiter:
    """
    Track per-session analysis usage via st.session_state.

    Initialises state keys on construction (idempotent — safe to call every
    Streamlit re-run).

    Usage::

        limiter = SessionRateLimiter()
        allowed, reason = limiter.check()
        if not allowed:
            st.warning(reason)
        else:
            # run analysis …
            limiter.record()
    """

    def __init__(self) -> None:
        """
        Initialise rate-limit keys in session_state if not already present.

        Returns:
            None.

        Raises:
            Nothing — fails silently to avoid crashing the app.
        """
        try:
            if "rate_limit_count" not in st.session_state:
                st.session_state.rate_limit_count = 0
            if "rate_limit_last_ts" not in st.session_state:
                st.session_state.rate_limit_last_ts = 0.0
        except Exception as exc:  # noqa: BLE001
            logger.error("SessionRateLimiter init error: %s", exc)

    def check(self) -> tuple[bool, str]:
        """
        Check whether the current session is allowed to run another analysis.

        Returns:
            (True, "") if the request is allowed.
            (False, human-readable reason) if blocked — reason is safe to
            display directly in the UI.

        Raises:
            Nothing — on any internal error returns (True, "") to fail open
            and not block legitimate users.
        """
        try:
            now = time.time()

            count: int = st.session_state.get("rate_limit_count", 0)
            last_ts: float = st.session_state.get("rate_limit_last_ts", 0.0)

            if count >= MAX_ANALYSES_PER_SESSION:
                return (
                    False,
                    (
                        f"You've run {MAX_ANALYSES_PER_SESSION} analyses this session. "
                        "Refresh tomorrow or contact me directly."
                    ),
                )

            elapsed = now - last_ts
            if elapsed < ANALYSIS_COOLDOWN_SECONDS and last_ts > 0:
                wait = int(ANALYSIS_COOLDOWN_SECONDS - elapsed)
                return False, f"Please wait {wait} seconds before running another analysis."

            return True, ""

        except Exception as exc:  # noqa: BLE001
            logger.error("SessionRateLimiter.check error: %s", exc)
            return True, ""   # fail open

    def record(self) -> None:
        """
        Record that an analysis was started right now.

        Increments the counter and updates the timestamp.

        Returns:
            None.

        Raises:
            Nothing.
        """
        try:
            st.session_state.rate_limit_count = (
                st.session_state.get("rate_limit_count", 0) + 1
            )
            st.session_state.rate_limit_last_ts = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.error("SessionRateLimiter.record error: %s", exc)

    @property
    def current_count(self) -> int:
        """Return the number of analyses run this session (read-only)."""
        return st.session_state.get("rate_limit_count", 0)
