"""
analytics.py — PostHog event tracking for AI Spend Auditor.

Every public function fails silently — analytics must NEVER crash the app.

Privacy rules enforced here:
  - No raw file contents, prompt text, or user inputs are ever sent
  - Only aggregated numerical metrics and boolean flags go to PostHog
  - Session IDs are anonymous UUIDs generated client-side in session_state

Configuration:
  Set POSTHOG_API_KEY and POSTHOG_HOST in .env (local) or Streamlit Secrets
  (Streamlit Community Cloud).
"""

import logging
import os
import time
import uuid
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostHog initialisation — done once at module import
# ---------------------------------------------------------------------------
_posthog_client = None


def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from Streamlit secrets first, then environment variables.

    Args:
        key:     Secret key name.
        default: Default value if not found anywhere.

    Returns:
        String value of the secret or default.

    Raises:
        Nothing.
    """
    try:
        return st.secrets[key]
    except Exception:  # noqa: BLE001
        return os.getenv(key, default)


def _init_posthog():
    """
    Lazily initialise the PostHog client on first use.

    Returns:
        posthog client instance or None if init fails.

    Raises:
        Nothing.
    """
    global _posthog_client  # noqa: PLW0603
    if _posthog_client is not None:
        return _posthog_client
    try:
        import posthog as ph  # type: ignore

        api_key = _get_secret("POSTHOG_API_KEY")
        host = _get_secret("POSTHOG_HOST", "https://us.i.posthog.com")

        if not api_key:
            logger.warning("POSTHOG_API_KEY not set — analytics disabled.")
            return None

        ph.project_api_key = api_key
        ph.host = host
        ph.disabled = False
        _posthog_client = ph
        return _posthog_client
    except Exception as exc:  # noqa: BLE001
        logger.error("PostHog init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """
    Return a persistent anonymous session UUID stored in session_state.

    The UUID is generated once per browser session.  It is never linked to
    any personally identifiable information.

    Returns:
        UUID4 string.

    Raises:
        Nothing — returns a fresh UUID on any error.
    """
    try:
        if "analytics_session_id" not in st.session_state:
            st.session_state.analytics_session_id = str(uuid.uuid4())
        return st.session_state.analytics_session_id
    except Exception:  # noqa: BLE001
        return str(uuid.uuid4())


def _time_on_page() -> float:
    """
    Return seconds elapsed since the page first loaded in this session.

    Returns:
        Float seconds, or 0.0 if page_load_time was never set.

    Raises:
        Nothing.
    """
    try:
        load_time = st.session_state.get("page_load_time", time.time())
        return round(time.time() - load_time, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def track(event: str, properties: Optional[dict] = None) -> None:
    """
    Send a single event to PostHog.

    Always injects `session_id` and `timestamp_utc` into every event.
    Fails silently — any exception is logged at WARNING level only.

    Args:
        event:      PostHog event name (snake_case string).
        properties: Optional dict of properties (numbers and booleans only —
                    no user content).

    Returns:
        None.

    Raises:
        Nothing.
    """
    try:
        client = _init_posthog()
        if client is None:
            return

        props = properties or {}
        props.setdefault("session_id", get_session_id())
        props.setdefault("timestamp_utc", time.time())

        client.capture(
            distinct_id=get_session_id(),
            event=event,
            properties=props,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PostHog track('%s') failed: %s", event, exc)


# ---------------------------------------------------------------------------
# Convenience wrappers for every tracked event
# — all business logic for what to send lives here, not in app.py
# ---------------------------------------------------------------------------

def track_page_viewed() -> None:
    """Track initial page load with referrer info."""
    try:
        ref = st.query_params.get("ref", "direct")
    except Exception:  # noqa: BLE001
        ref = "direct"
    track("page_viewed", {
        "referrer": ref,
    })


def track_upload_attempted(
    has_openai: bool,
    has_anthropic: bool,
    has_log_file: bool,
    has_app_description: bool,
    openai_kb: float = 0.0,
    anthropic_kb: float = 0.0,
) -> None:
    """Track that the user clicked 'Analyze' and attempted an upload."""
    track("upload_attempted", {
        "has_openai_csv": has_openai,
        "has_anthropic_csv": has_anthropic,
        "has_log_file": has_log_file,
        "has_app_description": has_app_description,
        "openai_file_size_kb": round(openai_kb, 2),
        "anthropic_file_size_kb": round(anthropic_kb, 2),
    })


def track_upload_failed(reason: str, provider: str) -> None:
    """Track a file that failed validation."""
    track("upload_failed", {"reason": reason, "provider": provider})


def track_upload_succeeded(
    has_openai: bool,
    has_anthropic: bool,
    openai_rows: int = 0,
    anthropic_rows: int = 0,
    openai_date_range_days: int = 0,
    anthropic_date_range_days: int = 0,
) -> None:
    """Track successful file parse and validation."""
    track("upload_succeeded", {
        "has_openai": has_openai,
        "has_anthropic": has_anthropic,
        "openai_row_count": openai_rows,
        "anthropic_row_count": anthropic_rows,
        "openai_date_range_days": openai_date_range_days,
        "anthropic_date_range_days": anthropic_date_range_days,
    })


def track_analysis_started(pricing_source: str, session_count: int) -> None:
    """Track that waste detection pipeline began."""
    track("analysis_started", {
        "pricing_source": pricing_source,
        "session_analysis_count": session_count,
    })


def track_analysis_completed(
    duration_seconds: float,
    total_monthly_spend: float,
    total_waste: float,
    rules_triggered: list,
    rules_not_triggered: list,
    models_detected: list,
    has_reasoning: bool,
    has_embedding: bool,
    pricing_source: str,
    groq_success: bool,
    groq_response_time: float,
) -> None:
    """Track completed analysis with aggregate financial metrics."""
    spend = round(total_monthly_spend, 2)
    waste = round(total_waste, 2)
    pct = round((waste / spend * 100) if spend > 0 else 0.0, 1)

    if spend < 200:
        bucket = "<$200"
    elif spend < 500:
        bucket = "$200-500"
    elif spend < 1000:
        bucket = "$500-1k"
    elif spend < 3000:
        bucket = "$1k-3k"
    else:
        bucket = ">$3k"

    track("analysis_completed", {
        "duration_seconds": round(duration_seconds, 2),
        "total_monthly_spend_usd": spend,
        "total_waste_found_usd": waste,
        "waste_percentage": pct,
        "rules_triggered": rules_triggered,
        "rules_not_triggered": rules_not_triggered,
        "spend_bucket": bucket,
        "models_detected": models_detected,
        "has_reasoning_models": has_reasoning,
        "has_embedding_calls": has_embedding,
        "pricing_source": pricing_source,
        "groq_call_success": groq_success,
        "groq_response_time_seconds": round(groq_response_time, 2),
    })


def track_analysis_failed(stage: str, error_type: str, duration: float) -> None:
    """Track a pipeline failure at a specific stage."""
    track("analysis_failed", {
        "stage": stage,
        "error_type": error_type,
        "duration_seconds": round(duration, 2),
    })


def track_rule_evaluated(
    rule_name: str,
    triggered: bool,
    severity: Optional[str],
    saving_usd_monthly: float,
) -> None:
    """Track one rule evaluation result regardless of trigger status."""
    track("rule_evaluated", {
        "rule_name": rule_name,
        "triggered": triggered,
        "severity": severity,
        "saving_usd_monthly": round(saving_usd_monthly, 2),
    })


def track_report_viewed(total_waste: float, rules_shown: int, spend_bucket: str) -> None:
    """Track when the report section first renders."""
    track("report_viewed", {
        "total_waste_usd": round(total_waste, 2),
        "rules_shown": rules_shown,
        "spend_bucket": spend_bucket,
    })


def track_report_section_expanded(section: str, severity: str) -> None:
    """Track when a user expands a finding section."""
    track("report_section_expanded", {
        "section": section,
        "severity": severity,
    })


def track_cta_viewed(total_waste: float) -> None:
    """Track that the CTA section rendered on screen."""
    track("cta_viewed", {
        "total_waste_usd": round(total_waste, 2),
        "time_on_page_seconds": _time_on_page(),
    })


def track_cta_clicked(total_waste: float, rules_triggered_count: int) -> None:
    """Track that the user interacted with the CTA / Tally form."""
    track("cta_clicked", {
        "total_waste_usd": round(total_waste, 2),
        "time_on_page_seconds": _time_on_page(),
        "rules_triggered_count": rules_triggered_count,
    })


def track_csv_injection_attempt(provider: str) -> None:
    """Track a detected CSV injection attempt."""
    track("csv_injection_attempt", {"provider": provider})


def track_rate_limit_hit(session_count: int, reason: str) -> None:
    """Track when a user hits the rate limit."""
    track("rate_limit_hit", {
        "session_analysis_count": session_count,
        "reason": reason,
    })


def track_pricing_fetch_failed(fallback_used: bool) -> None:
    """Track failure to fetch live LiteLLM pricing."""
    track("pricing_fetch_failed", {"fallback_used": fallback_used})


def track_groq_error(error_type: str, retry_attempted: bool) -> None:
    """Track a Groq API error during report generation."""
    track("groq_error", {
        "error_type": error_type,
        "retry_attempted": retry_attempted,
    })
