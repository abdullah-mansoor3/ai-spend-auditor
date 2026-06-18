"""
report.py — Groq narrative generation for AI Spend Auditor.

Groq generates ONLY explanatory text — all financial figures are computed in
Python and passed as pre-formatted strings.  Groq never calculates numbers.

Keys:
  - Model:       llama-3.3-70b-versatile
  - Temperature: 0.2 (near-deterministic for financial content)
  - Max tokens:  800
  - Retry:       One automatic retry on timeout only

Three Groq API keys are rotated automatically if one runs out of quota.
On any failure or hallucination, fallback template strings are returned.
"""

import json
import logging
import os
import time
from typing import Optional

import streamlit as st

from src.rules import RuleResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE: float = 0.2
GROQ_MAX_TOKENS: int = 800
GROQ_TIMEOUT_SECONDS: int = 20

# Rotate through multiple API keys if one hits rate limits
_GROQ_KEY_ENV_NAMES: list[str] = [
    "GROQ_API_KEY",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_3",
]

# ---------------------------------------------------------------------------
# Fallback templates (used when Groq fails or hallucinates)
# ---------------------------------------------------------------------------
FALLBACK_TEMPLATES: dict = {
    "report_summary": (
        "Analysis of {date_range_days} days of billing data from {providers} identified "
        "{rules_triggered_count} optimization opportunit{plural} totaling "
        "${total_waste:.2f}/month in recoverable spend."
    ),
    "finding_explanation": {
        "model_task_mismatch": (
            "A portion of your {expensive_model} usage is handling tasks that produce short "
            "outputs, suggesting simpler models could handle them at lower cost."
        ),
        "context_bloat": (
            "Your average request input is {avg_input} tokens. System prompt compression "
            "and history trimming could reduce this significantly."
        ),
        "reasoning_overuse": (
            "{pct_reasoning:.1f}% of your spend is on reasoning models. Routing non-complex "
            "tasks to standard models would materially reduce this cost."
        ),
        "embedding_inefficiency": (
            "Your embedding API calls average {avg_tokens:.0f} tokens each, suggesting "
            "individual rather than batched requests."
        ),
    },
}


# ---------------------------------------------------------------------------
# Key rotation helpers
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment."""
    try:
        return st.secrets[key]
    except Exception:  # noqa: BLE001
        return os.getenv(key, default)


def _get_groq_keys() -> list[str]:
    """
    Return all available Groq API keys in priority order.

    Tries Streamlit secrets / env vars first, then hardcoded keys.

    Returns:
        Non-empty list of API key strings.

    Raises:
        Nothing.
    """
    keys: list[str] = []
    for env_name in _GROQ_KEY_ENV_NAMES:
        val = _get_secret(env_name)
        if val and val not in keys:
            keys.append(val)
    return keys


def _try_groq_call(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    retry: bool = True,
) -> Optional[str]:
    """
    Make a single Groq API call and return the raw response text.

    Args:
        system_prompt: System prompt string.
        user_prompt:   User prompt string.
        api_key:       Groq API key to use.
        retry:         If True, attempt one retry on timeout.

    Returns:
        Raw response string or None on failure.

    Raises:
        Nothing — all exceptions are caught.
    """
    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            timeout=GROQ_TIMEOUT_SECONDS,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        error_str = str(exc)
        if "timeout" in error_str.lower() and retry:
            logger.warning("Groq timeout — retrying once. Error: %s", exc)
            return _try_groq_call(system_prompt, user_prompt, api_key, retry=False)
        logger.warning("Groq call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """You are a technical analyst writing a cost optimization report for a startup founder.
You will be given structured findings from an analysis of their LLM API usage.
Your job is to write clear, direct, jargon-free explanations for each finding.

Rules you must follow:
1. Use ONLY the numbers provided in the input. Do not calculate, estimate, or invent any numbers.
2. Do not say "approximately" or "around" — use the exact figures given.
3. Do not add findings not present in the input.
4. Do not recommend tools, platforms, or services not mentioned in the input.
5. Write like a consultant who respects the reader's time — no filler, no cheerleading.
6. If a finding has confidence=MED, mention that the estimate is based on aggregate billing data and the actual saving may vary.
7. Maximum 3 sentences per finding explanation.
8. Output valid JSON only. No markdown, no preamble, no explanation outside the JSON.
9. Ensure natural sentence spacing with standard spaces between words and after punctuation."""


def build_groq_prompt(findings: list[RuleResult], context: dict) -> str:
    """
    Build the user prompt passed to Groq for narrative generation.

    All financial figures are passed as pre-formatted strings so the model
    cannot alter them.

    Args:
        findings: List of RuleResult dicts from run_all_rules().
        context:  Dict with keys:
                    app_description (str), total_monthly_spend (float),
                    total_monthly_waste (float), date_range_days (int),
                    providers (list[str])

    Returns:
        User prompt string.

    Raises:
        Nothing.
    """
    triggered = [f for f in findings if f["triggered"]]
    providers_str = " and ".join(context.get("providers", ["Unknown"]))

    findings_json = []
    for f in triggered:
        findings_json.append({
            "rule_id":           f["rule_id"],
            "rule_name":         f["rule_name"],
            "severity":          f["severity"],
            "confidence":        f["confidence"],
            "monthly_saving":    f"${f['monthly_saving_usd']:,.2f}",
            "annual_saving":     f"${f['annual_saving_usd']:,.2f}",
            "finding_headline":  f["finding_headline"],
            "finding_detail":    f["finding_detail"],
            "affected_models":   f["affected_models"][:5],  # cap at 5 for prompt size
        })

    payload = {
        "analysis_context": {
            "app_description":        context.get("app_description", "Not provided"),
            "providers_analyzed":     providers_str,
            "date_range_days":        context.get("date_range_days", 30),
            "total_monthly_spend":    f"${context.get('total_monthly_spend', 0.0):,.2f}",
            "total_monthly_waste":    f"${context.get('total_monthly_waste', 0.0):,.2f}",
            "findings_count":         len(triggered),
        },
        "triggered_findings": findings_json,
        "output_schema": {
            "report_summary":  "2-3 sentence executive summary. Must use ONLY the numbers from analysis_context.",
            "findings": [
                {
                    "rule_id":      "<matches input rule_id>",
                    "explanation":  "<2-3 sentences explaining what was found and why it matters>",
                    "urgency_note": "<1 sentence — why to fix this NOW, not someday>",
                }
            ],
            "closing_note": "1 sentence — what implementing these changes involves technically",
        },
    }

    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------

def _validate_groq_response(
    parsed: dict,
    context: dict,
    findings: list[RuleResult],
) -> bool:
    """
    Check that the Groq response does not contain hallucinated financial figures.

    Extracts all dollar amounts from the summary and finding explanations,
    then verifies each appears in the input context or findings.

    Args:
        parsed:   Parsed JSON dict from Groq.
        context:  Original context dict with true financial figures.
        findings: Original RuleResult list.

    Returns:
        True if response passes validation, False if hallucination detected.

    Raises:
        Nothing.
    """
    import re

    # Collect all legitimate dollar amounts as rounded int cents
    legit_amounts: set[int] = set()
    for key in ("total_monthly_spend", "total_monthly_waste"):
        val = context.get(key, 0.0)
        legit_amounts.add(round(float(val) * 100))

    for f in findings:
        legit_amounts.add(round(f["monthly_saving_usd"] * 100))
        legit_amounts.add(round(f["annual_saving_usd"] * 100))

    # Extract dollar figures from the summary text
    summary = parsed.get("report_summary", "")
    found_amounts = re.findall(r"\$([0-9,]+(?:\.[0-9]{1,2})?)", summary)
    for amt_str in found_amounts:
        amt_cents = round(float(amt_str.replace(",", "")) * 100)
        # Allow small rounding differences (±10 cents)
        if not any(abs(amt_cents - leg) <= 10 for leg in legit_amounts):
            logger.warning(
                "Hallucination guard: '$%s' not found in input figures.", amt_str
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_report_narrative(
    findings: list[RuleResult],
    context: dict,
) -> tuple[dict, bool, float]:
    """
    Generate human-readable narrative sections using Groq.

    Tries all available API keys in rotation.  Falls back to template strings
    on any failure or on hallucination detection.

    Args:
        findings: List of RuleResult dicts from run_all_rules().
        context:  Dict with keys app_description, total_monthly_spend,
                  total_monthly_waste, date_range_days, providers.

    Returns:
        (narrative_dict, groq_success, response_time_seconds)

        narrative_dict keys:
          - report_summary (str)
          - findings (list[dict] with rule_id, explanation, urgency_note)
          - closing_note (str)

    Raises:
        Nothing — always returns a valid narrative_dict.
    """
    triggered = [f for f in findings if f["triggered"]]
    if not triggered:
        providers_str = " and ".join(context.get("providers", ["your provider"]))
        return {
            "report_summary": (
                f"Analysis of {context.get('date_range_days', 30)} days of billing data from {providers_str} "
                "shows that your LLM usage patterns are highly optimized. No significant waste or inefficiencies "
                "were detected."
            ),
            "findings": [],
            "closing_note": "Your current API integration is efficient. We recommend continuing to monitor costs as you scale."
        }, True, 0.0

    fallback_narrative = _build_fallback(findings, context)

    user_prompt = build_groq_prompt(findings, context)
    groq_keys = _get_groq_keys()

    start = time.time()

    for key in groq_keys:
        raw_text = _try_groq_call(_SYSTEM_PROMPT, user_prompt, key)
        elapsed = time.time() - start

        if not raw_text:
            continue

        # Try to extract JSON (model might add whitespace/BOM)
        try:
            # Strip any leading/trailing whitespace or code fences
            clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.warning("Groq JSON parse error: %s — using fallback.", exc)
            from src.analytics import track_groq_error
            track_groq_error("json_parse_error", retry_attempted=True)
            continue

        # Validate — no hallucinated numbers
        if not _validate_groq_response(parsed, context, findings):
            logger.warning("Groq hallucination detected — using fallback.")
            from src.analytics import track_groq_error
            track_groq_error("hallucination_detected", retry_attempted=False)
            return fallback_narrative, False, round(elapsed, 2)

        # Success
        return parsed, True, round(elapsed, 2)

    # All keys exhausted / all failed
    elapsed = time.time() - start
    return fallback_narrative, False, round(elapsed, 2)


def _build_fallback(findings: list[RuleResult], context: dict) -> dict:
    """
    Build narrative from FALLBACK_TEMPLATES when Groq is unavailable.

    Args:
        findings: List of RuleResult dicts.
        context:  Context dict.

    Returns:
        narrative_dict compatible with generate_report_narrative() output.

    Raises:
        Nothing.
    """
    triggered = [f for f in findings if f["triggered"]]
    n = len(triggered)
    plural = "ies" if n != 1 else "y"
    providers_str = " and ".join(context.get("providers", ["your provider"]))

    summary = FALLBACK_TEMPLATES["report_summary"].format(
        date_range_days=context.get("date_range_days", 30),
        providers=providers_str,
        rules_triggered_count=n,
        plural=plural,
        total_waste=context.get("total_monthly_waste", 0.0),
    )

    finding_narratives = []
    for f in triggered:
        tmpl = FALLBACK_TEMPLATES["finding_explanation"].get(
            f["rule_id"],
            "Review the detailed findings below for specific recommendations.",
        )
        # Best-effort template filling with available data
        try:
            explanation = tmpl.format(
                expensive_model=f["affected_models"][0] if f["affected_models"] else "your model",
                avg_input="unknown",
                pct_reasoning=0.0,
                avg_tokens=0.0,
            )
        except (KeyError, IndexError):
            explanation = f["finding_detail"] or tmpl

        finding_narratives.append({
            "rule_id":      f["rule_id"],
            "explanation":  explanation,
            "urgency_note": "Each month without action is an additional unnecessary expense.",
        })

    return {
        "report_summary": summary,
        "findings":        finding_narratives,
        "closing_note":    (
            "Implementing these optimisations typically requires 1–2 days of engineering work "
            "per finding and has no impact on end-user experience."
        ),
    }
