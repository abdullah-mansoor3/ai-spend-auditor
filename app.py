"""
app.py — Main Streamlit entrypoint for AI Spend Auditor.

ROADMAP (do not build these now — build only after v1 has real users)

v2 — Log-based duplicate detection
  - Accept LiteLLM JSON log export as optional third upload
  - Extract prompt text from logs
  - Run sentence-transformers semantic clustering (Rule B)
  - Show exact duplicate prompt pairs as examples in the report

v2 — GitHub code scanner
  - Accept GitHub repo URL
  - Scan Python files for openai/anthropic API calls
  - Produce same waste report from static analysis (no billing data needed)
  - CTA: "I'll implement these optimizations in your codebase"

v3 — Automated monitoring
  - User connects their OpenAI/Anthropic API key (read-only usage scope)
  - Weekly automated audit delivered by email via Resend
  - Pricing: $50/month per workspace

v3 — Multi-workspace support
  - Auth via magic link email
  - Store per-user audit history in Supabase free tier
  - Trend view: waste over time, which rules triggered each month

v4 — LiteLLM proxy integration
  - One-click connect to user's self-hosted LiteLLM proxy
  - Real-time cost monitoring, not batch analysis
"""

import time
import os
import logging

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# Local modules
from src.analytics import (
    track_page_viewed, track_upload_attempted, track_upload_failed,
    track_upload_succeeded, track_analysis_started, track_analysis_completed,
    track_analysis_failed, track_rule_evaluated, track_report_viewed,
    track_report_section_expanded, track_cta_viewed, track_cta_clicked,
    track_csv_injection_attempt, track_rate_limit_hit, track_pricing_fetch_failed,
    track_groq_error, get_session_id,
)
from src.validators import validate_uploaded_file, validate_dataframe, detect_csv_injection, sanitize_text_input
from src.rate_limiter import SessionRateLimiter
from src.parser import parse_openai_csv, parse_anthropic_csv, merge_providers
from src.pricing import load_pricing_table, enrich_dataframe
from src.rules import run_all_rules
from src.report import generate_report_narrative

logger = logging.getLogger(__name__)


def inject_tally_feedback_popup() -> None:
    """Add the Tally feedback popup trigger to the Streamlit page."""
    components.html(
        """
<script>
(function () {
  const FORM_ID = "Pd4ENe";
  const SCRIPT_SRC = "https://tally.so/widgets/embed.js";
  const BUTTON_ID = "tally-feedback-popup-button";

  function install(doc) {
    if (!doc || doc.getElementById(BUTTON_ID)) return;
    const win = doc.defaultView || window;

    if (!doc.querySelector('script[src="' + SCRIPT_SRC + '"]')) {
      const script = doc.createElement("script");
      script.async = true;
      script.src = SCRIPT_SRC;
      script.onload = function () {
        if (win.Tally && win.Tally.loadEmbeds) win.Tally.loadEmbeds();
      };
      doc.head.appendChild(script);
    }

    const button = doc.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.textContent = "Feedback";
    button.setAttribute("data-tally-open", FORM_ID);
    button.setAttribute("data-tally-emoji-text", "👋");
    button.setAttribute("data-tally-emoji-animation", "wave");
    button.addEventListener("click", function () {
      if (win.Tally && win.Tally.openPopup) {
        win.Tally.openPopup(FORM_ID, {
          emoji: { text: "👋", animation: "wave" }
        });
      }
    });
    button.style.cssText = [
      "position:fixed",
      "right:18px",
      "bottom:18px",
      "z-index:2147483647",
      "border:1px solid rgba(0,200,150,0.65)",
      "border-radius:999px",
      "background:#00C896",
      "color:#07110e",
      "font:600 14px/1.1 Inter, sans-serif",
      "padding:11px 16px",
      "box-shadow:0 8px 24px rgba(0,0,0,0.28)",
      "cursor:pointer"
    ].join(";");

    doc.body.appendChild(button);
    if (win.Tally && win.Tally.loadEmbeds) win.Tally.loadEmbeds();
  }

  try {
    install(window.parent.document);
  } catch (err) {
    install(document);
  }
})();
</script>
        """,
        height=0,
    )

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Spend Auditor — Find your LLM waste",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_tally_feedback_popup()

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero headline */
.hero-headline {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.2;
    background: linear-gradient(135deg, #E6EDF3 0%, #00C896 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #8B949E;
    line-height: 1.6;
    margin-bottom: 1.8rem;
}

/* Trust bar */
.trust-bar {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
    padding: 0.9rem 1.2rem;
    background: #161B22;
    border-radius: 10px;
    border: 1px solid #21262D;
}
.trust-item {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.85rem;
    color: #8B949E;
}
.trust-item span.icon { font-size: 1.1rem; }
.trust-item span.label { color: #C9D1D9; }

/* Metric cards */
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-size: 0.78rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #E6EDF3;
}
.metric-value.green { color: #00C896; }

/* Severity badges */
.badge-HIGH { background:#ff4444; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }
.badge-MED  { background:#f0a500; color:#000; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }
.badge-LOW  { background:#4a90d9; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }

/* CTA section */
.cta-box {
    background: linear-gradient(135deg, #161B22 0%, #0d1117 100%);
    border: 1px solid #00C896;
    border-radius: 14px;
    padding: 2rem 2rem 1.5rem;
    margin-top: 2rem;
}
.cta-heading {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 0.5rem;
}
.cta-body { color: #8B949E; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem; }
.cta-bullets { color: #C9D1D9; font-size: 0.9rem; line-height: 1.8; }

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #21262D;
    margin: 2rem 0;
}

/* Check line */
.check-line { color: #00C896; font-size: 0.88rem; margin: 0.3rem 0; }

/* Footer */
.footer-text {
    font-size: 0.75rem;
    color: #484F58;
    text-align: center;
    margin-top: 3rem;
    line-height: 1.6;
}

/* Pulse animation for analyze button */
@keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(0,200,150,0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(0,200,150,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,200,150,0); }
}
div[data-testid="stButton"] > button[kind="primary"] {
    animation: pulse-green 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
def _init_session_state() -> None:
    """Initialise all session_state keys with defaults on first load."""
    defaults = {
        "analysis_complete": False,
        "analysis_results": {},
        "last_run_timestamp": 0.0,
        "page_load_time": time.time(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()

# ─────────────────────────────────────────────
# Analytics — page viewed (once per session)
# ─────────────────────────────────────────────
if "page_tracked" not in st.session_state:
    track_page_viewed()
    st.session_state.page_tracked = True


# ─────────────────────────────────────────────
# SECTION 1 — Hero
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-headline">Find out exactly where your AI bill is wasted.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload your OpenAI or Anthropic billing export. '
    "Get a line-itemized report of recoverable spend — in dollars, with exact actions. "
    "Free. No account required.</p>",
    unsafe_allow_html=True,
)

st.markdown("""
<div class="trust-bar">
  <div class="trust-item"><span class="icon">🔒</span><span class="label">Your data never leaves this session</span></div>
  <div class="trust-item"><span class="icon">⚡</span><span class="label">Analysis in under 30 seconds</span></div>
  <div class="trust-item"><span class="icon">💰</span><span class="label">Average finding: $200–800/month recoverable</span></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 2 — Upload zone
# ─────────────────────────────────────────────
with st.expander("📋 How to export your billing CSV", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI**")
        st.markdown("1. Go to [platform.openai.com](https://platform.openai.com)")
        st.markdown("2. Click **Usage** in the left sidebar")
        st.markdown("3. Click **Export** (top right) → Download CSV")
    with col2:
        st.markdown("**Anthropic**")
        st.markdown("1. Go to [console.anthropic.com](https://console.anthropic.com)")
        st.markdown("2. Click **Usage** in the left sidebar")
        st.markdown("3. Click **Export** → Download CSV")

st.markdown("---")

openai_file = st.file_uploader(
    "OpenAI billing CSV",
    type=["csv"],
    key="openai_upload",
    help="Required. Export from platform.openai.com → Usage → Export",
)

anthropic_file = st.file_uploader(
    "Anthropic billing CSV (optional — add if you use both)",
    type=["csv"],
    key="anthropic_upload",
    help="Optional. Export from console.anthropic.com → Usage → Export",
)

app_description_raw = st.text_input(
    "Describe your app in one sentence (optional)",
    placeholder="e.g. a customer support chatbot for e-commerce stores",
    max_chars=500,
    help="Helps make the report more specific to your use case",
)
app_description = sanitize_text_input(app_description_raw)

analyze_clicked = st.button("Analyze my spend →", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 3 — Analysis pipeline
# ─────────────────────────────────────────────
def _file_kb(f) -> float:
    """Return file size in KB, 0.0 if not available."""
    try:
        return round(f.size / 1024, 2) if f else 0.0
    except Exception:
        return 0.0


def _date_range_days(df: pd.DataFrame) -> int:
    """Return the number of days spanned by the 'date' column."""
    try:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return 0
        return max((dates.max() - dates.min()).days, 1)
    except Exception:
        return 0


def run_analysis(openai_f, anthropic_f, app_desc: str) -> None:
    """
    Execute the full analysis pipeline with progress feedback.

    Parses files, loads pricing, runs rules, generates Groq narrative, then
    stores results in session_state for persistent display.

    Args:
        openai_f:    Streamlit UploadedFile or None.
        anthropic_f: Streamlit UploadedFile or None.
        app_desc:    Sanitised app description string.

    Returns:
        None (writes to st.session_state).

    Raises:
        Nothing — all errors are caught and displayed to the user.
    """
    pipeline_start = time.time()

    # Track upload attempt
    track_upload_attempted(
        has_openai=openai_f is not None,
        has_anthropic=anthropic_f is not None,
        has_log_file=False,
        has_app_description=bool(app_desc),
        openai_kb=_file_kb(openai_f),
        anthropic_kb=_file_kb(anthropic_f),
    )

    progress_bar = st.progress(0, text="Starting analysis...")
    status_placeholder = st.empty()

    # ── Step 1: Parse ──────────────────────────────────────────
    status_placeholder.markdown("**Step 1/4:** Parsing billing data...")
    progress_bar.progress(0.25, text="Parsing billing data...")

    openai_df = None
    anthropic_df = None

    if openai_f is not None:
        try:
            openai_df = parse_openai_csv(openai_f)
        except ValueError as exc:
            track_upload_failed(str(exc)[:80], "openai")
            track_analysis_failed("parse", type(exc).__name__, time.time() - pipeline_start)
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"⚠️ OpenAI CSV error: {exc}")
            return

    if anthropic_f is not None:
        try:
            anthropic_df = parse_anthropic_csv(anthropic_f)
        except ValueError as exc:
            track_upload_failed(str(exc)[:80], "anthropic")
            track_analysis_failed("parse", type(exc).__name__, time.time() - pipeline_start)
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"⚠️ Anthropic CSV error: {exc}")
            return

    if openai_df is None and anthropic_df is None:
        progress_bar.empty()
        status_placeholder.empty()
        st.warning("Please upload at least one billing CSV file.")
        return

    try:
        unified_df = merge_providers(openai_df, anthropic_df)
    except ValueError as exc:
        track_analysis_failed("parse", "merge_error", time.time() - pipeline_start)
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"⚠️ Data merge error: {exc}")
        return

    # Track upload success
    track_upload_succeeded(
        has_openai=openai_df is not None,
        has_anthropic=anthropic_df is not None,
        openai_rows=len(openai_df) if openai_df is not None else 0,
        anthropic_rows=len(anthropic_df) if anthropic_df is not None else 0,
        openai_date_range_days=_date_range_days(openai_df) if openai_df is not None else 0,
        anthropic_date_range_days=_date_range_days(anthropic_df) if anthropic_df is not None else 0,
    )

    # ── Step 2: Pricing ────────────────────────────────────────
    status_placeholder.markdown("**Step 2/4:** Loading current pricing...")
    progress_bar.progress(0.50, text="Loading current pricing...")

    try:
        pricing, pricing_source = load_pricing_table()
    except Exception as exc:
        track_pricing_fetch_failed(fallback_used=True)
        pricing_source = "fallback_hardcoded"
        from src.pricing import HARDCODED_FALLBACK
        pricing = HARDCODED_FALLBACK

    if pricing_source != "live":
        track_pricing_fetch_failed(fallback_used=True)

    track_analysis_started(pricing_source, st.session_state.get("rate_limit_count", 0))

    try:
        enriched_df = enrich_dataframe(unified_df, pricing)
    except Exception as exc:
        track_analysis_failed("pricing", type(exc).__name__, time.time() - pipeline_start)
        progress_bar.empty()
        status_placeholder.empty()
        st.error("⚠️ Pricing enrichment failed. Please try again.")
        return

    # ── Step 3: Rules ──────────────────────────────────────────
    status_placeholder.markdown("**Step 3/4:** Detecting waste patterns...")
    progress_bar.progress(0.75, text="Detecting waste patterns...")

    try:
        results = run_all_rules(enriched_df, pricing)
    except Exception as exc:
        track_analysis_failed("rules", type(exc).__name__, time.time() - pipeline_start)
        progress_bar.empty()
        status_placeholder.empty()
        st.error("⚠️ Rule analysis failed. Please try again.")
        return

    for r in results:
        track_rule_evaluated(r["rule_id"], r["triggered"], r["severity"], r["monthly_saving_usd"])

    # ── Step 4: Groq narrative ─────────────────────────────────
    status_placeholder.markdown("**Step 4/4:** Generating report...")
    progress_bar.progress(1.0, text="Generating report...")

    total_monthly_spend = enriched_df["cost_calculated"].sum() * (30.0 / max(_date_range_days(enriched_df), 1))
    total_waste = sum(r["monthly_saving_usd"] for r in results if r["triggered"])
    providers_used = sorted(enriched_df["provider"].unique().tolist())
    date_range = _date_range_days(enriched_df)

    context = {
        "app_description":    app_desc,
        "total_monthly_spend": total_monthly_spend,
        "total_monthly_waste": total_waste,
        "date_range_days":     date_range,
        "providers":           providers_used,
    }

    groq_start = time.time()
    narrative, groq_success, groq_rt = generate_report_narrative(results, context)
    if not groq_success:
        track_groq_error("generation_failed", retry_attempted=True)

    duration = time.time() - pipeline_start
    triggered_names = [r["rule_id"] for r in results if r["triggered"]]
    not_triggered_names = [r["rule_id"] for r in results if not r["triggered"]]
    models_detected = sorted(enriched_df["model_normalized"].unique().tolist())
    has_reasoning = bool(enriched_df["is_reasoning"].any())
    has_embedding = bool(enriched_df["is_embedding"].any())

    if total_monthly_spend < 200:
        bucket = "<$200"
    elif total_monthly_spend < 500:
        bucket = "$200-500"
    elif total_monthly_spend < 1000:
        bucket = "$500-1k"
    elif total_monthly_spend < 3000:
        bucket = "$1k-3k"
    else:
        bucket = ">$3k"

    track_analysis_completed(
        duration_seconds=duration,
        total_monthly_spend=total_monthly_spend,
        total_waste=total_waste,
        rules_triggered=triggered_names,
        rules_not_triggered=not_triggered_names,
        models_detected=models_detected,
        has_reasoning=has_reasoning,
        has_embedding=has_embedding,
        pricing_source=pricing_source,
        groq_success=groq_success,
        groq_response_time=groq_rt,
    )

    progress_bar.empty()
    status_placeholder.empty()

    # Persist results
    st.session_state.analysis_complete = True
    st.session_state.analysis_results = {
        "results":             results,
        "narrative":           narrative,
        "total_monthly_spend": total_monthly_spend,
        "total_waste":         total_waste,
        "date_range":          date_range,
        "providers":           providers_used,
        "models_detected":     models_detected,
        "spend_bucket":        bucket,
        "triggered_names":     triggered_names,
    }


# ─────────────────────────────────────────────
# Handle analyze button click
# ─────────────────────────────────────────────
if analyze_clicked:
    limiter = SessionRateLimiter()
    allowed, reason = limiter.check()

    if not allowed:
        track_rate_limit_hit(limiter.current_count, reason)
        st.warning(f"⏳ {reason}")
    elif openai_file is None and anthropic_file is None:
        st.warning("⚠️ Please upload at least one billing CSV to continue.")
    else:
        limiter.record()
        run_analysis(openai_file, anthropic_file, app_description)


# ─────────────────────────────────────────────
# SECTION 4 — Report output (persisted)
# ─────────────────────────────────────────────
if st.session_state.analysis_complete and st.session_state.analysis_results:
    res = st.session_state.analysis_results
    results: list = res["results"]
    narrative: dict = res["narrative"]
    total_spend: float = res["total_monthly_spend"]
    total_waste: float = res["total_waste"]
    bucket: str = res["spend_bucket"]
    triggered_names: list = res["triggered_names"]

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 📊 Your AI Spend Report")

    # ── Header metrics ─────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Monthly Spend</div>
          <div class="metric-value">${total_spend:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Recoverable Waste</div>
          <div class="metric-value green">${total_waste:,.0f}/mo</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Annual Impact</div>
          <div class="metric-value green">${total_waste * 12:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary ────────────────────────────────────────────────
    summary_text = narrative.get("report_summary", "")
    if summary_text:
        st.info(f"📝 {summary_text}")

    track_report_viewed(total_waste, len(triggered_names), bucket)

    # ── Findings ───────────────────────────────────────────────
    triggered_results = [r for r in results if r["triggered"]]
    not_triggered_results = [r for r in results if not r["triggered"]]

    if not triggered_results:
        st.success("✅ Your usage patterns look well-optimised. No significant waste patterns detected.")
    else:
        st.markdown("### 🔍 Findings")

        # Build narrative lookup
        narrative_findings = {
            f["rule_id"]: f for f in narrative.get("findings", [])
        }

        for rule in triggered_results:
            rid = rule["rule_id"]
            sev = rule["severity"] or "LOW"
            saving = rule["monthly_saving_usd"]

            # Special label for embedding rule (no dollar saving)
            if rid == "embedding_inefficiency":
                label = f"[{sev}] {rule['rule_name']} — No cost saving, but reduces latency risk"
            else:
                label = f"[{sev}] {rule['rule_name']} — save ${saving:,.2f}/month"

            with st.expander(label, expanded=(sev == "HIGH")):
                track_report_section_expanded(rid, sev)

                # Groq explanation
                nf = narrative_findings.get(rid, {})
                explanation = nf.get("explanation", rule["finding_detail"])
                if explanation:
                    st.markdown(explanation)

                # Metric row
                if rule["affected_models"]:
                    st.markdown("**Affected models:**")
                    for m in rule["affected_models"][:6]:
                        st.markdown(f"  - `{m}`")

                # Confidence note
                if rule["confidence"] == "MED":
                    st.caption("ℹ️ Confidence: MEDIUM — estimate based on aggregate billing data; actual saving may vary.")
                else:
                    st.caption("✅ Confidence: HIGH")

                # Recommended action (deterministic — not from Groq)
                if rule["recommended_action"]:
                    st.markdown("**Recommended action:**")
                    st.markdown(rule["recommended_action"])

                # Urgency note from Groq
                urgency = nf.get("urgency_note", "")
                if urgency:
                    st.warning(f"⏰ {urgency}")

        # Not triggered summary
        if not_triggered_results:
            st.markdown("<br>", unsafe_allow_html=True)
            for rule in not_triggered_results:
                st.markdown(
                    f'<div class="check-line">✓ No {rule["rule_name"].lower()} issues detected</div>',
                    unsafe_allow_html=True,
                )

    # Closing note
    closing = narrative.get("closing_note", "")
    if closing:
        st.caption(f"💡 {closing}")

    # ── SECTION 5 — CTA ───────────────────────────────────────
    track_cta_viewed(total_waste)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
<div class="cta-box">
  <div class="cta-heading">Want these implemented?</div>
  <div class="cta-body">
    I can implement the top optimizations from this report in 5 days for a fixed price of <strong style="color:#00C896">$400</strong>.
    No hourly billing, no scope creep.
  </div>
  <div class="cta-bullets">
    Typically included:<br>
    • Model routing changes with testing<br>
    • Semantic caching setup<br>
    • Context compression for your longest prompts<br>
    • Configuration-level fixes (batching, etc.)
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tally embed
    tally_html = """
<iframe data-tally-src="https://tally.so/embed/aQlMbq?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"
  loading="lazy" width="100%" height="644" frameborder="0"
  marginheight="0" marginwidth="0" title="AI Price Audit"></iframe>
<script>
var d=document,w="https://tally.so/widgets/embed.js",
v=function(){"undefined"!=typeof Tally?Tally.loadEmbeds():d.querySelectorAll("iframe[data-tally-src]:not([src])").forEach(function(e){e.src=e.dataset.tallySrc})};
if("undefined"!=typeof Tally)v();
else if(d.querySelector('script[src="'+w+'"]')==null){var s=d.createElement("script");s.src=w,s.onload=v,s.onerror=v,d.body.appendChild(s);}
</script>
"""
    components.html(tally_html, height=680, scrolling=False)

    st.markdown(
        '<p style="font-size:0.83rem;color:#484F58;text-align:center;margin-top:1rem;">'
        "Prefer to just ask a question? "
        '<a href="mailto:abdullah.binmansoor4@gmail.com" style="color:#00C896;">abdullah.binmansoor4@gmail.com</a>'
        ' or visit my <a href="https://portfolio-website-4lcxy4pa2-abdullah-mansoor3s-projects.vercel.app/" target="_blank" style="color:#00C896;">personal website</a>.'
        "</p>",
        unsafe_allow_html=True,
    )

    # CTA click tracking via injected JS
    components.html("""
<script>
document.querySelectorAll('a[href*="tally.so"], iframe[src*="tally.so"]').forEach(function(el){
  el.addEventListener('click', function(){
    fetch('/healthz').catch(function(){});
    console.log('cta_clicked');
  });
});
</script>
""", height=0)


# ─────────────────────────────────────────────
# SECTION 6 — Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
  Built by <a href="https://portfolio-website-4lcxy4pa2-abdullah-mansoor3s-projects.vercel.app/" target="_blank" style="color:#00C896;text-decoration:none;">Abdullah</a> &nbsp;|&nbsp; Rawalpindi, Pakistan → Serving US/UK/AU markets<br>
  This tool processes your data in memory only. Nothing is stored or transmitted
  except aggregate usage metrics to PostHog for product improvement.
</div>
""", unsafe_allow_html=True)
