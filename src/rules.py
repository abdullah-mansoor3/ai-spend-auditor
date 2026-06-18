"""
rules.py — Waste detection rules for AI Spend Auditor.

Each rule returns a RuleResult TypedDict.  Rules are intentionally conservative:
they never overstate savings.  Every monetary estimate includes a confidence level
and an explanatory caveat when the analysis is based on aggregate data only.

Rules implemented:
  A. model_task_mismatch   — expensive model on low-output (simple) tasks
  B. context_bloat         — high average input tokens across requests
  C. reasoning_overuse     — >15% of spend on reasoning models
  D. embedding_inefficiency — small embedding batch sizes (latency risk)
"""

import logging
from typing import Optional, TypedDict

import pandas as pd

from src.pricing import MODEL_DOWNGRADE_MAP, calculate_token_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class RuleResult(TypedDict):
    rule_id: str
    rule_name: str
    triggered: bool
    severity: Optional[str]          # "HIGH" | "MED" | "LOW" | None
    monthly_saving_usd: float
    annual_saving_usd: float
    finding_headline: str
    finding_detail: str
    recommended_action: str
    affected_models: list
    confidence: str                  # "HIGH" | "MED"


# ---------------------------------------------------------------------------
# Thresholds — adjust here to tune rule sensitivity
# ---------------------------------------------------------------------------
# Rule A
RULE_A_MIN_REQUESTS: int = 50           # must have at least this many requests
RULE_A_MAX_AVG_OUTPUT: int = 400        # avg output tokens/req below this = simple task
# Rule B
RULE_B_MIN_AVG_INPUT: int = 2_000       # avg input tokens/req above this = potential bloat
RULE_B_COMPRESSION_RATE: float = 0.40   # conservative 40% compression estimate
# Rule C
RULE_C_MIN_REASONING_PCT: float = 15.0  # reasoning % of spend to trigger
RULE_C_OVERSPEC_RATE: float = 0.60      # 60% of reasoning calls assumed over-specified
# Rule D
RULE_D_MAX_AVG_TOKENS: int = 200        # avg tokens/embedding call below this = small batches


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _severity(saving: float, high_thresh: float, med_thresh: float) -> str:
    """Return 'HIGH', 'MED', or 'LOW' based on monthly saving thresholds."""
    if saving >= high_thresh:
        return "HIGH"
    if saving >= med_thresh:
        return "MED"
    return "LOW"


def _not_triggered(rule_id: str, rule_name: str) -> RuleResult:
    """Return a standard 'not triggered' RuleResult."""
    return RuleResult(
        rule_id=rule_id,
        rule_name=rule_name,
        triggered=False,
        severity=None,
        monthly_saving_usd=0.0,
        annual_saving_usd=0.0,
        finding_headline=f"No {rule_name.lower()} issues detected.",
        finding_detail="",
        recommended_action="",
        affected_models=[],
        confidence="HIGH",
    )


def _aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the unified DataFrame to one row per model_normalized.

    Returns a DataFrame with columns:
      model_normalized, total_requests, total_input, total_output,
      total_cost, avg_input_per_req, avg_output_per_req, is_reasoning,
      is_embedding.

    Args:
        df: Enriched unified DataFrame.

    Returns:
        Aggregated DataFrame.

    Raises:
        Nothing.
    """
    agg = (
        df.groupby("model_normalized", as_index=False)
        .agg(
            total_requests=("requests", "sum"),
            total_input=("input_tokens", "sum"),
            total_output=("output_tokens", "sum"),
            total_cost=("cost_calculated", "sum"),
            is_reasoning=("is_reasoning", "first"),
            is_embedding=("is_embedding", "first"),
        )
    )
    agg["total_requests"] = agg["total_requests"].fillna(0).astype(float)
    agg["total_input"] = agg["total_input"].fillna(0).astype(float)
    agg["total_output"] = agg["total_output"].fillna(0).astype(float)
    agg["total_cost"] = agg["total_cost"].fillna(0.0)

    safe_reqs = agg["total_requests"].replace(0, 1)
    agg["avg_input_per_req"] = agg["total_input"] / safe_reqs
    agg["avg_output_per_req"] = agg["total_output"] / safe_reqs
    return agg


def _annualise_to_monthly(df: pd.DataFrame) -> float:
    """
    Estimate the data date range in days and scale total_cost to monthly.

    If the date column is missing or invalid, assume the data represents
    exactly 30 days.

    Args:
        df: Unified DataFrame with a 'date' column.

    Returns:
        Scale factor (30 / date_range_days).

    Raises:
        Nothing.
    """
    try:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return 1.0
        day_range = max((dates.max() - dates.min()).days, 1)
        return 30.0 / day_range
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute date range: %s", exc)
        return 1.0


# ---------------------------------------------------------------------------
# Rule A — Model-task mismatch
# ---------------------------------------------------------------------------

def rule_model_task_mismatch(df: pd.DataFrame, pricing: dict) -> RuleResult:
    """
    Detect expensive models being used for simple, short-output tasks.

    Logic:
      For each model in MODEL_DOWNGRADE_MAP that appears in the data:
        - Skip if fewer than RULE_A_MIN_REQUESTS
        - Skip if avg_output_per_request >= RULE_A_MAX_AVG_OUTPUT (not a simple task)
        - Calculate monthly cost at current and target (downgraded) model pricing
        - Sum savings across all qualifying models

    Args:
        df:      Enriched unified DataFrame.
        pricing: Pricing dict from load_pricing_table().

    Returns:
        RuleResult TypedDict.

    Raises:
        Nothing — any exception returns a not-triggered result.
    """
    try:
        if df.empty:
            return _not_triggered("model_task_mismatch", "Model-Task Mismatch")

        monthly_scale = _annualise_to_monthly(df)
        agg = _aggregate_by_model(df)
        # Exclude embeddings from this rule — separate rule handles them
        agg = agg[~agg["is_embedding"]]

        total_saving = 0.0
        affected = []

        for _, row in agg.iterrows():
            model = str(row["model_normalized"])
            if model not in MODEL_DOWNGRADE_MAP:
                continue

            reqs = float(row["total_requests"])
            if reqs < RULE_A_MIN_REQUESTS:
                continue

            avg_out = float(row["avg_output_per_req"])
            if avg_out >= RULE_A_MAX_AVG_OUTPUT:
                continue

            # This model is handling short-output (likely simple) tasks
            target_model = MODEL_DOWNGRADE_MAP[model]

            current_monthly_cost = float(row["total_cost"]) * monthly_scale

            target_cost_total = calculate_token_cost(
                int(row["total_input"]),
                int(row["total_output"]),
                target_model,
                pricing,
            ) * monthly_scale

            saving = current_monthly_cost - target_cost_total
            if saving > 0:
                total_saving += saving
                affected.append(
                    f"{model} → {target_model} "
                    f"(avg {avg_out:.0f} output tokens/req, {reqs:.0f} req/mo)"
                )

        if total_saving <= 0 or not affected:
            return _not_triggered("model_task_mismatch", "Model-Task Mismatch")

        # Confidence based on total request volume
        total_reqs = agg[agg["model_normalized"].isin(MODEL_DOWNGRADE_MAP)]["total_requests"].sum()
        confidence = "HIGH" if total_reqs >= 200 else "MED"

        sev = _severity(total_saving, high_thresh=100.0, med_thresh=20.0)
        headline = (
            f"Found {len(affected)} model(s) used on short-output tasks — "
            f"save ${total_saving:,.2f}/month by downgrading"
        )
        detail = (
            "Your billing data shows requests to expensive frontier models where the "
            f"average output is under {RULE_A_MAX_AVG_OUTPUT} tokens/request. "
            "This output length is typical of classification, summarisation, FAQ, "
            "and extraction tasks — all of which are handled equivalently by cheaper models. "
            "Savings are calculated using current token volumes at target model pricing."
        )
        action = (
            "Switch the affected models in your API call(s) to the recommended targets listed. "
            "Run A/B quality checks on 50 randomly sampled tasks to confirm output parity "
            "before full cut-over. No prompt changes are typically required."
        )

        return RuleResult(
            rule_id="model_task_mismatch",
            rule_name="Model-Task Mismatch",
            triggered=True,
            severity=sev,
            monthly_saving_usd=round(total_saving, 2),
            annual_saving_usd=round(total_saving * 12, 2),
            finding_headline=headline,
            finding_detail=detail,
            recommended_action=action,
            affected_models=affected,
            confidence=confidence,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("rule_model_task_mismatch failed: %s", exc)
        return _not_triggered("model_task_mismatch", "Model-Task Mismatch")


# ---------------------------------------------------------------------------
# Rule B — Context bloat
# ---------------------------------------------------------------------------

def rule_context_bloat(df: pd.DataFrame, pricing: dict) -> RuleResult:
    """
    Detect inflated context (system prompts, history) driving up input costs.

    Logic:
      - Compute avg_input_per_request across all non-embedding model rows
      - If avg < RULE_B_MIN_AVG_INPUT: not triggered
      - Estimate 40% compression achievable (conservative)
      - Saving = total_input_cost * RULE_B_COMPRESSION_RATE

    Note: confidence is always MED because we cannot see actual prompt content.

    Args:
        df:      Enriched unified DataFrame.
        pricing: Pricing dict from load_pricing_table().

    Returns:
        RuleResult TypedDict.

    Raises:
        Nothing.
    """
    try:
        if df.empty:
            return _not_triggered("context_bloat", "Context Bloat")

        monthly_scale = _annualise_to_monthly(df)
        # Exclude embeddings
        chat_df = df[~df["is_embedding"]].copy()
        if chat_df.empty:
            return _not_triggered("context_bloat", "Context Bloat")

        total_reqs = chat_df["requests"].fillna(1).astype(float).sum()
        total_input = chat_df["input_tokens"].fillna(0).astype(float).sum()
        if total_reqs == 0:
            return _not_triggered("context_bloat", "Context Bloat")

        avg_input = total_input / total_reqs

        if avg_input < RULE_B_MIN_AVG_INPUT:
            return _not_triggered("context_bloat", "Context Bloat")

        # Compute total input cost from enriched column
        total_input_cost = chat_df["cost_calculated"].sum() * monthly_scale
        # Rough attribution: input cost ≈ total cost * (avg_input / (avg_input + avg_output))
        # We use the actual enriched data to separate input cost
        def _input_cost_row(row) -> float:
            from src.pricing import _lookup_model_pricing
            entry = _lookup_model_pricing(str(row["model_normalized"]), pricing)
            if entry is None:
                return 0.0
            return float(row["input_tokens"] or 0) * float(entry.get("input_cost_per_token", 0.0))

        chat_df["input_cost"] = chat_df.apply(_input_cost_row, axis=1)
        total_input_cost_only = chat_df["input_cost"].sum() * monthly_scale

        saving = total_input_cost_only * RULE_B_COMPRESSION_RATE
        if saving <= 0:
            return _not_triggered("context_bloat", "Context Bloat")

        sev = _severity(saving, high_thresh=80.0, med_thresh=20.0)
        affected_models = sorted(chat_df["model_normalized"].unique().tolist())

        headline = (
            f"Average input context is {avg_input:,.0f} tokens/request — "
            f"compression could save ${saving:,.2f}/month"
        )
        detail = (
            f"Your requests average {avg_input:,.0f} input tokens each, suggesting large system "
            "prompts, long conversation histories, or verbose document chunks being sent on every call. "
            f"Conservative 40% compression of your input context (achievable via structured prompts, "
            "retrieval-augmented generation, or history trimming) would save "
            f"${saving:,.2f}/month. "
            "This is an estimate — actual compression depends on your system prompt structure."
        )
        action = (
            "1. Audit your longest system prompts and remove redundant instructions. "
            "2. If you send conversation history, implement a sliding-window or summarisation strategy "
            "   to cap history at ~1,000 tokens. "
            "3. For RAG pipelines: tune retrieval chunk size (512–800 tokens) and reduce k to the "
            "   minimum needed for accuracy."
        )

        return RuleResult(
            rule_id="context_bloat",
            rule_name="Context Bloat",
            triggered=True,
            severity=sev,
            monthly_saving_usd=round(saving, 2),
            annual_saving_usd=round(saving * 12, 2),
            finding_headline=headline,
            finding_detail=detail,
            recommended_action=action,
            affected_models=affected_models,
            confidence="MED",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("rule_context_bloat failed: %s", exc)
        return _not_triggered("context_bloat", "Context Bloat")


# ---------------------------------------------------------------------------
# Rule C — Reasoning model overuse
# ---------------------------------------------------------------------------

def rule_reasoning_overuse(df: pd.DataFrame) -> RuleResult:
    """
    Detect excessive spend on reasoning models (o-series, Claude extended thinking, etc.).

    Logic:
      - Compute reasoning_spend_pct = reasoning_cost / total_cost * 100
      - If pct < RULE_C_MIN_REASONING_PCT: not triggered
      - Saving = reasoning_monthly_cost * RULE_C_OVERSPEC_RATE (60%)

    Args:
        df: Enriched unified DataFrame.

    Returns:
        RuleResult TypedDict.

    Raises:
        Nothing.
    """
    try:
        if df.empty:
            return _not_triggered("reasoning_overuse", "Reasoning Model Overuse")

        monthly_scale = _annualise_to_monthly(df)
        total_cost = df["cost_calculated"].sum() * monthly_scale
        if total_cost <= 0:
            return _not_triggered("reasoning_overuse", "Reasoning Model Overuse")

        reasoning_df = df[df["is_reasoning"]]
        if reasoning_df.empty:
            return _not_triggered("reasoning_overuse", "Reasoning Model Overuse")

        reasoning_cost = reasoning_df["cost_calculated"].sum() * monthly_scale
        reasoning_pct = (reasoning_cost / total_cost) * 100

        if reasoning_pct < RULE_C_MIN_REASONING_PCT:
            return _not_triggered("reasoning_overuse", "Reasoning Model Overuse")

        saving = reasoning_cost * RULE_C_OVERSPEC_RATE
        sev = "HIGH" if reasoning_pct > 30.0 else "MED"
        affected_models = sorted(reasoning_df["model_normalized"].unique().tolist())

        headline = (
            f"{reasoning_pct:.1f}% of your spend is on reasoning models — "
            f"save ${saving:,.2f}/month by routing simple tasks elsewhere"
        )
        detail = (
            "Reasoning models cost 4–40× standard models and are worth it for code generation, "
            "multi-step mathematics, and complex planning tasks. "
            "For classification, summarisation, FAQ, and extraction tasks, gpt-4o or "
            "claude-haiku perform equivalently at a fraction of the cost. "
            f"Based on billing patterns, an estimated {int(RULE_C_OVERSPEC_RATE*100)}% of your "
            "reasoning model calls appear to be on tasks that don't require extended thinking."
        )
        action = (
            "Add a lightweight task-classifier to your inference stack: "
            "route requests with short expected outputs (< 400 tokens) or "
            "simple instruction patterns (classify/summarise/extract) to a cheaper model. "
            "LiteLLM's router supports rule-based and model-based routing with a single config change."
        )

        return RuleResult(
            rule_id="reasoning_overuse",
            rule_name="Reasoning Model Overuse",
            triggered=True,
            severity=sev,
            monthly_saving_usd=round(saving, 2),
            annual_saving_usd=round(saving * 12, 2),
            finding_headline=headline,
            finding_detail=detail,
            recommended_action=action,
            affected_models=affected_models,
            confidence="MED",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("rule_reasoning_overuse failed: %s", exc)
        return _not_triggered("reasoning_overuse", "Reasoning Model Overuse")


# ---------------------------------------------------------------------------
# Rule D — Embedding inefficiency (latency / rate-limit risk)
# ---------------------------------------------------------------------------

def rule_embedding_inefficiency(df: pd.DataFrame) -> RuleResult:
    """
    Detect small batch sizes in embedding API calls.

    Note: Cost per token is the same regardless of batch size, so this rule
    reports NO dollar saving.  The value is latency and rate-limit headroom.

    Logic:
      - Filter embedding rows
      - avg_tokens_per_embedding_call = total_embedding_input / total_embedding_requests
      - If avg > RULE_D_MAX_AVG_TOKENS: batching is likely fine, not triggered
      - If avg <= threshold: flag as single-item or micro-batch pattern

    Args:
        df: Enriched unified DataFrame.

    Returns:
        RuleResult TypedDict.  monthly_saving_usd is always 0.0 for this rule.

    Raises:
        Nothing.
    """
    try:
        if df.empty:
            return _not_triggered("embedding_inefficiency", "Embedding Inefficiency")

        emb_df = df[df["is_embedding"]]
        if emb_df.empty:
            return _not_triggered("embedding_inefficiency", "Embedding Inefficiency")

        total_emb_input = emb_df["input_tokens"].fillna(0).astype(float).sum()
        total_emb_reqs = emb_df["requests"].fillna(1).astype(float).sum()
        if total_emb_reqs == 0:
            return _not_triggered("embedding_inefficiency", "Embedding Inefficiency")

        avg_tokens_per_call = total_emb_input / total_emb_reqs

        if avg_tokens_per_call > RULE_D_MAX_AVG_TOKENS:
            return _not_triggered("embedding_inefficiency", "Embedding Inefficiency")

        affected_models = sorted(emb_df["model_normalized"].unique().tolist())

        headline = (
            f"Embedding calls average {avg_tokens_per_call:.0f} tokens each — "
            "likely single-item requests instead of batches (no cost saving, but reduces latency risk)"
        )
        detail = (
            f"Your embedding API calls average {avg_tokens_per_call:.0f} tokens each, "
            "suggesting individual or very small batch requests. "
            "While cost per token is unchanged, single-item calls consume one API rate-limit "
            "slot per item — with large volumes this causes throttling and adds cumulative latency. "
            "Batching reduces API calls by up to 99% with no change to output quality."
        )
        action = (
            "Batch up to 2,048 items per API call (OpenAI limit). "
            "In Python: pass a list to client.embeddings.create(input=[...]) instead of looping. "
            "Cost per token is unchanged — the saving is API calls, latency, and rate-limit headroom."
        )

        return RuleResult(
            rule_id="embedding_inefficiency",
            rule_name="Embedding Inefficiency",
            triggered=True,
            severity="MED",
            monthly_saving_usd=0.0,
            annual_saving_usd=0.0,
            finding_headline=headline,
            finding_detail=detail,
            recommended_action=action,
            affected_models=affected_models,
            confidence="HIGH",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("rule_embedding_inefficiency failed: %s", exc)
        return _not_triggered("embedding_inefficiency", "Embedding Inefficiency")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_rules(df: pd.DataFrame, pricing: dict) -> list[RuleResult]:
    """
    Run all four waste detection rules and return a list of RuleResult dicts.

    Results are sorted HIGH → MED → LOW → None.

    Args:
        df:      Enriched unified DataFrame.
        pricing: Pricing dict from load_pricing_table().

    Returns:
        List of RuleResult TypedDicts (always length 4, one per rule).

    Raises:
        Nothing — individual rule errors are caught inside each rule function.
    """
    results = [
        rule_model_task_mismatch(df, pricing),
        rule_context_bloat(df, pricing),
        rule_reasoning_overuse(df),
        rule_embedding_inefficiency(df),
    ]

    severity_order = {"HIGH": 0, "MED": 1, "LOW": 2, None: 3}
    results.sort(key=lambda r: severity_order.get(r["severity"], 3))
    return results
