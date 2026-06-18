"""
pricing.py — LiteLLM pricing fetch, cache, and enrichment for AI Spend Auditor.

Attempts to load pricing from the LiteLLM GitHub repo (live), falls back to
the local JSON file, then falls back to a hardcoded minimal table.

All monetary values are stored as full-precision float64 internally.
"""

import logging
import os
from typing import Optional

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LITELLM_PRICING_URL: str = (
    "https://raw.githubusercontent.com/BerriAI/litellm"
    "/main/model_prices_and_context_window.json"
)
FALLBACK_PRICING_PATH: str = "data/litellm_pricing.json"
FETCH_TIMEOUT_SECONDS: int = 10

# ---------------------------------------------------------------------------
# Hardcoded fallback pricing — updated June 2026 with SOTA models
# All prices are per TOKEN (not per 1M tokens)
# ---------------------------------------------------------------------------
HARDCODED_FALLBACK: dict[str, dict[str, float]] = {
    # ──────────────── OpenAI ────────────────
    # GPT-4o family
    "gpt-4o":                         {"input_cost_per_token": 0.0000025,   "output_cost_per_token": 0.00001},
    "gpt-4o-mini":                    {"input_cost_per_token": 0.00000015,  "output_cost_per_token": 0.0000006},
    # GPT-4 legacy
    "gpt-4-turbo":                    {"input_cost_per_token": 0.00001,     "output_cost_per_token": 0.00003},
    "gpt-4":                          {"input_cost_per_token": 0.00003,     "output_cost_per_token": 0.00006},
    "gpt-3.5-turbo":                  {"input_cost_per_token": 0.0000005,   "output_cost_per_token": 0.0000015},
    # GPT-5 family (2025-2026)
    "gpt-5":                          {"input_cost_per_token": 0.0000025,   "output_cost_per_token": 0.000015},
    "gpt-5-pro":                      {"input_cost_per_token": 0.000005,    "output_cost_per_token": 0.00003},
    "gpt-5-mini":                     {"input_cost_per_token": 0.00000075,  "output_cost_per_token": 0.0000045},
    # o-series reasoning (higher cost — internal thinking tokens)
    "o1":                             {"input_cost_per_token": 0.000015,    "output_cost_per_token": 0.00006},
    "o1-mini":                        {"input_cost_per_token": 0.000001,    "output_cost_per_token": 0.000004},
    "o1-preview":                     {"input_cost_per_token": 0.000015,    "output_cost_per_token": 0.00006},
    "o3":                             {"input_cost_per_token": 0.00001,     "output_cost_per_token": 0.00004},
    "o3-mini":                        {"input_cost_per_token": 0.0000011,   "output_cost_per_token": 0.0000044},
    "o4":                             {"input_cost_per_token": 0.00001,     "output_cost_per_token": 0.00004},
    "o4-mini":                        {"input_cost_per_token": 0.0000011,   "output_cost_per_token": 0.0000044},
    # Embeddings
    "text-embedding-3-small":         {"input_cost_per_token": 0.00000002,  "output_cost_per_token": 0.0},
    "text-embedding-3-large":         {"input_cost_per_token": 0.00000013,  "output_cost_per_token": 0.0},
    "text-embedding-ada-002":         {"input_cost_per_token": 0.0000001,   "output_cost_per_token": 0.0},

    # ──────────────── Anthropic ────────────────
    # Claude 3 series
    "claude-3-opus":                  {"input_cost_per_token": 0.000015,    "output_cost_per_token": 0.000075},
    "claude-3-sonnet":                {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    "claude-3-haiku":                 {"input_cost_per_token": 0.00000025,  "output_cost_per_token": 0.00000125},
    # Versioned equivalents (alias lookup fallback)
    "claude-3-opus-20240229":         {"input_cost_per_token": 0.000015,    "output_cost_per_token": 0.000075},
    "claude-3-haiku-20240307":        {"input_cost_per_token": 0.00000025,  "output_cost_per_token": 0.00000125},
    # Claude 3.5 series
    "claude-3-5-sonnet":              {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    "claude-3-5-sonnet-20241022":     {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    "claude-3-5-haiku":               {"input_cost_per_token": 0.0000008,   "output_cost_per_token": 0.000004},
    "claude-3-5-haiku-20241022":      {"input_cost_per_token": 0.0000008,   "output_cost_per_token": 0.000004},
    # Claude 3.7 (extended thinking)
    "claude-3-7-sonnet":              {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    "claude-3-7-sonnet-20250219":     {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    # Claude 4 series (2025)
    "claude-opus-4":                  {"input_cost_per_token": 0.000015,    "output_cost_per_token": 0.000075},
    "claude-sonnet-4":                {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    "claude-haiku-4":                 {"input_cost_per_token": 0.0000008,   "output_cost_per_token": 0.000004},
    # Claude 4.5 / 4.6 (2025-2026)
    "claude-opus-4-5":                {"input_cost_per_token": 0.000005,    "output_cost_per_token": 0.000025},
    "claude-sonnet-4-6":              {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    # Claude Fable 5 (frontier, mid-2026)
    "claude-fable-5":                 {"input_cost_per_token": 0.00001,     "output_cost_per_token": 0.00005},

    # ──────────────── Google Gemini ────────────────
    "gemini-2.5-pro":                 {"input_cost_per_token": 0.00000125,  "output_cost_per_token": 0.00001},
    "gemini-2.5-flash":               {"input_cost_per_token": 0.0000003,   "output_cost_per_token": 0.0000025},
    "gemini-1.5-pro":                 {"input_cost_per_token": 0.00000125,  "output_cost_per_token": 0.000005},
    "gemini-1.5-flash":               {"input_cost_per_token": 0.000000075, "output_cost_per_token": 0.0000003},
    # Gemini 3.x (2026)
    "gemini-3.1-pro":                 {"input_cost_per_token": 0.000002,    "output_cost_per_token": 0.000012},
    "gemini-3.5-flash":               {"input_cost_per_token": 0.0000015,   "output_cost_per_token": 0.000009},

    # ──────────────── DeepSeek ────────────────
    # DeepSeek V3 family (chat / coding)
    "deepseek-chat":                  {"input_cost_per_token": 0.00000028,  "output_cost_per_token": 0.00000042},
    "deepseek-v3":                    {"input_cost_per_token": 0.00000027,  "output_cost_per_token": 0.0000011},
    "deepseek-coder-v2":              {"input_cost_per_token": 0.00000014,  "output_cost_per_token": 0.00000028},
    # DeepSeek R1 (reasoning)
    "deepseek-r1":                    {"input_cost_per_token": 0.00000055,  "output_cost_per_token": 0.00000219},
    "deepseek-r1-0528":               {"input_cost_per_token": 0.00000055,  "output_cost_per_token": 0.00000219},
    "deepseek-r1-lite-preview":       {"input_cost_per_token": 0.00000028,  "output_cost_per_token": 0.00000110},
    # DeepSeek V4 (2026)
    "deepseek-v4":                    {"input_cost_per_token": 0.00000030,  "output_cost_per_token": 0.00000050},
    "deepseek-v4-pro":                {"input_cost_per_token": 0.000000435, "output_cost_per_token": 0.00000087},
    "deepseek-v4-flash":              {"input_cost_per_token": 0.00000014,  "output_cost_per_token": 0.00000028},

    # ──────────────── Moonshot / Kimi ────────────────
    "kimi-k2":                        {"input_cost_per_token": 0.00000055,  "output_cost_per_token": 0.0000025},
    "kimi-k2-6":                      {"input_cost_per_token": 0.00000065,  "output_cost_per_token": 0.0000030},
    "kimi-k2-7":                      {"input_cost_per_token": 0.00000095,  "output_cost_per_token": 0.000004},
    "moonshot-v1-8k":                 {"input_cost_per_token": 0.00000100,  "output_cost_per_token": 0.00000300},
    "moonshot-v1-32k":                {"input_cost_per_token": 0.00000200,  "output_cost_per_token": 0.00000600},

    # ──────────────── Mistral ────────────────
    "mistral-large":                  {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000009},
    "mistral-medium":                 {"input_cost_per_token": 0.0000027,   "output_cost_per_token": 0.0000081},
    "mistral-small":                  {"input_cost_per_token": 0.000001,    "output_cost_per_token": 0.000003},
    "mistral-7b":                     {"input_cost_per_token": 0.00000025,  "output_cost_per_token": 0.00000025},
    "mixtral-8x7b":                   {"input_cost_per_token": 0.0000007,   "output_cost_per_token": 0.0000007},
    "mixtral-8x22b":                  {"input_cost_per_token": 0.000002,    "output_cost_per_token": 0.000006},
    "mistral-embed":                  {"input_cost_per_token": 0.0000001,   "output_cost_per_token": 0.0},

    # ──────────────── Meta / Open-source ────────────────
    "llama-3.1-405b":                 {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000003},
    "llama-3.1-70b":                  {"input_cost_per_token": 0.00000088,  "output_cost_per_token": 0.00000088},
    "llama-3.1-8b":                   {"input_cost_per_token": 0.00000018,  "output_cost_per_token": 0.00000018},
    "llama-3.3-70b":                  {"input_cost_per_token": 0.00000088,  "output_cost_per_token": 0.00000088},

    # ──────────────── Cohere ────────────────
    "command-r-plus":                 {"input_cost_per_token": 0.000003,    "output_cost_per_token": 0.000015},
    "command-r":                      {"input_cost_per_token": 0.0000005,   "output_cost_per_token": 0.0000015},
    "embed-english-v3.0":             {"input_cost_per_token": 0.0000001,   "output_cost_per_token": 0.0},
    "embed-multilingual-v3.0":        {"input_cost_per_token": 0.0000001,   "output_cost_per_token": 0.0},

    # ──────────────── Qwen (Alibaba) ────────────────
    "qwen-turbo":                     {"input_cost_per_token": 0.00000050,  "output_cost_per_token": 0.00000150},
    "qwen-plus":                      {"input_cost_per_token": 0.0000008,   "output_cost_per_token": 0.0000024},
    "qwen-max":                       {"input_cost_per_token": 0.0000024,   "output_cost_per_token": 0.0000096},
    "qwq-32b":                        {"input_cost_per_token": 0.0000012,   "output_cost_per_token": 0.0000048},
    "qwq-plus":                       {"input_cost_per_token": 0.0000016,   "output_cost_per_token": 0.0000064},
}

# ---------------------------------------------------------------------------
# Model downgrade map — updated June 2026 with SOTA models
# Maps expensive_model → recommended cheaper alternative
# Selected based on approximate capability parity, not raw benchmark scores
# ---------------------------------------------------------------------------
MODEL_DOWNGRADE_MAP: dict[str, str] = {
    # OpenAI — expensive → cheaper with similar capability for most tasks
    "gpt-4o":                         "gpt-4o-mini",
    "gpt-4-turbo":                    "gpt-4o-mini",
    "gpt-4":                          "gpt-4o-mini",
    "gpt-5-pro":                      "gpt-5",
    "gpt-5":                          "gpt-5-mini",
    # o-series reasoning → standard model for non-reasoning tasks
    "o1":                             "gpt-4o",
    "o1-preview":                     "gpt-4o",
    "o3":                             "o4-mini",
    "o4":                             "o4-mini",
    "o4-mini":                        "gpt-4o-mini",
    # Anthropic — expensive → cheaper (similar capability tier)
    "claude-opus-4":                  "claude-sonnet-4",
    "claude-opus-4-5":                "claude-sonnet-4-6",
    "claude-fable-5":                 "claude-opus-4-5",
    "claude-3-opus":                  "claude-3-5-sonnet",
    "claude-3-opus-20240229":         "claude-3-5-sonnet",
    "claude-3-5-sonnet":              "claude-3-5-haiku",
    "claude-3-5-sonnet-20241022":     "claude-3-haiku",
    "claude-sonnet-4":                "claude-haiku-4",
    "claude-sonnet-4-6":              "claude-haiku-4",
    "claude-3-7-sonnet":              "claude-3-5-sonnet",
    # DeepSeek — reasoning → chat for non-reasoning tasks
    "deepseek-r1":                    "deepseek-v4",
    "deepseek-r1-0528":               "deepseek-v4",
    "deepseek-v4-pro":                "deepseek-v4",
    "deepseek-v4":                    "deepseek-v4-flash",
    # Kimi — tiered downgrade
    "kimi-k2-7":                      "kimi-k2-6",
    "kimi-k2-6":                      "kimi-k2",
    "kimi-k2":                        "deepseek-v4-flash",
    # Gemini — pro → flash
    "gemini-3.1-pro":                 "gemini-3.5-flash",
    "gemini-2.5-pro":                 "gemini-2.5-flash",
    "gemini-1.5-pro":                 "gemini-1.5-flash",
    # Mistral
    "mistral-large":                  "mistral-small",
    "mixtral-8x22b":                  "mixtral-8x7b",
    "command-r-plus":                 "command-r",
}


# ---------------------------------------------------------------------------
# Pricing loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def load_pricing_table() -> tuple[dict, str]:
    """
    Load the LiteLLM model pricing table from one of three sources.

    Attempt order:
      1. Live fetch from LITELLM_PRICING_URL (10-second timeout)
      2. Local file at FALLBACK_PRICING_PATH (bundled in repo)
      3. HARDCODED_FALLBACK dict defined in this module

    Returns:
        (pricing_dict, source) where source is one of:
        "live" | "cached_file" | "fallback_hardcoded"

    Raises:
        Nothing — always returns a valid (dict, str) tuple.
    """
    # 1. Live fetch
    try:
        resp = requests.get(LITELLM_PRICING_URL, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Pricing loaded from live LiteLLM URL (%d models).", len(data))
        return data, "live"
    except requests.RequestException as exc:
        logger.warning("Live pricing fetch failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live pricing parse error: %s", exc)

    # 2. Local fallback file
    try:
        import json

        path = FALLBACK_PRICING_PATH
        if not os.path.isabs(path):
            # Resolve relative to project root (where app.py lives)
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, FALLBACK_PRICING_PATH)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Pricing loaded from local file (%d models).", len(data))
        return data, "cached_file"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local pricing file load failed: %s", exc)

    # 3. Hardcoded fallback
    logger.warning("Using hardcoded pricing fallback (%d models).", len(HARDCODED_FALLBACK))
    return HARDCODED_FALLBACK, "fallback_hardcoded"


# ---------------------------------------------------------------------------
# Cost calculation helpers
# ---------------------------------------------------------------------------

def _lookup_model_pricing(model_normalized: str, pricing: dict) -> Optional[dict]:
    """
    Look up a model's pricing entry with progressive fallback strategies.

    Strategies tried in order:
      1. Exact key match
      2. Strip version suffix (split on "-20", take everything before)
      3. HARDCODED_FALLBACK direct lookup (ensures common models always work)

    Args:
        model_normalized: Canonical model name from parser.normalize_model_name().
        pricing:          Pricing dict (e.g. from load_pricing_table()).

    Returns:
        Dict with "input_cost_per_token" / "output_cost_per_token", or None.

    Raises:
        Nothing.
    """
    if not model_normalized:
        return None

    # 1. Exact match
    if model_normalized in pricing:
        return pricing[model_normalized]

    # 2. Strip versioned suffix (e.g. "claude-3-5-sonnet-20241022" → "claude-3-5-sonnet")
    parts = model_normalized.rsplit("-20", 1)
    if len(parts) == 2:
        stripped = parts[0]
        if stripped in pricing:
            return pricing[stripped]

    # 3. Hardcoded fallback (always reliable for known models)
    if model_normalized in HARDCODED_FALLBACK:
        return HARDCODED_FALLBACK[model_normalized]

    logger.debug("No pricing found for model '%s'.", model_normalized)
    return None


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    model_normalized: str,
    pricing: dict,
) -> float:
    """
    Compute the cost of a single API call given token counts.

    Formula:
        cost = input_tokens * input_cost_per_token
               + output_tokens * output_cost_per_token

    Args:
        input_tokens:     Number of input tokens.
        output_tokens:    Number of output tokens.
        model_normalized: Canonical model name for pricing lookup.
        pricing:          Pricing dict from load_pricing_table().

    Returns:
        Cost in USD as a float.  Returns 0.0 if model not found in pricing.

    Raises:
        Nothing.
    """
    try:
        entry = _lookup_model_pricing(model_normalized, pricing)
        if entry is None:
            return 0.0

        in_cost = float(entry.get("input_cost_per_token", 0.0))
        out_cost = float(entry.get("output_cost_per_token", 0.0))
        return (input_tokens * in_cost) + (output_tokens * out_cost)
    except Exception as exc:  # noqa: BLE001
        logger.warning("calculate_token_cost failed for '%s': %s", model_normalized, exc)
        return 0.0


def enrich_dataframe(df: pd.DataFrame, pricing: dict) -> pd.DataFrame:
    """
    Add cost and usage efficiency columns to the unified DataFrame.

    New columns added:
      - cost_calculated (float64):  computed from token counts + pricing table
      - cost_delta      (float64):  cost_reported - cost_calculated
      - cost_per_request (float64): cost_calculated / requests (0 when requests=0)
      - avg_input_per_request (float64)
      - avg_output_per_request (float64)

    Args:
        df:      Unified schema DataFrame from parser.merge_providers().
        pricing: Pricing dict from load_pricing_table().

    Returns:
        Enriched DataFrame with all original columns plus the new ones.

    Raises:
        Nothing — any per-row error results in 0.0 for that row.
    """
    df = df.copy()

    def _row_cost(row) -> float:
        return calculate_token_cost(
            int(row["input_tokens"] or 0),
            int(row["output_tokens"] or 0),
            str(row["model_normalized"]),
            pricing,
        )

    df["cost_calculated"] = df.apply(_row_cost, axis=1)
    df["cost_delta"] = (
        df["cost_reported"].fillna(0.0) - df["cost_calculated"]
    )

    reqs = df["requests"].fillna(1).astype(float).replace(0, 1)
    df["cost_per_request"] = df["cost_calculated"] / reqs
    df["avg_input_per_request"] = df["input_tokens"].fillna(0).astype(float) / reqs
    df["avg_output_per_request"] = df["output_tokens"].fillna(0).astype(float) / reqs

    return df
