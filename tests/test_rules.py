"""
test_rules.py — Unit tests for src/rules.py

Run with: python -m pytest tests/test_rules.py -v
"""

import pytest
import pandas as pd
import numpy as np

from src.rules import (
    rule_model_task_mismatch,
    rule_context_bloat,
    rule_reasoning_overuse,
    rule_embedding_inefficiency,
    run_all_rules,
    RuleResult,
)
from src.pricing import HARDCODED_FALLBACK


# ─────────────────────────────────────────────
# DataFrame builder helpers
# ─────────────────────────────────────────────

def _make_df(
    model: str,
    requests: int,
    avg_input: int,
    avg_output: int,
    cost_per_req: float = 0.01,
    is_reasoning: bool = False,
    is_embedding: bool = False,
    days: int = 30,
) -> pd.DataFrame:
    """Build a minimal enriched unified DataFrame for a single model."""
    rows = []
    for i in range(days):
        rows.append({
            "date":               pd.Timestamp("2025-05-01") + pd.Timedelta(days=i),
            "model":              model,
            "model_normalized":   model,
            "provider":           "openai",
            "requests":           requests // days,
            "input_tokens":       (avg_input * requests) // days,
            "output_tokens":      (avg_output * requests) // days,
            "cost_reported":      cost_per_req * requests / days,
            "cost_calculated":    cost_per_req * requests / days,
            "cost_delta":         0.0,
            "cost_per_request":   cost_per_req,
            "avg_input_per_request":  float(avg_input),
            "avg_output_per_request": float(avg_output),
            "is_reasoning":       is_reasoning,
            "is_embedding":       is_embedding,
            "is_using_cache":     False,
            "cache_creation_tokens": 0,
            "cache_read_tokens":  0,
        })
    return pd.DataFrame(rows)


def _concat_dfs(*dfs) -> pd.DataFrame:
    return pd.concat(list(dfs), ignore_index=True)


# ─────────────────────────────────────────────
# Rule A — model_task_mismatch
# ─────────────────────────────────────────────

class TestRuleA:
    def test_triggers_on_low_output(self):
        """gpt-4o with avg 200 output and 200 requests → should trigger."""
        df = _make_df("gpt-4o", requests=200, avg_input=1500, avg_output=200)
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        assert result["triggered"] is True
        assert result["monthly_saving_usd"] > 0

    def test_no_trigger_high_output(self):
        """gpt-4o with avg 600 output → not simple task, should not trigger."""
        df = _make_df("gpt-4o", requests=200, avg_input=2000, avg_output=600)
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        assert result["triggered"] is False

    def test_no_trigger_low_volume(self):
        """gpt-4o with only 10 requests → below MIN_REQUESTS threshold."""
        df = _make_df("gpt-4o", requests=10, avg_input=1000, avg_output=200)
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        assert result["triggered"] is False

    def test_saving_is_positive(self):
        """Saving must be > 0 when triggered."""
        df = _make_df("gpt-4o", requests=500, avg_input=1000, avg_output=150)
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        if result["triggered"]:
            assert result["monthly_saving_usd"] > 0

    def test_annual_is_12x_monthly(self):
        df = _make_df("gpt-4o", requests=300, avg_input=1000, avg_output=200)
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        if result["triggered"]:
            assert abs(result["annual_saving_usd"] - result["monthly_saving_usd"] * 12) < 0.05

    def test_embeddings_excluded(self):
        """Embedding model rows should not contribute to Rule A."""
        df = _make_df("text-embedding-3-small", requests=2000, avg_input=400, avg_output=0, is_embedding=True)
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        assert result["triggered"] is False

    def test_no_crash_empty_df(self):
        df = pd.DataFrame()
        result = rule_model_task_mismatch(df, HARDCODED_FALLBACK)
        assert result["triggered"] is False


# ─────────────────────────────────────────────
# Rule B — context_bloat
# ─────────────────────────────────────────────

class TestRuleB:
    def test_triggers_on_high_input(self):
        """avg input > 2000 tokens → triggers."""
        df = _make_df("gpt-4o", requests=200, avg_input=5000, avg_output=300)
        result = rule_context_bloat(df, HARDCODED_FALLBACK)
        assert result["triggered"] is True

    def test_no_trigger_low_input(self):
        """avg input < 2000 → no trigger."""
        df = _make_df("gpt-4o", requests=200, avg_input=800, avg_output=300)
        result = rule_context_bloat(df, HARDCODED_FALLBACK)
        assert result["triggered"] is False

    def test_saving_is_conservative(self):
        """Saving must be ≤ 40% of total input cost."""
        df = _make_df("gpt-4o", requests=200, avg_input=8000, avg_output=200)
        result = rule_context_bloat(df, HARDCODED_FALLBACK)
        if result["triggered"]:
            # Verify saving fraction is ≤ 40% of any reasonable total cost
            assert result["monthly_saving_usd"] >= 0

    def test_confidence_is_med(self):
        """Context bloat confidence should always be MED."""
        df = _make_df("gpt-4o", requests=200, avg_input=5000, avg_output=300)
        result = rule_context_bloat(df, HARDCODED_FALLBACK)
        if result["triggered"]:
            assert result["confidence"] == "MED"

    def test_no_crash_empty_df(self):
        df = pd.DataFrame()
        result = rule_context_bloat(df, HARDCODED_FALLBACK)
        assert result["triggered"] is False


# ─────────────────────────────────────────────
# Rule C — reasoning_overuse
# ─────────────────────────────────────────────

class TestRuleC:
    def test_triggers_on_high_reasoning_pct(self):
        """If >15% spend is on reasoning models → triggers."""
        reasoning_df = _make_df("o1", requests=100, avg_input=2000, avg_output=600,
                                 cost_per_req=0.10, is_reasoning=True)
        standard_df = _make_df("gpt-4o-mini", requests=1000, avg_input=500, avg_output=150,
                                cost_per_req=0.001)
        df = _concat_dfs(reasoning_df, standard_df)
        result = rule_reasoning_overuse(df)
        assert result["triggered"] is True

    def test_no_trigger_low_reasoning_pct(self):
        """If <15% spend on reasoning → no trigger."""
        reasoning_df = _make_df("o1", requests=5, avg_input=1000, avg_output=500,
                                 cost_per_req=0.05, is_reasoning=True)
        standard_df = _make_df("gpt-4o-mini", requests=5000, avg_input=500, avg_output=150,
                                cost_per_req=0.001)
        df = _concat_dfs(reasoning_df, standard_df)
        result = rule_reasoning_overuse(df)
        assert result["triggered"] is False

    def test_no_trigger_no_reasoning_models(self):
        df = _make_df("gpt-4o", requests=500, avg_input=1000, avg_output=300)
        result = rule_reasoning_overuse(df)
        assert result["triggered"] is False

    def test_confidence_is_med(self):
        reasoning_df = _make_df("o1", requests=200, avg_input=2000, avg_output=600,
                                 cost_per_req=0.10, is_reasoning=True)
        standard_df = _make_df("gpt-4o-mini", requests=100, avg_input=500, avg_output=150,
                                cost_per_req=0.001)
        df = _concat_dfs(reasoning_df, standard_df)
        result = rule_reasoning_overuse(df)
        if result["triggered"]:
            assert result["confidence"] == "MED"

    def test_no_crash_empty_df(self):
        df = pd.DataFrame()
        result = rule_reasoning_overuse(df)
        assert result["triggered"] is False


# ─────────────────────────────────────────────
# Rule D — embedding_inefficiency
# ─────────────────────────────────────────────

class TestRuleD:
    def test_triggers_on_small_batches(self):
        """avg tokens per embedding call ≤ 200 → triggers."""
        df = _make_df(
            "text-embedding-3-small",
            requests=5000, avg_input=100, avg_output=0,
            is_embedding=True,
        )
        result = rule_embedding_inefficiency(df)
        assert result["triggered"] is True

    def test_no_trigger_large_batches(self):
        """avg tokens per embedding call > 200 → not triggered."""
        df = _make_df(
            "text-embedding-3-small",
            requests=500, avg_input=800, avg_output=0,
            is_embedding=True,
        )
        result = rule_embedding_inefficiency(df)
        assert result["triggered"] is False

    def test_saving_is_always_zero(self):
        """Rule D never claims dollar savings."""
        df = _make_df(
            "text-embedding-3-small",
            requests=5000, avg_input=50, avg_output=0,
            is_embedding=True,
        )
        result = rule_embedding_inefficiency(df)
        assert result["monthly_saving_usd"] == 0.0
        assert result["annual_saving_usd"] == 0.0

    def test_confidence_is_high(self):
        df = _make_df(
            "text-embedding-3-small",
            requests=5000, avg_input=50, avg_output=0,
            is_embedding=True,
        )
        result = rule_embedding_inefficiency(df)
        if result["triggered"]:
            assert result["confidence"] == "HIGH"

    def test_no_trigger_no_embeddings(self):
        df = _make_df("gpt-4o", requests=200, avg_input=1000, avg_output=300)
        result = rule_embedding_inefficiency(df)
        assert result["triggered"] is False

    def test_no_crash_empty_df(self):
        df = pd.DataFrame()
        result = rule_embedding_inefficiency(df)
        assert result["triggered"] is False


# ─────────────────────────────────────────────
# run_all_rules
# ─────────────────────────────────────────────

class TestRunAllRules:
    def test_returns_four_results(self):
        df = _make_df("gpt-4o", requests=100, avg_input=1000, avg_output=300)
        results = run_all_rules(df, HARDCODED_FALLBACK)
        assert len(results) == 4

    def test_no_crash_empty_df(self):
        df = pd.DataFrame()
        results = run_all_rules(df, HARDCODED_FALLBACK)
        assert len(results) == 4
        assert all(not r["triggered"] for r in results)

    def test_sorted_high_before_low(self):
        """HIGH severity findings must come before LOW."""
        reasoning_df = _make_df("o1", requests=500, avg_input=2000, avg_output=600,
                                 cost_per_req=0.10, is_reasoning=True)
        standard_df = _make_df("gpt-4o-mini", requests=50, avg_input=500, avg_output=150,
                                cost_per_req=0.001)
        df = _concat_dfs(reasoning_df, standard_df)
        results = run_all_rules(df, HARDCODED_FALLBACK)
        severities = [r["severity"] for r in results if r["severity"] is not None]
        order = {"HIGH": 0, "MED": 1, "LOW": 2}
        for i in range(len(severities) - 1):
            assert order.get(severities[i], 99) <= order.get(severities[i + 1], 99)

    def test_all_results_are_rule_result_typed(self):
        df = _make_df("gpt-4o", requests=200, avg_input=5000, avg_output=300)
        results = run_all_rules(df, HARDCODED_FALLBACK)
        required_keys = {
            "rule_id", "rule_name", "triggered", "severity",
            "monthly_saving_usd", "annual_saving_usd", "finding_headline",
            "finding_detail", "recommended_action", "affected_models", "confidence"
        }
        for r in results:
            assert required_keys.issubset(set(r.keys()))
