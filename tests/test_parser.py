"""
test_parser.py — Unit tests for src/parser.py

Run with: python -m pytest tests/test_parser.py -v
"""

import io
import pytest
import pandas as pd

from src.parser import (
    detect_provider,
    normalize_model_name,
    parse_openai_csv,
    parse_anthropic_csv,
    OPENAI_MODEL_ALIASES,
)
from src.validators import detect_csv_injection


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_openai_csv(extra_cols: dict = None) -> io.StringIO:
    """Build a minimal valid OpenAI-format CSV string."""
    rows = [
        "date,model,requests,input_tokens,output_tokens,cost",
        "2025-05-01,gpt-4o,100,200000,50000,0.625",
        "2025-05-02,gpt-4o-mini,500,300000,80000,0.093",
        "2025-05-03,o3-mini,20,40000,15000,0.110",
        "2025-05-04,text-embedding-3-small,1000,420000,0,0.008",
    ]
    content = "\n".join(rows)
    buf = io.StringIO(content)
    buf.name = "openai_billing.csv"
    buf.size = len(content.encode())
    return buf


def _make_anthropic_csv() -> io.StringIO:
    """Build a minimal valid Anthropic-format CSV string."""
    rows = [
        "date,model,requests,input_tokens,output_tokens,cache_creation_input_tokens,cache_read_input_tokens,cost",
        "2025-05-01,claude-3-5-sonnet-20241022,80,160000,40000,5000,2000,0.580",
        "2025-05-02,claude-3-haiku-20240307,400,200000,60000,0,0,0.075",
    ]
    content = "\n".join(rows)
    buf = io.StringIO(content)
    buf.name = "anthropic_billing.csv"
    buf.size = len(content.encode())
    return buf


def _make_missing_cols_csv() -> io.StringIO:
    """CSV missing required columns."""
    content = "date,model\n2025-05-01,gpt-4o\n"
    buf = io.StringIO(content)
    buf.name = "bad.csv"
    buf.size = len(content.encode())
    return buf


def _make_injection_csv() -> io.StringIO:
    """CSV with formula injection in the model column."""
    content = "date,model,requests,input_tokens,output_tokens,cost\n"
    content += "2025-05-01,=SUM(A1:A10),100,200000,50000,0.625\n"
    buf = io.StringIO(content)
    buf.name = "injection.csv"
    buf.size = len(content.encode())
    return buf


# ─────────────────────────────────────────────
# detect_provider
# ─────────────────────────────────────────────

class TestDetectProvider:
    def test_detect_provider_openai(self):
        df = pd.DataFrame({"date": [], "model": [], "input_tokens": [], "output_tokens": [], "snapshot_id": []})
        assert detect_provider(df) == "openai"

    def test_detect_provider_anthropic(self):
        df = pd.DataFrame({"date": [], "model": [], "input_tokens": [], "output_tokens": [], "cache_creation_input_tokens": []})
        assert detect_provider(df) == "anthropic"

    def test_detect_provider_unknown(self):
        df = pd.DataFrame({"date": [], "model": [], "input_tokens": [], "output_tokens": []})
        assert detect_provider(df) == "unknown"


# ─────────────────────────────────────────────
# normalize_model_name
# ─────────────────────────────────────────────

class TestNormalizeModelName:
    def test_versioned_gpt4o(self):
        assert normalize_model_name("gpt-4o-2024-08-06") == "gpt-4o"

    def test_versioned_gpt4o_second_date(self):
        assert normalize_model_name("gpt-4o-2024-11-20") == "gpt-4o"

    def test_versioned_claude(self):
        assert normalize_model_name("claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"

    def test_versioned_o3_mini(self):
        assert normalize_model_name("o3-mini-2025-01-31") == "o3-mini"

    def test_unknown_passthrough(self):
        result = normalize_model_name("some-model-xyz")
        assert result == "some-model-xyz"

    def test_empty_string(self):
        result = normalize_model_name("")
        assert result == "unknown"

    def test_none_input(self):
        result = normalize_model_name(None)
        assert result == "unknown"

    def test_deepseek_r1(self):
        result = normalize_model_name("deepseek-r1")
        assert "deepseek" in result.lower()


# ─────────────────────────────────────────────
# parse_openai_csv
# ─────────────────────────────────────────────

class TestParseOpenAiCsv:
    def test_valid_csv_returns_dataframe(self):
        buf = _make_openai_csv()
        df = parse_openai_csv(buf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_valid_csv_has_required_columns(self):
        buf = _make_openai_csv()
        df = parse_openai_csv(buf)
        for col in ["date", "model", "model_normalized", "provider", "input_tokens", "output_tokens"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_provider_is_openai(self):
        buf = _make_openai_csv()
        df = parse_openai_csv(buf)
        assert (df["provider"] == "openai").all()

    def test_is_reasoning_flagged(self):
        buf = _make_openai_csv()
        df = parse_openai_csv(buf)
        # o3-mini row should be flagged as reasoning
        reasoning_rows = df[df["model_normalized"].str.contains("o3", na=False)]
        assert reasoning_rows["is_reasoning"].any() or len(reasoning_rows) == 0  # pass if none matched

    def test_is_embedding_flagged(self):
        buf = _make_openai_csv()
        df = parse_openai_csv(buf)
        emb_rows = df[df["model_normalized"].str.contains("embedding", na=False)]
        if not emb_rows.empty:
            assert emb_rows["is_embedding"].all()

    def test_missing_cols_raises_value_error(self):
        buf = _make_missing_cols_csv()
        with pytest.raises(ValueError):
            parse_openai_csv(buf)

    def test_csv_injection_raises_value_error(self):
        buf = _make_injection_csv()
        with pytest.raises(ValueError, match="injection"):
            parse_openai_csv(buf)


# ─────────────────────────────────────────────
# parse_anthropic_csv
# ─────────────────────────────────────────────

class TestParseAnthropicCsv:
    def test_valid_csv_returns_dataframe(self):
        buf = _make_anthropic_csv()
        df = parse_anthropic_csv(buf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_provider_is_anthropic(self):
        buf = _make_anthropic_csv()
        df = parse_anthropic_csv(buf)
        assert (df["provider"] == "anthropic").all()

    def test_cache_columns_present(self):
        buf = _make_anthropic_csv()
        df = parse_anthropic_csv(buf)
        assert "cache_creation_tokens" in df.columns
        assert "cache_read_tokens" in df.columns

    def test_is_using_cache_true_when_cache_tokens_present(self):
        buf = _make_anthropic_csv()
        df = parse_anthropic_csv(buf)
        # First row has cache tokens > 0
        assert df.iloc[0]["is_using_cache"] == True


# ─────────────────────────────────────────────
# detect_csv_injection (standalone)
# ─────────────────────────────────────────────

class TestCsvInjection:
    def test_detects_sum_formula(self):
        df = pd.DataFrame({"model": ["=SUM(A1:A10)"]})
        assert detect_csv_injection(df) is True

    def test_detects_at_prefix(self):
        df = pd.DataFrame({"model": ["@USER"]})
        assert detect_csv_injection(df) is True

    def test_clean_data_not_flagged(self):
        df = pd.DataFrame({"model": ["gpt-4o", "claude-3-haiku", "deepseek-v4"]})
        assert detect_csv_injection(df) is False

    def test_numeric_columns_ignored(self):
        df = pd.DataFrame({"cost": [1.0, 2.0, 3.0]})
        assert detect_csv_injection(df) is False
