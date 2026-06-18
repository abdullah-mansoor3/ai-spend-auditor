"""
parser.py — CSV parsing and normalisation for AI Spend Auditor.

Produces a unified DataFrame schema regardless of whether the source is
OpenAI or Anthropic billing data.  All user data stays in memory — nothing
is written to disk here.

Unified schema after parse (see UNIFIED_SCHEMA dict at bottom for dtypes):
  date, model, model_normalized, provider, requests,
  input_tokens, output_tokens, cost_reported,
  is_reasoning, is_embedding, is_using_cache,
  cache_creation_tokens, cache_read_tokens
"""

import logging
import re
from io import BytesIO, StringIO
from typing import Optional

import pandas as pd

from src.validators import validate_uploaded_file, validate_dataframe, detect_csv_injection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required column sets
# ---------------------------------------------------------------------------
OPENAI_REQUIRED_COLS: set[str] = {"date", "model", "input_tokens", "output_tokens"}
ANTHROPIC_REQUIRED_COLS: set[str] = {"date", "model", "input_tokens", "output_tokens"}

# Accepted names for the cost column (tried in order)
COST_COL_VARIANTS: list[str] = ["cost", "cost_usd", "total_cost", "amount", "total_amount"]

# ---------------------------------------------------------------------------
# Model alias tables — updated June 2026
# ---------------------------------------------------------------------------
OPENAI_MODEL_ALIASES: dict[str, str] = {
    # GPT-4o family
    "gpt-4o-2024-05-13":                "gpt-4o",
    "gpt-4o-2024-08-06":                "gpt-4o",
    "gpt-4o-2024-11-20":                "gpt-4o",
    "gpt-4o-2025-03-26":                "gpt-4o",
    "chatgpt-4o-latest":                "gpt-4o",
    # GPT-4o mini
    "gpt-4o-mini-2024-07-18":           "gpt-4o-mini",
    "gpt-4o-mini-2025-04-01":           "gpt-4o-mini",
    # GPT-4 turbo / legacy
    "gpt-4-turbo-2024-04-09":           "gpt-4-turbo",
    "gpt-4-turbo-preview":              "gpt-4-turbo",
    "gpt-4-0125-preview":               "gpt-4-turbo",
    "gpt-4-1106-preview":               "gpt-4-turbo",
    "gpt-4-0613":                       "gpt-4",
    # GPT-5 family (2025-2026)
    "gpt-5.4":                          "gpt-5",
    "gpt-5.5":                          "gpt-5-pro",
    "gpt-5.4-mini":                     "gpt-5-mini",
    # o-series reasoning models
    "o1-2024-12-17":                    "o1",
    "o1-mini-2024-09-12":               "o1-mini",
    "o1-preview-2024-09-12":            "o1-preview",
    "o3-2025-04-16":                    "o3",
    "o3-mini-2025-01-31":               "o3-mini",
    "o4-mini-2025-04-16":               "o4-mini",
    "o4-2025-05-20":                    "o4",
    # Embeddings
    "text-embedding-3-small":           "text-embedding-3-small",
    "text-embedding-3-large":           "text-embedding-3-large",
    "text-embedding-ada-002":           "text-embedding-ada-002",
}

ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    # Claude 3 series
    "claude-3-opus-20240229":           "claude-3-opus",
    "claude-3-sonnet-20240229":         "claude-3-sonnet",
    "claude-3-haiku-20240307":          "claude-3-haiku",
    # Claude 3.5 series
    "claude-3-5-sonnet-20240620":       "claude-3-5-sonnet",
    "claude-3-5-sonnet-20241022":       "claude-3-5-sonnet",
    "claude-3-5-haiku-20241022":        "claude-3-5-haiku",
    # Claude 3.7 (extended thinking = reasoning)
    "claude-3-7-sonnet-20250219":       "claude-3-7-sonnet",
    "claude-3-7-sonnet-latest":         "claude-3-7-sonnet",
    # Claude 4 series (2025)
    "claude-opus-4-20250514":           "claude-opus-4",
    "claude-sonnet-4-20250514":         "claude-sonnet-4",
    "claude-haiku-4-20250514":          "claude-haiku-4",
    # Claude 4.5 / 4.6 (2025-2026)
    "claude-opus-4-5":                  "claude-opus-4-5",
    "claude-sonnet-4-6":                "claude-sonnet-4-6",
    "claude-sonnet-4-6-20260415":       "claude-sonnet-4-6",
    "claude-fable-5-20260501":          "claude-fable-5",
}

# All aliases combined (OpenAI first, then Anthropic — no overlap expected)
ALL_MODEL_ALIASES: dict[str, str] = {**OPENAI_MODEL_ALIASES, **ANTHROPIC_MODEL_ALIASES}

# ---------------------------------------------------------------------------
# Model classification sets — updated June 2026 with SOTA models
# ---------------------------------------------------------------------------
REASONING_MODELS: set[str] = {
    # OpenAI o-series
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4", "o4-mini",
    # OpenAI GPT-5 frontier (includes chain-of-thought reasoning)
    "gpt-5", "gpt-5-pro",
    # Anthropic extended thinking
    "claude-3-7-sonnet",
    "claude-opus-4", "claude-opus-4-5",
    "claude-fable-5",
    # DeepSeek reasoning
    "deepseek-r1", "deepseek-r1-0528", "deepseek-r1-lite-preview",
    "deepseek-r2",
    # Kimi K2 (reasoning-class)
    "kimi-k2", "kimi-k2-6", "kimi-k2-7",
    # Google Gemini reasoning variants
    "gemini-2-5-pro-thinking", "gemini-3-1-pro-thinking",
    # Qwen QwQ reasoning
    "qwq-32b", "qwq-plus",
}

EMBEDDING_MODELS: set[str] = {
    # OpenAI
    "text-embedding-3-small", "text-embedding-3-large",
    "text-embedding-ada-002",
    # Cohere
    "embed-english-v3.0", "embed-multilingual-v3.0",
    # Amazon
    "amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2",
    # Mistral
    "mistral-embed",
    # Voyage AI (popular with Anthropic users)
    "voyage-large-2", "voyage-code-2", "voyage-3", "voyage-3-lite",
    # Jina
    "jina-embeddings-v2-base-en", "jina-embeddings-v3",
    # Google
    "text-embedding-004", "text-embedding-005",
}

# Regex for stripping date-version suffixes like "-20240820" or "-2025-01-31"
_VERSION_SUFFIX_RE = re.compile(r"-20\d{6}$|-20\d{2}-\d{2}-\d{2}$")

# ---------------------------------------------------------------------------
# Unified output schema
# ---------------------------------------------------------------------------
UNIFIED_SCHEMA: dict[str, str] = {
    "date":                    "datetime64[ns]",
    "model":                   "object",          # original name from CSV
    "model_normalized":        "object",          # canonical / alias-resolved name
    "provider":                "object",
    "requests":                "Int64",           # nullable integer
    "input_tokens":            "Int64",
    "output_tokens":           "Int64",
    "cost_reported":           "float64",
    "is_reasoning":            "bool",
    "is_embedding":            "bool",
    "is_using_cache":          "bool",
    "cache_creation_tokens":   "Int64",
    "cache_read_tokens":       "Int64",
}


# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def detect_provider(df: pd.DataFrame) -> str:
    """
    Guess the billing CSV provider from its column signature.

    Heuristic: if any column contains 'cache' in its name → Anthropic.
    Otherwise falls back to checking for OpenAI-specific column names.

    Args:
        df: Raw DataFrame as read from CSV.

    Returns:
        "openai" | "anthropic" | "unknown"

    Raises:
        Nothing.
    """
    cols_lower = {c.lower() for c in df.columns}
    if any("cache" in c for c in cols_lower):
        return "anthropic"
    # OpenAI exports sometimes include 'organization' or 'snapshot'
    openai_hints = {"organization", "snapshot_id", "project_id"}
    if cols_lower & openai_hints:
        return "openai"
    return "unknown"


def normalize_model_name(model: str) -> str:
    """
    Resolve a raw model name from a billing CSV to its canonical name.

    Resolution order:
      1. Exact match in ALL_MODEL_ALIASES
      2. Strip date-version suffix (e.g. "-20241022") and try alias lookup again
      3. Lowercase + strip whitespace and return as-is

    Args:
        model: Raw model name string from the CSV.

    Returns:
        Canonical model name string.

    Raises:
        Nothing.
    """
    if not model or not isinstance(model, str):
        return "unknown"

    key = model.strip()

    # 1. Exact match
    if key in ALL_MODEL_ALIASES:
        return ALL_MODEL_ALIASES[key]

    # 2. Try stripping version suffix
    stripped = _VERSION_SUFFIX_RE.sub("", key)
    if stripped != key and stripped in ALL_MODEL_ALIASES:
        return ALL_MODEL_ALIASES[stripped]

    # 3. Case-insensitive exact
    key_lower = key.lower()
    for alias_key, alias_val in ALL_MODEL_ALIASES.items():
        if alias_key.lower() == key_lower:
            return alias_val

    # 4. Pass-through (lowercase)
    return key_lower


def _find_cost_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the cost column by checking COST_COL_VARIANTS against df.columns.

    Args:
        df: DataFrame with original column names.

    Returns:
        The matching column name string, or None if not found.

    Raises:
        Nothing.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for variant in COST_COL_VARIANTS:
        if variant in cols_lower:
            return cols_lower[variant]
    return None


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and strip column names, replacing spaces with underscores.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with normalised column names.

    Raises:
        Nothing.
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _apply_unified_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast each column in df to its UNIFIED_SCHEMA dtype, filling missing
    columns with sensible defaults.

    Args:
        df: DataFrame with all required columns already present.

    Returns:
        Schema-conforming DataFrame.

    Raises:
        Nothing — cast errors result in NaN/pd.NA which is handled downstream.
    """
    # Fill missing optional columns with defaults
    defaults = {
        "requests":              pd.array([0] * len(df), dtype="Int64"),
        "cache_creation_tokens": pd.array([0] * len(df), dtype="Int64"),
        "cache_read_tokens":     pd.array([0] * len(df), dtype="Int64"),
        "cost_reported":         0.0,
        "is_using_cache":        False,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col, dtype in UNIFIED_SCHEMA.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype == "bool":
                df[col] = df[col].astype(bool)
            else:
                df[col] = df[col].astype(str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema cast failed for column '%s': %s", col, exc)

    return df


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------

def parse_openai_csv(file) -> pd.DataFrame:
    """
    Parse an OpenAI billing CSV export into the unified schema DataFrame.

    Validates the file (extension, size, injection) and the resulting
    DataFrame (schema, row count) before returning.

    Args:
        file: Streamlit UploadedFile or a file-like object (for tests).

    Returns:
        Unified schema DataFrame with provider="openai".

    Raises:
        ValueError: If validation fails or required columns are missing.
    """
    # Validate file object
    if hasattr(file, "name"):  # UploadedFile
        ok, err = validate_uploaded_file(file)
        if not ok:
            raise ValueError(f"File validation failed: {err}")

    # Read CSV
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = pd.read_csv(file, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    # CSV injection check before any processing
    if detect_csv_injection(raw):
        raise ValueError("CSV injection patterns detected. File rejected.")

    raw = _normalise_column_names(raw)

    # Required columns check
    missing = OPENAI_REQUIRED_COLS - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Are you sure this is an OpenAI billing CSV?"
        )

    # DataFrame validation
    ok, err = validate_dataframe(raw)
    if not ok:
        raise ValueError(f"Data validation failed: {err}")

    # Build unified DataFrame
    df = pd.DataFrame()
    df["date"] = raw["date"]
    df["model"] = raw["model"].astype(str)
    df["model_normalized"] = df["model"].apply(normalize_model_name)
    df["provider"] = "openai"
    df["requests"] = raw.get("requests", pd.array([1] * len(raw), dtype="Int64"))
    df["input_tokens"] = raw["input_tokens"]
    df["output_tokens"] = raw["output_tokens"]

    cost_col = _find_cost_column(raw)
    df["cost_reported"] = raw[cost_col].astype(float) if cost_col else 0.0

    df["is_reasoning"] = df["model_normalized"].isin(REASONING_MODELS)
    df["is_embedding"] = df["model_normalized"].isin(EMBEDDING_MODELS)
    df["is_using_cache"] = False
    df["cache_creation_tokens"] = pd.array([0] * len(df), dtype="Int64")
    df["cache_read_tokens"] = pd.array([0] * len(df), dtype="Int64")

    df = _apply_unified_schema(df)
    return df


def parse_anthropic_csv(file) -> pd.DataFrame:
    """
    Parse an Anthropic billing CSV export into the unified schema DataFrame.

    Handles Anthropic-specific cache columns:
      - cache_creation_input_tokens (defaults to 0 if absent)
      - cache_read_input_tokens     (defaults to 0 if absent)

    Args:
        file: Streamlit UploadedFile or file-like object (for tests).

    Returns:
        Unified schema DataFrame with provider="anthropic".

    Raises:
        ValueError: If validation fails or required columns are missing.
    """
    # Validate file object
    if hasattr(file, "name"):
        ok, err = validate_uploaded_file(file)
        if not ok:
            raise ValueError(f"File validation failed: {err}")

    try:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = pd.read_csv(file, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if detect_csv_injection(raw):
        raise ValueError("CSV injection patterns detected. File rejected.")

    raw = _normalise_column_names(raw)

    missing = ANTHROPIC_REQUIRED_COLS - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Are you sure this is an Anthropic billing CSV?"
        )

    ok, err = validate_dataframe(raw)
    if not ok:
        raise ValueError(f"Data validation failed: {err}")

    df = pd.DataFrame()
    df["date"] = raw["date"]
    df["model"] = raw["model"].astype(str)
    df["model_normalized"] = df["model"].apply(normalize_model_name)
    df["provider"] = "anthropic"
    df["requests"] = raw.get("requests", pd.array([1] * len(raw), dtype="Int64"))
    df["input_tokens"] = raw["input_tokens"]
    df["output_tokens"] = raw["output_tokens"]

    cost_col = _find_cost_column(raw)
    df["cost_reported"] = raw[cost_col].astype(float) if cost_col else 0.0

    # Anthropic cache columns (optional)
    cache_create_col = next(
        (c for c in raw.columns if "cache_creation" in c), None
    )
    cache_read_col = next(
        (c for c in raw.columns if "cache_read" in c), None
    )

    df["cache_creation_tokens"] = (
        pd.to_numeric(raw[cache_create_col], errors="coerce").fillna(0).astype("Int64")
        if cache_create_col
        else pd.array([0] * len(df), dtype="Int64")
    )
    df["cache_read_tokens"] = (
        pd.to_numeric(raw[cache_read_col], errors="coerce").fillna(0).astype("Int64")
        if cache_read_col
        else pd.array([0] * len(df), dtype="Int64")
    )
    df["is_using_cache"] = (
        (df["cache_creation_tokens"] > 0) | (df["cache_read_tokens"] > 0)
    )

    df["is_reasoning"] = df["model_normalized"].isin(REASONING_MODELS)
    df["is_embedding"] = df["model_normalized"].isin(EMBEDDING_MODELS)

    df = _apply_unified_schema(df)
    return df


def merge_providers(
    openai_df: Optional[pd.DataFrame] = None,
    anthropic_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge one or both provider DataFrames into a single unified DataFrame.

    If only one provider is supplied, return it unchanged.
    If both are supplied, concatenate and return the combined frame.

    Args:
        openai_df:    Parsed OpenAI DataFrame (or None).
        anthropic_df: Parsed Anthropic DataFrame (or None).

    Returns:
        Combined unified DataFrame.

    Raises:
        ValueError: If both arguments are None.
    """
    frames = [df for df in [openai_df, anthropic_df] if df is not None and not df.empty]

    if not frames:
        raise ValueError("At least one provider DataFrame must be supplied.")

    if len(frames) == 1:
        return frames[0].copy()

    merged = pd.concat(frames, ignore_index=True)
    return merged
