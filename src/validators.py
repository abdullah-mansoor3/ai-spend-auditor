"""
validators.py — Input validation and security hardening for AI Spend Auditor.

All public functions return (bool, str) tuples so the caller always knows
what went wrong without needing to catch exceptions.  Nothing here writes to
disk or stores state — pure stateless validation.
"""

import re
import logging
from io import BytesIO

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants — adjust here for quick tuning
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTENSIONS: set[str] = {".csv"}
MAX_ROWS: int = 100_000
MAX_COLUMNS: int = 50
MAX_CELL_STRING_LENGTH: int = 10_000          # potential injection attempt threshold
MAX_NAN_FRACTION: float = 0.20               # reject numeric column if >20% NaN after coerce

# Columns we must have after column aliasing (validated in parse.py after alias mapping)
REQUIRED_COLUMNS: set[str] = {"date", "model", "input_tokens", "output_tokens"}

# Excel / Sheets formula injection prefixes
CSV_INJECTION_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@")

# Prompt injection patterns (case-insensitive)
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all)\s+instructions?",
        r"disregard\s+(previous|all|the)",
        r"new\s+task\s*:",
        r"forget\s+(everything|all|previous)",
        r"system\s*:\s*you\s+are",
        r"<\s*system\s*>",
        r"\bDAN\b",                         # "Do Anything Now" jailbreak token
        r"act\s+as\s+if\s+you\s+(have|are)",
    ]
]

_HTML_TAG_PATTERN: re.Pattern = re.compile(r"<[^>]+>")
_SAFE_FILENAME_PATTERN: re.Pattern = re.compile(r"[^a-zA-Z0-9_\-\.]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_uploaded_file(file) -> tuple[bool, str]:
    """
    Validate a Streamlit UploadedFile object before any parsing.

    Checks (in order):
      1. Extension must be .csv
      2. File size must be ≤ MAX_FILE_SIZE_BYTES
      3. Filename must not contain path traversal sequences
      4. Filename characters are stripped to alphanumeric + dash + underscore + dot

    Args:
        file: A Streamlit ``UploadedFile`` instance.

    Returns:
        (True, "") on success, (False, human-readable error) on failure.

    Raises:
        Nothing — all errors are returned as the second tuple element.
    """
    if file is None:
        return False, "No file provided."

    name: str = file.name or ""

    # 1. Extension check
    if not any(name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, f"Only .csv files are accepted. Got: '{name}'"

    # 2. Path traversal guard
    if ".." in name or "/" in name or "\\" in name:
        return False, "Filename contains invalid path characters."

    # 3. File size
    try:
        size: int = file.size  # Streamlit exposes .size directly
    except AttributeError:
        # Fall back to reading the buffer
        data = file.read()
        size = len(data)
        file.seek(0)

    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        return False, f"File too large: {mb:.1f} MB (max {MAX_FILE_SIZE_BYTES // (1024*1024)} MB)."

    # 4. Sanitised filename check (informational — we don't mutate the name here)
    sanitized = _SAFE_FILENAME_PATTERN.sub("", name)
    if not sanitized:
        return False, "Filename is empty after sanitisation."

    return True, ""


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate a parsed DataFrame before running analysis.

    Checks (in order):
      1. Row count ≤ MAX_ROWS
      2. Column count ≤ MAX_COLUMNS
      3. No object column contains strings longer than MAX_CELL_STRING_LENGTH
      4. Required columns are present (date, model, input_tokens, output_tokens)
      5. Numeric columns (input_tokens, output_tokens) are actually numeric;
         rejects if >20 % NaN after coercion

    Args:
        df: A pandas DataFrame produced by initial CSV reading.

    Returns:
        (True, "") on success, (False, human-readable error) on failure.

    Raises:
        Nothing — all errors are returned as the second tuple element.
    """
    if df is None or df.empty:
        return False, "The uploaded file appears to be empty."

    # 1. Row count
    if len(df) > MAX_ROWS:
        return False, f"File has {len(df):,} rows — maximum allowed is {MAX_ROWS:,}."

    # 2. Column count
    if len(df.columns) > MAX_COLUMNS:
        return False, f"File has {len(df.columns)} columns — maximum allowed is {MAX_COLUMNS}."

    # 3. String cell length (injection/DoS guard)
    for col in df.select_dtypes(include=["object"]).columns:
        max_len = df[col].dropna().astype(str).str.len().max()
        if max_len > MAX_CELL_STRING_LENGTH:
            return (
                False,
                f"Column '{col}' contains strings longer than {MAX_CELL_STRING_LENGTH} characters. "
                "This may indicate a malformed or malicious file.",
            )

    # 4. Required columns — normalise column names for comparison
    normalised_cols = {c.strip().lower().replace(" ", "_") for c in df.columns}
    missing = REQUIRED_COLUMNS - normalised_cols
    # Also accept common cost column variants (handled later in parser)
    if missing - {"cost"}:   # cost is optional at validation stage
        actual_missing = missing - {"cost"}
        if actual_missing:
            return (
                False,
                f"Missing required columns: {', '.join(sorted(actual_missing))}. "
                "Please check the CSV format or the export instructions.",
            )

    # 5. Numeric column sanity check
    numeric_candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("token", "request", "cost", "amount"))
    ]
    for col in numeric_candidates:
        coerced = pd.to_numeric(df[col], errors="coerce")
        nan_frac = coerced.isna().mean()
        if nan_frac > MAX_NAN_FRACTION:
            return (
                False,
                f"Column '{col}' has {nan_frac:.0%} non-numeric values after coercion. "
                "Expected a numeric column.",
            )

    return True, ""


def sanitize_text_input(text: str, max_length: int = 500) -> str:
    """
    Sanitise free-text user input from the 'describe your app' field.

    Steps:
      1. Strip HTML tags
      2. Remove prompt injection patterns
      3. Truncate to max_length

    Args:
        text:       Raw string from the Streamlit text_input widget.
        max_length: Maximum number of characters to keep (default 500).

    Returns:
        Sanitised string, guaranteed ≤ max_length characters.

    Raises:
        Nothing.
    """
    if not text:
        return ""

    # 1. Strip HTML
    cleaned = _HTML_TAG_PATTERN.sub("", text)

    # 2. Remove prompt injection patterns
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[removed]", cleaned)

    # 3. Truncate
    cleaned = cleaned[:max_length].strip()

    return cleaned


def detect_csv_injection(df: pd.DataFrame) -> bool:
    """
    Detect potential CSV/spreadsheet formula injection in string columns.

    Returns True if any string cell starts with one of the formula-trigger
    characters: = + - @

    The check is intentionally conservative — a leading '-' in a numeric-
    looking string is fine, but we flag it here and let the caller decide
    (typically by rejecting the file).

    Args:
        df: DataFrame to inspect.

    Returns:
        True if injection patterns are detected, False otherwise.

    Raises:
        Nothing — failures are logged and return False to avoid false positives
        crashing the app.
    """
    try:
        for col in df.select_dtypes(include=["object"]).columns:
            suspicious = df[col].dropna().astype(str).str.startswith(
                CSV_INJECTION_PREFIXES
            )
            if suspicious.any():
                logger.warning(
                    "CSV injection attempt detected in column '%s': %d suspicious cells.",
                    col,
                    suspicious.sum(),
                )
                return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during CSV injection check: %s", exc)
    return False
