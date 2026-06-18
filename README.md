# 💸 AI Spend Auditor

A Streamlit-based web tool that analyzes AI spend CSV exports and automatically generates a detailed audit report with actionable recommendations, powered by Groq's LLMs.

This tool acts as a **diagnostic sales funnel** for early-stage founders and small startup CTOs at 2–20 person companies spending $200–$3,000/month on LLM inference.

## 🎯 Overview

**AI Spend Auditor** helps companies identify unnecessary AI costs and provides data-driven recommendations on where to optimize. It transforms raw spend data into intelligent insights, highlighting cost-saving opportunities and suggesting alternative models.

### Key Features

- **Spend Analysis**: Analyzes input/output tokens, model usage, and request patterns.
- **Automated Rules Engine**: Detects common waste patterns like model-task mismatch, context bloat, reasoning model overuse, and embedding inefficiency.
- **AI-Generated Reports**: Uses Groq LLMs (`llama-3.3-70b-versatile`) to write comprehensive audit reports with recommendations.
- **Interactive Dashboard**: Visualizes spend trends, model performance, and saving opportunities.
- **Privacy First**: Processes all CSV data in-memory only; no raw logs, prompts, or billing data leave the browser session.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 to 3.12 (NumPy/Sentence-Transformers compatible)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-spend-auditor
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and fill in your API keys:
     ```env
     # Groq API keys (rotated if rate limits are hit)
     GROQ_API_KEY=gsk_...
     GROQ_API_KEY_2=gsk_...
     GROQ_API_KEY_3=gsk_...

     # PostHog analytics config
     POSTHOG_API_KEY=phc_...
     POSTHOG_HOST=https://us.i.posthog.com
     ```

## 💻 Run the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Usage

1. **Upload AI Spend CSV**: Drag and drop your OpenAI and/or Anthropic billing export CSVs.
2. **Context**: (Optional) Describe your app in one sentence to make the generated report specific to your use case.
3. **Generate Report**: Click "Analyze my spend" to parse, normalize, run rules, and call Groq for narrative report generation.
4. **Take Action**: Review recommendations, and use the embedded Tally form to request optimization services.

## 📋 Rule Configuration

You can tune the detection rules by modifying the thresholds in `src/rules.py`:

```python
# Rule A - Model-task mismatch
RULE_A_MIN_REQUESTS: int = 50         # minimum requests to trigger rule
RULE_A_MAX_AVG_OUTPUT: int = 400      # max avg output tokens for simple tasks

# Rule B - Context bloat
RULE_B_MIN_AVG_INPUT: int = 2_000     # minimum avg input tokens to trigger
RULE_B_COMPRESSION_RATE: float = 0.40 # conservative 40% compression estimate

# Rule C - Reasoning overuse
RULE_C_MIN_REASONING_PCT: float = 15.0  # reasoning % of spend to trigger
RULE_C_OVERSPEC_RATE: float = 0.60      # 60% of reasoning calls assumed over-specified

# Rule D - Embedding inefficiency
RULE_D_MAX_AVG_TOKENS: int = 200        # max avg tokens per embedding call (batching check)
```

## 🧩 Data Schema & Mappings

The parser handles raw standard CSV exports from OpenAI and Anthropic consoles automatically.

### Expected Raw CSV Formats

#### OpenAI billing export CSV
Requires:
- `date`
- `model`
- `input_tokens` (or token counts)
- `output_tokens`
- `cost` (or `cost_usd`, `total_cost`, `amount`)

#### Anthropic billing export CSV
Requires:
- `date`
- `model`
- `input_tokens`
- `output_tokens`
- Optional cache columns: `cache_creation_input_tokens`, `cache_read_input_tokens`

### Model Downgrade Map (SOTA June 2026)
Defined in `src/pricing.py`:
- `gpt-4o` / `gpt-4-turbo` → `gpt-4o-mini`
- `gpt-5-pro` → `gpt-5` → `gpt-5-mini`
- `o1` / `o1-preview` → `gpt-4o`
- `o3` / `o4` → `o4-mini` → `gpt-4o-mini`
- `claude-3-opus` / `claude-opus-4` → `claude-sonnet-4` / `claude-sonnet-4-6`
- `claude-3-5-sonnet` → `claude-3-5-haiku` / `claude-3-haiku`
- `claude-3-7-sonnet` → `claude-3-5-sonnet`
- `deepseek-r1` → `deepseek-v4` → `deepseek-v4-flash`
- `kimi-k2-7` → `kimi-k2-6` → `kimi-k2` → `deepseek-v4-flash`
- `gemini-3.1-pro` → `gemini-3.5-flash`

## 📊 Report Structure

Each audit report includes:

### Summary Metrics
- **Monthly Spend**: Projected monthly cost.
- **Recoverable Waste**: In dollars per month.
- **Annual Impact**: Projected annual savings (12 × monthly waste).

### Findings Expanders
- **Explanation**: Jargon-free, Groq-generated or fallback template narrative.
- **Affected Models**: Lists models triggering the rule.
- **Recommended Action**: Exact technical step to take to achieve savings.
- **Urgency Note**: Chronological context on why this matters now.

## 🛠 Development

### Running Tests

Run the unit tests using `pytest`:

```bash
python -m pytest tests/ -v
```

### Adding New Rules

To add a new detection rule:
1. Define the rule logic in `src/rules.py` returning a `RuleResult` TypedDict.
2. Register and aggregate the rule in `run_all_rules()` in `src/rules.py`.

## 🔮 Roadmap

The following future features are planned after v1 validation:

- **v2 — Log-based duplicate detection**: Accept LiteLLM JSON log export, extract prompts, and cluster semantically to find exact duplicates.
- **v2 — GitHub code scanner**: Scan codebase for API calls and audit code directly.
- **v3 — Automated monitoring**: User connects read-only API keys for weekly automated email audits.
- **v3 — Multi-workspace support**: User auth via magic links and trend/history views.
- **v4 — LiteLLM proxy integration**: One-click connect to user's self-hosted proxy for real-time monitoring.

## 🔐 Security & Privacy

- **Memory Only**: All dataframes are processed in-memory and discarded after the browser session is closed.
- **No Data Storage**: No raw billing data, tokens, or descriptions are ever written to disk or sent to any server except the aggregated metrics to PostHog and aggregated numeric findings to Groq.
