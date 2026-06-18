"""
generate_samples.py — Generates three high-quality, realistic billing CSV exports
for testing the AI Spend Auditor:
  1. data/sample_openai_waste.csv: heavy OpenAI waste (triggers Rules A, B, C, D)
  2. data/sample_anthropic_waste.csv: heavy Anthropic waste with cache columns
  3. data/sample_optimized.csv: fully optimized billing showing clean health check
"""

import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

def generate_samples():
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    days = 30
    start_dt = date(2026, 5, 1)

    # 1. OpenAI High Waste Sample
    openai_rows = []
    # Profiles: (avg_in, avg_out, req_range, cost_in_1M, cost_out_1M)
    profiles = {
        # Rule A: Expensive model used for short-output tasks
        "gpt-4o": (1800, 150, (1000, 1800), 2.50, 10.00),
        # Rule B: Context bloat (high average input tokens)
        "gpt-4o-large-input": (8500, 400, (400, 900), 2.50, 10.00),
        # Rule C: Reasoning model overuse (huge spend, small tasks)
        "o1-preview": (3500, 200, (150, 350), 15.00, 60.00),
        "o3-mini": (2200, 300, (200, 600), 1.10, 4.40),
        # Rule D: Embedding call batching issues (lots of tiny calls)
        "text-embedding-3-small": (80, 0, (6000, 15000), 0.02, 0.00),
        # Baseline mini usage
        "gpt-4o-mini": (900, 250, (3000, 7000), 0.15, 0.60)
    }

    for d in range(days):
        curr_date = start_dt + timedelta(days=d)
        date_str = curr_date.strftime("%Y-%m-%d")
        for model_name, (avg_in, avg_out, req_range, in_p, out_p) in profiles.items():
            reqs = int(np.random.randint(*req_range))
            in_t = int(reqs * avg_in * np.random.uniform(0.85, 1.15))
            out_t = int(reqs * avg_out * np.random.uniform(0.85, 1.15))
            cost = (in_t * in_p + out_t * out_p) / 1_000_000
            
            # Map back custom profile keys to standard names
            real_model = "gpt-4o" if model_name == "gpt-4o-large-input" else model_name
            
            openai_rows.append({
                "Date": date_str,
                "Model": real_model,
                "Requests": reqs,
                "Input tokens": in_t,
                "Output tokens": out_t,
                "Cost": round(cost, 4)
            })

    df_openai = pd.DataFrame(openai_rows)
    df_openai.to_csv("data/sample_openai_waste.csv", index=False)
    print(f"Generated data/sample_openai_waste.csv | Total spend: ${df_openai['Cost'].sum():.2f}")

    # 2. Anthropic Waste Sample
    anthropic_rows = []
    anthropic_profiles = {
        # Rule A: Sonnet used for simple tasks (short outputs)
        "claude-3-5-sonnet-20241022": (2200, 180, (800, 1500), 3.00, 15.00),
        # Context bloat Sonnet (heavy system prompts)
        "claude-3-5-sonnet-bloat": (9000, 600, (300, 700), 3.00, 15.00),
        # Baseline Haiku usage
        "claude-3-haiku-20240307": (1200, 300, (2000, 5000), 0.25, 1.25)
    }

    for d in range(days):
        curr_date = start_dt + timedelta(days=d)
        date_str = curr_date.strftime("%Y-%m-%d")
        for model_name, (avg_in, avg_out, req_range, in_p, out_p) in anthropic_profiles.items():
            reqs = int(np.random.randint(*req_range))
            in_t = int(reqs * avg_in * np.random.uniform(0.85, 1.15))
            out_t = int(reqs * avg_out * np.random.uniform(0.85, 1.15))
            
            # Simulated cache columns (30% cache rate)
            cache_creation = int(in_t * 0.1)
            cache_read = int(in_t * 0.2)
            
            cost = ((in_t - cache_read) * in_p + (cache_read * in_p * 0.1) + out_t * out_p) / 1_000_000
            
            real_model = "claude-3-5-sonnet-20241022" if model_name == "claude-3-5-sonnet-bloat" else model_name
            
            anthropic_rows.append({
                "Date": date_str,
                "Model": real_model,
                "Requests": reqs,
                "Input tokens": in_t,
                "Output tokens": out_t,
                "Cache creation input tokens": cache_creation,
                "Cache read input tokens": cache_read,
                "Cost": round(cost, 4)
            })

    df_anthropic = pd.DataFrame(anthropic_rows)
    df_anthropic.to_csv("data/sample_anthropic_waste.csv", index=False)
    print(f"Generated data/sample_anthropic_waste.csv | Total spend: ${df_anthropic['Cost'].sum():.2f}")

    # 3. Optimized Sample
    opt_rows = []
    opt_profiles = {
        # Mini model for short/medium tasks (no Rule A trigger)
        "gpt-4o-mini": (1000, 300, (4000, 8000), 0.15, 0.60),
        # Large model used correctly (high avg output tokens)
        "gpt-4o": (1500, 1100, (200, 500), 2.50, 10.00),
        # Embedding model batched correctly (large avg input)
        "text-embedding-3-small": (4500, 0, (200, 500), 0.02, 0.00)
    }

    for d in range(days):
        curr_date = start_dt + timedelta(days=d)
        date_str = curr_date.strftime("%Y-%m-%d")
        for model, (avg_in, avg_out, req_range, in_p, out_p) in opt_profiles.items():
            reqs = int(np.random.randint(*req_range))
            in_t = int(reqs * avg_in * np.random.uniform(0.85, 1.15))
            out_t = int(reqs * avg_out * np.random.uniform(0.85, 1.15))
            cost = (in_t * in_p + out_t * out_p) / 1_000_000
            opt_rows.append({
                "Date": date_str,
                "Model": model,
                "Requests": reqs,
                "Input tokens": in_t,
                "Output tokens": out_t,
                "Cost": round(cost, 4)
            })

    df_opt = pd.DataFrame(opt_rows)
    df_opt.to_csv("data/sample_optimized.csv", index=False)
    print(f"Generated data/sample_optimized.csv | Total spend: ${df_opt['Cost'].sum():.2f}")

if __name__ == "__main__":
    generate_samples()
