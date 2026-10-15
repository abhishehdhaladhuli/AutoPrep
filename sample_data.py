"""A deliberately messy customer-churn dataset for trying the app without your own file."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_data(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Customer churn data with the usual problems: placeholders, '$1,234' text, mixed case,
    stray spaces, leading-zero zip codes, outliers, skew, duplicates, an ID column, a constant
    column, a duplicated/correlated column and some rows with no target."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.normal(65, 25, n).clip(15, 150), 2)
    support_calls = rng.poisson(1.5, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "None"], n, p=[0.35, 0.45, 0.20])
    income = np.round(rng.lognormal(10.5, 0.7, n))                       # right-skewed
    logit = (-1.6 + 1.1 * (contract == "Month-to-month") - 0.02 * tenure
             + 0.012 * (monthly - 65) + 0.22 * support_calls)
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No").astype(object)

    age = rng.integers(18, 80, n).astype(float)
    age[rng.choice(n, 4, replace=False)] = 150                           # impossible ages
    age[rng.choice(n, int(0.05 * n), replace=False)] = np.nan

    gender = rng.choice(["Male", "Female"], n).astype(object)
    idx = rng.choice(n, int(0.15 * n), replace=False)
    gender[idx] = [g.lower() if i % 2 == 0 else f"  {g.upper()} " for i, g in enumerate(gender[idx])]

    city = rng.choice(["Hyderabad", "Mumbai", "Delhi", "Pune", "Chennai", "Kolkata"], n).astype(object)
    city[rng.choice(n, int(0.04 * n), replace=False)] = rng.choice(["NA", "?", "n/a", ""], int(0.04 * n))

    monthly_txt = np.array([f"${v:,.2f}" for v in monthly], dtype=object)
    monthly_txt[rng.choice(n, int(0.03 * n), replace=False)] = "-"

    tenure_f = tenure.astype(float)
    tenure_f[rng.choice(n, int(0.03 * n), replace=False)] = np.nan
    income = income.astype(float)
    income[rng.choice(n, int(0.06 * n), replace=False)] = np.nan
    churn[rng.choice(n, 8, replace=False)] = np.nan                       # rows with no target

    df = pd.DataFrame({
        "Customer ID": [f"CUST-{i:05d}" for i in range(n)],
        "Age": age,
        "Gender": gender,
        "City": city,
        "Zip Code": rng.choice(["02134", "10001", "00501", "94105", "60601"], n),
        "Contract": contract,
        "Internet Service": internet,
        "Monthly Charges": monthly_txt,
        "Tenure (months)": tenure_f,
        "Tenure (years)": np.round(tenure / 12, 3),                      # same information
        "Support Calls": support_calls,
        "Annual Income": income,
        "Country": "India",                                              # constant
        "Churn": churn,
    })
    df = pd.concat([df, df.sample(12, random_state=seed)], ignore_index=True)   # duplicate rows
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
