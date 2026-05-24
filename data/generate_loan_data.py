import numpy as np
import pandas as pd
from pathlib import Path

MIN_BALANCE = 75_000.0
MAX_BALANCE = 1_250_000.0
RATE_MEAN = 0.0525
RATE_STD = 0.0100
MIN_RATE = 0.0250
MAX_RATE = 0.0900
FICO_MEAN = 740.0
FICO_STD = 45.0
MIN_FICO = 620
MAX_FICO = 850
MIN_LTV = 0.45
MAX_LTV = 0.97
MAX_SEASONING = 360

STATE_WEIGHTS = {
    "CA": 0.18,
    "TX": 0.11,
    "FL": 0.10,
    "NY": 0.08,
    "NJ": 0.06,
    "IL": 0.06,
    "PA": 0.06,
    "GA": 0.06,
    "NC": 0.06,
    "VA": 0.05,
    "WA": 0.05,
    "AZ": 0.05,
    "Other": 0.08,
}

def generate_loan_pool(num_loans: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic agency-style mortgage pool."""
    rng = np.random.default_rng(seed)

    balance = np.clip(rng.lognormal(mean=12.55, sigma=0.45, size=num_loans), MIN_BALANCE, MAX_BALANCE)
    mortgage_rate = np.clip(rng.normal(RATE_MEAN, RATE_STD, size=num_loans), MIN_RATE, MAX_RATE)
    fico = np.clip(rng.normal(FICO_MEAN, FICO_STD, size=num_loans), MIN_FICO, MAX_FICO).round().astype(int)
    seasoning = (rng.beta(a=2.0, b=2.8, size=num_loans) * MAX_SEASONING).round().astype(int)

    fico_z = (fico - FICO_MEAN) / FICO_STD
    base_ltv = 0.78 - 0.04 * fico_z + rng.normal(0.0, 0.05, size=num_loans)
    ltv = np.clip(base_ltv, MIN_LTV, MAX_LTV)

    states = rng.choice(
        list(STATE_WEIGHTS.keys()),
        size=num_loans,
        p=np.array(list(STATE_WEIGHTS.values())),
    )

    return pd.DataFrame(
        {
            "loan_id": np.arange(1, num_loans + 1),
            "balance": balance.round(2),
            "mortgage_rate": mortgage_rate.round(6),
            "seasoning_months": seasoning,
            "fico": fico,
            "ltv": ltv.round(4),
            "state": states,
        }
    )


def save_loan_pool_csv(output_path: str | Path, num_loans: int = 5000, seed: int = 42) -> None:
    """Generate a synthetic pool and save it as a CSV file."""
    loan_pool = generate_loan_pool(num_loans=num_loans, seed=seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    loan_pool.to_csv(path, index=False)


if __name__ == "__main__":
    save_loan_pool_csv("data/loan_pool.csv", num_loans=5000, seed=42)    