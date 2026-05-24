import numpy as np
import pandas as pd

MONTHS_IN_YEAR = 12.0

BETA_0 = -4.20
BETA_FICO = -0.55
BETA_LTV = 3.25
BETA_SEASONING = -0.06

MIN_PD = 0.0001
MAX_PD = 0.0800

LGD_BASE = 0.20
LGD_LTV_SLOPE = 0.60
LGD_ANCHOR_LTV = 0.60
MIN_LGD = 0.10
MAX_LGD = 0.65

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Return elementwise logistic transform."""
    return 1.0 / (1.0 + np.exp(-x))


def monthly_default_probability(
    fico: np.ndarray | pd.Series,
    ltv: np.ndarray | pd.Series,
    seasoning_months: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute monthly default probability using a simple logistic model."""
    fico_array = np.asarray(fico, dtype=float)
    ltv_array = np.asarray(ltv, dtype=float)
    seasoning_array = np.asarray(seasoning_months, dtype=float)

    fico_norm = (fico_array - 700.0) / 50.0
    ltv_centered = ltv_array - 0.80
    seasoning_years = seasoning_array / MONTHS_IN_YEAR

    logit = (
        BETA_0
        + BETA_FICO * fico_norm
        + BETA_LTV * ltv_centered
        + BETA_SEASONING * seasoning_years
    )
    pd_monthly = _sigmoid(logit)
    return np.clip(pd_monthly, MIN_PD, MAX_PD)

def pool_default_probability(loan_pool: pd.DataFrame) -> pd.Series:
    """Return loan-level monthly default probabilities for a pool."""
    pd_monthly = monthly_default_probability(
        fico=loan_pool["fico"].to_numpy(),
        ltv=loan_pool["ltv"].to_numpy(),
        seasoning_months=loan_pool["seasoning_months"].to_numpy(),
    )
    return pd.Series(pd_monthly, index=loan_pool.index, name="pd_monthly")


def loss_given_default(ltv: np.ndarray | pd.Series | float) -> np.ndarray:
    """Return deterministic LGD as a function of current LTV."""
    ltv_array = np.asarray(ltv, dtype=float)
    raw_lgd = LGD_BASE + LGD_LTV_SLOPE * (ltv_array - LGD_ANCHOR_LTV)
    return np.clip(raw_lgd, MIN_LGD, MAX_LGD)