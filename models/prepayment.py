import numpy as np
import pandas as pd

PSA_PEAK_CPR = 0.06
PSA_RAMP_MONTHS = 30
MONTHS_IN_YEAR = 12
REFI_BETA = 25.0
MIN_REFI_MULTIPLIER = 0.40
MAX_REFI_MULTIPLIER = 3.00
MAX_CPR = 0.60

def psa_cpr(month: int, psa_speed: float = 100.0) -> float:
    """Return annual CPR from PSA convention for a given month."""
    ramp_month = max(1, month)
    base_cpr = PSA_PEAK_CPR * min(ramp_month, PSA_RAMP_MONTHS) / PSA_RAMP_MONTHS
    return base_cpr * (psa_speed / 100.0)

def cpr_to_smm(cpr: np.ndarray | float) -> np.ndarray | float:
    """Convert annual CPR to monthly SMM."""
    return 1.0 - np.power(1.0 - cpr, 1.0 / MONTHS_IN_YEAR)

def refinancing_multiplier(
    mortgage_rate: np.ndarray | pd.Series | float,
    market_rate: float,
    beta: float = REFI_BETA,
) -> np.ndarray:
    """Return prepayment multiplier from refinancing incentive."""
    mortgage_rate_array = np.asarray(mortgage_rate, dtype=float)
    spread = mortgage_rate_array - market_rate
    raw_multiplier = 1.0 + beta * spread
    return np.clip(raw_multiplier, MIN_REFI_MULTIPLIER, MAX_REFI_MULTIPLIER)

def loan_smm(
    mortgage_rate: np.ndarray | pd.Series,
    month: int,
    market_rate: float,
    psa_speed: float = 100.0,
) -> np.ndarray:
    """Return loan-level SMMs after PSA and refinance adjustment."""
    base_cpr = psa_cpr(month=month, psa_speed=psa_speed)
    incentive_multiplier = refinancing_multiplier(mortgage_rate=mortgage_rate, market_rate=market_rate)
    adjusted_cpr = np.clip(base_cpr * incentive_multiplier, 0.0, MAX_CPR)
    return cpr_to_smm(adjusted_cpr)

def pool_smm(
    loan_pool: pd.DataFrame,
    month: int,
    market_rate: float,
    psa_speed: float = 100.0,
) -> pd.Series:
    """Compute pool loan-by-loan SMM for a projection month."""
    smm = loan_smm(
        mortgage_rate=loan_pool["mortgage_rate"].to_numpy(),
        month=month,
        market_rate=market_rate,
        psa_speed=psa_speed,
    )
    return pd.Series(smm, index=loan_pool.index, name="smm")
