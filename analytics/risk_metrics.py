import numpy as np
import pandas as pd

MONTHS_IN_YEAR = 12.0
MIN_ANNUAL_YIELD = -0.90
MAX_ANNUAL_YIELD = 1.50
YIELD_TOLERANCE = 1e-10
MAX_BISECTION_STEPS = 200


def _require_columns(cashflow_df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in cashflow_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _price_from_annual_yield(
    months: np.ndarray,
    cashflows: np.ndarray,
    annual_yield: float,
) -> float:
    periodic_rate = 1.0 + annual_yield / MONTHS_IN_YEAR
    if periodic_rate <= 0.0:
        return np.inf
    discount = np.power(periodic_rate, months)
    return float(np.sum(cashflows / discount))


def yield_from_price(
    cashflow_df: pd.DataFrame,
    price_pct: float = 100.0,
    face_value: float | None = None,
) -> float:
    """Solve annualized yield (monthly compounding) from projected cash flows and price."""
    _require_columns(cashflow_df, ["month", "total_cash_flow", "starting_balance"])
    if cashflow_df.empty:
        raise ValueError("cashflow_df is empty.")

    notional = float(cashflow_df["starting_balance"].iloc[0]) if face_value is None else float(face_value)
    target_price = notional * (price_pct / 100.0)

    months = cashflow_df["month"].to_numpy(dtype=float)
    cashflows = cashflow_df["total_cash_flow"].to_numpy(dtype=float)

    low, high = MIN_ANNUAL_YIELD, MAX_ANNUAL_YIELD
    f_low = _price_from_annual_yield(months, cashflows, low) - target_price
    f_high = _price_from_annual_yield(months, cashflows, high) - target_price

    if f_low * f_high > 0.0:
        raise ValueError("Yield root is not bracketed by current bounds.")

    for _ in range(MAX_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        f_mid = _price_from_annual_yield(months, cashflows, mid) - target_price
        if abs(f_mid) < YIELD_TOLERANCE:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def macaulay_duration_years(cashflow_df: pd.DataFrame, annual_yield: float) -> float:
    """Compute Macaulay duration in years from projected monthly cash flows."""
    _require_columns(cashflow_df, ["month", "total_cash_flow"])
    if cashflow_df.empty:
        return 0.0

    months = cashflow_df["month"].to_numpy(dtype=float)
    cashflows = cashflow_df["total_cash_flow"].to_numpy(dtype=float)

    periodic_rate = 1.0 + annual_yield / MONTHS_IN_YEAR
    if periodic_rate <= 0.0:
        raise ValueError("annual_yield is too low for monthly compounding.")

    discount = np.power(periodic_rate, months)
    pv_cashflows = cashflows / discount
    price = float(np.sum(pv_cashflows))
    if price <= 0.0:
        return 0.0

    times_years = months / MONTHS_IN_YEAR
    return float(np.sum(times_years * pv_cashflows) / price)


def modified_duration_years(cashflow_df: pd.DataFrame, annual_yield: float) -> float:
    """Compute modified duration in years from Macaulay duration."""
    macaulay = macaulay_duration_years(cashflow_df=cashflow_df, annual_yield=annual_yield)
    return macaulay / (1.0 + annual_yield / MONTHS_IN_YEAR)


def weighted_average_life_years(cashflow_df: pd.DataFrame) -> float:
    """Compute principal-weighted average life (WAL) in years."""
    _require_columns(cashflow_df, ["month", "total_principal"])
    if cashflow_df.empty:
        return 0.0

    months = cashflow_df["month"].to_numpy(dtype=float)
    principal = np.maximum(cashflow_df["total_principal"].to_numpy(dtype=float), 0.0)

    total_principal = float(np.sum(principal))
    if total_principal <= 0.0:
        return 0.0

    wal_months = float(np.sum(months * principal) / total_principal)
    return wal_months / MONTHS_IN_YEAR


def cumulative_loss_rate(cashflow_df: pd.DataFrame) -> float:
    """Compute cumulative loss divided by original pool balance."""
    _require_columns(cashflow_df, ["loss", "starting_balance"])
    if cashflow_df.empty:
        return 0.0

    original_balance = float(cashflow_df["starting_balance"].iloc[0])
    if original_balance <= 0.0:
        return 0.0

    total_loss = float(np.sum(np.maximum(cashflow_df["loss"].to_numpy(dtype=float), 0.0)))
    return total_loss / original_balance


def summarize_risk_metrics(
    cashflow_df: pd.DataFrame,
    price_pct: float = 100.0,
    face_value: float | None = None,
) -> dict[str, float]:
    """Return core pool analytics: yield, WAL, duration, and cumulative loss rate."""
    annual_yield = yield_from_price(cashflow_df=cashflow_df, price_pct=price_pct, face_value=face_value)
    wal = weighted_average_life_years(cashflow_df)
    macaulay = macaulay_duration_years(cashflow_df=cashflow_df, annual_yield=annual_yield)
    modified = modified_duration_years(cashflow_df=cashflow_df, annual_yield=annual_yield)
    cum_loss = cumulative_loss_rate(cashflow_df)

    return {
        "wal_years": wal,
        "annual_yield": annual_yield,
        "macaulay_duration_years": macaulay,
        "modified_duration_years": modified,
        "cumulative_loss_rate": cum_loss,
    }