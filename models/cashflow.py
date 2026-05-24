import numpy as np
import pandas as pd

from models.prepayment import pool_smm
from models.default import pool_default_probability, loss_given_default

MONTHS_IN_YEAR = 12.0
ORIGINAL_TERM_MONTHS = 360
EPSILON = 1e-12


def _monthly_payment(
    balance: np.ndarray,
    annual_rate: np.ndarray,
    remaining_term_months: np.ndarray,
) -> np.ndarray:
    remaining_term = np.maximum(remaining_term_months, 1.0)
    monthly_rate = annual_rate / MONTHS_IN_YEAR

    payment_with_rate = balance * monthly_rate / (
        1.0 - np.power(1.0 + monthly_rate, -remaining_term)
    )
    payment_zero_rate = balance / remaining_term

    return np.where(np.abs(monthly_rate) > EPSILON, payment_with_rate, payment_zero_rate)


def _interest_and_scheduled_principal(
    balance: np.ndarray,
    annual_rate: np.ndarray,
    remaining_term_months: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    monthly_rate = annual_rate / MONTHS_IN_YEAR
    interest = balance * monthly_rate
    payment = _monthly_payment(balance, annual_rate, remaining_term_months)
    scheduled_principal = np.minimum(np.maximum(payment - interest, 0.0), balance)
    return interest, scheduled_principal


def _current_ltv(
    current_balance: np.ndarray,
    original_balance: np.ndarray,
    original_ltv: np.ndarray,
) -> np.ndarray:
    paydown_ratio = np.divide(
        current_balance,
        original_balance,
        out=np.zeros_like(current_balance),
        where=original_balance > 0.0,
    )
    current_ltv = original_ltv * paydown_ratio
    return np.clip(current_ltv, 0.0, 2.0)


def _project_one_month(
    projection_month: int,
    current_balance: np.ndarray,
    mortgage_rate: np.ndarray,
    seasoning_months: np.ndarray,
    fico: np.ndarray,
    original_balance: np.ndarray,
    original_ltv: np.ndarray,
    market_rate: float,
    psa_speed: float,
    original_term_months: int,
) -> tuple[dict[str, float], np.ndarray]:
    remaining_term = np.maximum(1.0, original_term_months - seasoning_months)

    interest, scheduled_principal = _interest_and_scheduled_principal(
        balance=current_balance,
        annual_rate=mortgage_rate,
        remaining_term_months=remaining_term,
    )
    balance_after_scheduled = np.maximum(current_balance - scheduled_principal, 0.0)

    current_ltv = _current_ltv(current_balance, original_balance, original_ltv)
    state = pd.DataFrame(
        {
            "mortgage_rate": mortgage_rate,
            "seasoning_months": seasoning_months,
            "fico": fico,
            "ltv": current_ltv,
        }
    )

    smm = pool_smm(
        loan_pool=state,
        month=projection_month,
        market_rate=market_rate,
        psa_speed=psa_speed,
    ).to_numpy(dtype=float)
    pd_monthly = pool_default_probability(state).to_numpy(dtype=float)

    prepayment = smm * balance_after_scheduled
    balance_after_prepay = np.maximum(balance_after_scheduled - prepayment, 0.0)

    default = pd_monthly * balance_after_prepay
    lgd = loss_given_default(current_ltv)
    loss = default * lgd
    recovery = default - loss

    ending_balance = np.maximum(balance_after_prepay - default, 0.0)

    total_principal = scheduled_principal + prepayment + recovery
    total_cash_flow = interest + total_principal

    start_balance_sum = float(current_balance.sum())
    balance_weights = np.divide(
        current_balance,
        start_balance_sum,
        out=np.zeros_like(current_balance),
        where=start_balance_sum > 0.0,
    )

    month_result = {
        "month": float(projection_month),
        "starting_balance": start_balance_sum,
        "interest": float(interest.sum()),
        "scheduled_principal": float(scheduled_principal.sum()),
        "prepayment_principal": float(prepayment.sum()),
        "default_principal": float(default.sum()),
        "recovery_principal": float(recovery.sum()),
        "loss": float(loss.sum()),
        "total_principal": float(total_principal.sum()),
        "total_cash_flow": float(total_cash_flow.sum()),
        "ending_balance": float(ending_balance.sum()),
        "pool_smm": float(np.sum(smm * balance_weights)),
        "pool_pd_monthly": float(np.sum(pd_monthly * balance_weights)),
    }
    return month_result, ending_balance


def project_pool_cashflows(
    loan_pool: pd.DataFrame,
    market_rate: float,
    psa_speed: float = 100.0,
    horizon_months: int = 360,
    original_term_months: int = ORIGINAL_TERM_MONTHS,
) -> pd.DataFrame:
    """Project monthly pool-level mortgage cash flows with prepayment and default."""
    current_balance = loan_pool["balance"].to_numpy(dtype=float).copy()
    mortgage_rate = loan_pool["mortgage_rate"].to_numpy(dtype=float)
    seasoning_months = loan_pool["seasoning_months"].to_numpy(dtype=float).copy()
    fico = loan_pool["fico"].to_numpy(dtype=float)

    original_balance = current_balance.copy()
    original_ltv = loan_pool["ltv"].to_numpy(dtype=float)

    results: list[dict[str, float]] = []

    for month in range(1, horizon_months + 1):
        if current_balance.sum() <= 0.0:
            break

        month_result, ending_balance = _project_one_month(
            projection_month=month,
            current_balance=current_balance,
            mortgage_rate=mortgage_rate,
            seasoning_months=seasoning_months,
            fico=fico,
            original_balance=original_balance,
            original_ltv=original_ltv,
            market_rate=market_rate,
            psa_speed=psa_speed,
            original_term_months=original_term_months,
        )
        results.append(month_result)

        current_balance = ending_balance
        seasoning_months = seasoning_months + (current_balance > 0.0).astype(float)

    return pd.DataFrame(results)