from pathlib import Path
import pandas as pd

from data.generate_loan_data import generate_loan_pool
from models.cashflow import project_pool_cashflows
from analytics.risk_metrics import summarize_risk_metrics

NUM_LOANS = 5000
RANDOM_SEED = 42
MARKET_RATE = 0.045
PSA_SPEED = 100.0
HORIZON_MONTHS = 360
PRICE_PCT = 100.0
RESULTS_DIR = Path("results")


def run_projection(
    num_loans: int = NUM_LOANS,
    seed: int = RANDOM_SEED,
    market_rate: float = MARKET_RATE,
    psa_speed: float = PSA_SPEED,
    horizon_months: int = HORIZON_MONTHS,
    price_pct: float = PRICE_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run full MBS pipeline: pool generation, cash flow projection, and risk metrics."""
    loan_pool = generate_loan_pool(num_loans=num_loans, seed=seed)
    cashflow_df = project_pool_cashflows(
        loan_pool=loan_pool,
        market_rate=market_rate,
        psa_speed=psa_speed,
        horizon_months=horizon_months,
    )
    metrics = summarize_risk_metrics(cashflow_df=cashflow_df, price_pct=price_pct)
    return loan_pool, cashflow_df, metrics


def save_results(
    loan_pool: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    metrics: dict[str, float],
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Save core outputs to results folder."""
    results_dir.mkdir(parents=True, exist_ok=True)

    loan_pool.to_csv(results_dir / "loan_pool_snapshot.csv", index=False)
    cashflow_df.to_csv(results_dir / "projected_cashflows.csv", index=False)
    pd.DataFrame([metrics]).to_csv(results_dir / "risk_metrics.csv", index=False)


def print_summary(metrics: dict[str, float]) -> None:
    """Print compact analytics summary."""
    print("MBS Risk Summary")
    print(f"WAL (years): {metrics['wal_years']:.4f}")
    print(f"Annual Yield: {metrics['annual_yield']:.4%}")
    print(f"Macaulay Duration (years): {metrics['macaulay_duration_years']:.4f}")
    print(f"Modified Duration (years): {metrics['modified_duration_years']:.4f}")
    print(f"Cumulative Loss Rate: {metrics['cumulative_loss_rate']:.4%}")


def main() -> None:
    """Execute baseline scenario and persist results."""
    loan_pool, cashflow_df, metrics = run_projection()
    save_results(loan_pool=loan_pool, cashflow_df=cashflow_df, metrics=metrics)
    print_summary(metrics)

    print(f"\nLoans: {len(loan_pool)}")
    print(f"Projected Months: {len(cashflow_df)}")
    print(f"Final Ending Balance: {cashflow_df['ending_balance'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()