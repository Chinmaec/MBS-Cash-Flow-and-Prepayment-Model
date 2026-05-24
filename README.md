# MBS Cash Flow & Prepayment Model

A compact, research-style Python project for modeling mortgage pool cash flows with prepayment and default risk, then converting those projected cash flows into core fixed-income analytics.

The objective is clarity and financial correctness, not overengineering.

## What This Project Models

For a synthetic mortgage pool, the model projects monthly:

- interest
- scheduled principal
- unscheduled principal (prepayment)
- defaulted principal
- recovery and loss
- ending balance

With the following analytics:

- weighted average life (WAL)
- yield (from price)
- Macaulay duration
- modified duration
- cumulative loss rate

## Financial Logic

### 1) Prepayment

- Baseline prepayment follows PSA:
  - CPR ramps linearly from 0% to 6% over the first 30 months (100 PSA).
- CPR is converted to monthly SMM:
  - `SMM = 1 - (1 - CPR)^(1/12)`
- A refinancing incentive adjusts CPR based on `(mortgage rate - market rate)`.

Interpretation:
- When market rates fall below mortgage rates, refinance incentive rises.
- Faster prepayments return principal earlier, shortening WAL/duration.
- This is the core MBS negative convexity intuition.

### 2) Default

Loan-level monthly default probability is modeled with the following logistic regression features:

- FICO
- LTV
- seasoning

Form:
- `PD = sigmoid(beta0 + beta_fico*x_fico + beta_ltv*x_ltv + beta_seasoning*x_seasoning)`


### 3) LGD (Loss Given Default)

LGD is deterministic and increasing in LTV, with floor/cap bounds.

Interpretation:
- higher leverage -> weaker collateral recovery and higher loss severity

### 4) Cash Flow Engine

Monthly sequence per loan:

1. compute scheduled payment and interest
2. compute scheduled principal
3. apply prepayment to post-scheduled balance
4. apply default to remaining balance
5. apply LGD to defaulted amount to split recovery vs loss
6. roll forward ending balance

Pool cash flow is the sum across all loans each month.

## Run

pip install -r requirements.txt
python main.py