# Finite-Horizon Dynamic Programming for Market-Maker Fees

This project implements the exact finite-horizon dynamic programming (DP) formulation for optimal market-maker bid/ask fee setting with inventory risk.

## Purpose

The market maker chooses ask fee `a_t` and bid fee `b_t` each period to trade off:
- fee income from client flow, and
- Sharpe-adjusted inventory risk penalties.

The objective is to minimize expected discounted cost over a finite horizon.

## State, Control, and Disturbance

- **State**: inventory only, `q_t`.
- **Control**: `(a_t, b_t)` from discretized fee grids.
- **Disturbance**:
  - `N_t^S ~ Poisson(lambda_S(a_t))` (public buys; dealer sells),
  - `N_t^B ~ Poisson(lambda_B(b_t))` (public sells; dealer buys),
  - `K_t = N_t^B - N_t^S ~ Skellam(lambda_B(b_t), lambda_S(a_t))`.

Inventory transition:

`q_{t+1} = q_t + Q * K_t`.

## Exact Bellman Recursion

Terminal condition:

`V_T(q) = 0`

Recursion:

`V_t(q) =
    min_{a,b}
    {
        sum_k P(K=k | a,b)
        [
            risk_coeff * (p(q+Qk))^2
            + delta V_{t+1}(Projection(q+Qk))
        ]
        - Q(a lambda_S(a) + b lambda_B(b))
    }`

where `Projection(q+Qk)` is nearest-grid projection clipped to `[q_min, q_max]`.

## Why the DP state is q-only

This implementation is intentionally computationally feasible and exact (up to Skellam truncation):
- `q_t` is the only dynamic state in the DP.
- `F`, `p`, and `Y` are **not** included as states.
- No stochastic `p_t` or `Y_t` is simulated inside the DP.

`p` and return assumptions affect decisions only through the fixed scalar `risk_coeff`.

## How return assumptions enter

Risk coefficient:

`risk_coeff = (SR_Y_ann / (2 * sigma_Y_ann * W0)) * sigma_perp_var_daily`

with:
- `SR_Y_ann = (mu_Y_ann - risk_free_rate_ann) / sigma_Y_ann`
- `sigma_perp_var_daily = [sigma_S_ann^2 * (1-rho_SY^2)] / 252`.

## Run

From this folder:

```bash
python main.py
```

## Outputs

The script creates files in `outputs/`, including:
- `V.npy`, `policy_a.npy`, `policy_b.npy`, `policy_control_index.npy`
- `q_grid.npy`, `a_grid.npy`, `b_grid.npy`, `k_grid.npy`
- `control_a.npy`, `control_b.npy`, `skellam_probs.npy`
- `params.json`, `dp_diagnostics.json`
- `policy_t0.csv`, `policy_selected_times.csv`
- `simulation_summary.csv`, `simulation_final_outcomes.csv`
- `simulation_paths_sample.csv`, `simulation_arrays.npz`
- `run_manifest.json`
- plots in `outputs/plots/*.png`.

## Changing parameters

Edit defaults in `parameters.py` (`DPParams`), then rerun:

```bash
python main.py
```
