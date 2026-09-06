from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parameters import DPParams


def _q_to_index(q: float, params: DPParams, n_q: int) -> int:
    q_clip = np.clip(q, params.q_min, params.q_max)
    idx = int(np.rint((q_clip - params.q_min) / params.dq))
    return int(np.clip(idx, 0, n_q - 1))


def simulate_optimal_policy(
    params: DPParams,
    q_grid: np.ndarray,
    policy_a: np.ndarray,
    policy_b: np.ndarray,
    n_sims: int,
    seed: int,
) -> dict[str, np.ndarray | pd.DataFrame]:
    T = params.T
    n_q = len(q_grid)
    rng = np.random.default_rng(seed)

    q_path = np.zeros((n_sims, T + 1), dtype=float)
    F_path = np.zeros((n_sims, T + 1), dtype=float)
    a_path = np.zeros((n_sims, T), dtype=float)
    b_path = np.zeros((n_sims, T), dtype=float)
    spread_path = np.zeros((n_sims, T), dtype=float)
    skew_path = np.zeros((n_sims, T), dtype=float)
    NS_path = np.zeros((n_sims, T), dtype=int)
    NB_path = np.zeros((n_sims, T), dtype=int)
    fee_income_path = np.zeros((n_sims, T), dtype=float)
    risk_charge_path = np.zeros((n_sims, T), dtype=float)
    cost_path = np.zeros((n_sims, T), dtype=float)
    discounted_cost_path = np.zeros((n_sims, T), dtype=float)

    q_path[:, 0] = params.q0
    F_path[:, 0] = params.F0

    discount_vec = params.delta ** np.arange(T)

    for m in range(n_sims):
        q = float(params.q0)
        F = float(params.F0)
        for t in range(T):
            q_idx = _q_to_index(q, params, n_q)
            a = policy_a[t, q_idx]
            b = policy_b[t, q_idx]

            lam_S = params.alpha_S - params.beta_S * a
            lam_B = params.alpha_B - params.beta_B * b
            lam_S = max(lam_S, 0.0)
            lam_B = max(lam_B, 0.0)

            NS = int(rng.poisson(lam_S))
            NB = int(rng.poisson(lam_B))

            q_next = q + params.Q * (NB - NS)
            F_next = F + (params.p + a) * params.Q * NS - (params.p - b) * params.Q * NB

            fees = a * params.Q * NS + b * params.Q * NB
            risk = params.risk_coeff * (params.p * q_next) ** 2
            cost = risk - fees
            dcost = discount_vec[t] * cost

            a_path[m, t] = a
            b_path[m, t] = b
            spread_path[m, t] = a + b
            skew_path[m, t] = a - b
            NS_path[m, t] = NS
            NB_path[m, t] = NB
            fee_income_path[m, t] = fees
            risk_charge_path[m, t] = risk
            cost_path[m, t] = cost
            discounted_cost_path[m, t] = dcost
            q_path[m, t + 1] = q_next
            F_path[m, t + 1] = F_next

            q = q_next
            F = F_next

    cumulative_cost = discounted_cost_path.sum(axis=1)
    cumulative_profit = -cumulative_cost
    total_fee_income = fee_income_path.sum(axis=1)
    total_risk_charge = risk_charge_path.sum(axis=1)
    terminal_inventory = q_path[:, -1]
    terminal_cash = F_path[:, -1]
    terminal_wealth = terminal_cash + params.p * terminal_inventory + params.Y0

    final_outcomes = pd.DataFrame(
        {
            "sim_id": np.arange(n_sims, dtype=int),
            "cumulative_cost": cumulative_cost,
            "cumulative_profit": cumulative_profit,
            "total_fee_income": total_fee_income,
            "total_risk_charge": total_risk_charge,
            "terminal_inventory": terminal_inventory,
            "terminal_cash": terminal_cash,
            "terminal_wealth": terminal_wealth,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "mean_cumulative_cost": float(np.mean(cumulative_cost)),
                "std_cumulative_cost": float(np.std(cumulative_cost, ddof=1)),
                "mean_cumulative_profit": float(np.mean(cumulative_profit)),
                "std_cumulative_profit": float(np.std(cumulative_profit, ddof=1)),
                "mean_terminal_inventory": float(np.mean(terminal_inventory)),
                "std_terminal_inventory": float(np.std(terminal_inventory, ddof=1)),
                "mean_terminal_wealth": float(np.mean(terminal_wealth)),
                "std_terminal_wealth": float(np.std(terminal_wealth, ddof=1)),
                "mean_total_fee_income": float(np.mean(total_fee_income)),
                "mean_total_risk_charge": float(np.mean(total_risk_charge)),
            }
        ]
    )

    return {
        "q_path": q_path,
        "F_path": F_path,
        "a_path": a_path,
        "b_path": b_path,
        "spread_path": spread_path,
        "skew_path": skew_path,
        "NS_path": NS_path,
        "NB_path": NB_path,
        "fee_income_path": fee_income_path,
        "risk_charge_path": risk_charge_path,
        "cost_path": cost_path,
        "discounted_cost_path": discounted_cost_path,
        "final_outcomes": final_outcomes,
        "summary": summary,
    }


def save_simulation_outputs(sim_results: dict[str, np.ndarray | pd.DataFrame], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: pd.DataFrame = sim_results["summary"]  # type: ignore[assignment]
    final_outcomes: pd.DataFrame = sim_results["final_outcomes"]  # type: ignore[assignment]
    summary.to_csv(out / "simulation_summary.csv", index=False)
    final_outcomes.to_csv(out / "simulation_final_outcomes.csv", index=False)

    sample_n = min(25, len(final_outcomes))
    sample_paths = pd.DataFrame(
        {
            "sim_id": np.repeat(np.arange(sample_n), sim_results["q_path"].shape[1]),  # type: ignore[index]
            "t": np.tile(np.arange(sim_results["q_path"].shape[1]), sample_n),  # type: ignore[index]
            "q": sim_results["q_path"][:sample_n, :].reshape(-1),  # type: ignore[index]
            "F": sim_results["F_path"][:sample_n, :].reshape(-1),  # type: ignore[index]
        }
    )
    sample_paths.to_csv(out / "simulation_paths_sample.csv", index=False)

    np.savez_compressed(
        out / "simulation_arrays.npz",
        q_path=sim_results["q_path"],
        F_path=sim_results["F_path"],
        a_path=sim_results["a_path"],
        b_path=sim_results["b_path"],
        spread_path=sim_results["spread_path"],
        skew_path=sim_results["skew_path"],
        NS_path=sim_results["NS_path"],
        NB_path=sim_results["NB_path"],
        fee_income_path=sim_results["fee_income_path"],
        risk_charge_path=sim_results["risk_charge_path"],
        cost_path=sim_results["cost_path"],
        discounted_cost_path=sim_results["discounted_cost_path"],
    )
