from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from tqdm import tqdm

from parameters import DPParams, build_grids


class ExactDPSolver:
    def __init__(self, params: DPParams):
        self.params = params
        self.q_grid, self.a_grid, self.b_grid, self.k_grid = build_grids(params)

        self.n_q = len(self.q_grid)
        self.n_a = len(self.a_grid)
        self.n_b = len(self.b_grid)
        self.n_k = len(self.k_grid)
        self.n_controls = self.n_a * self.n_b

        self.V = np.zeros((params.T + 1, self.n_q), dtype=float)
        self.policy_a = np.zeros((params.T, self.n_q), dtype=float)
        self.policy_b = np.zeros((params.T, self.n_q), dtype=float)
        self.policy_control_index = np.zeros((params.T, self.n_q), dtype=int)

        self.control_a = np.array([])
        self.control_b = np.array([])
        self.lambda_S = np.array([])
        self.lambda_B = np.array([])
        self.expected_fee = np.array([])
        self.probs = np.array([])
        self.next_idx = np.array([])
        self.risk_matrix = np.array([])
        self.q_next_raw = np.array([])
        self.dp_diagnostics: dict[str, float | int] = {}

    def _precompute_controls(self) -> None:
        A, B = np.meshgrid(self.a_grid, self.b_grid, indexing="ij")
        self.control_a = A.ravel()
        self.control_b = B.ravel()

        self.lambda_S = self.params.alpha_S - self.params.beta_S * self.control_a
        self.lambda_B = self.params.alpha_B - self.params.beta_B * self.control_b
        if np.any(self.lambda_S < -1e-12) or np.any(self.lambda_B < -1e-12):
            raise ValueError("Negative control-dependent intensities found.")

        self.lambda_S = np.clip(self.lambda_S, 0.0, None)
        self.lambda_B = np.clip(self.lambda_B, 0.0, None)
        self.expected_fee = self.params.Q * (
            self.control_a * self.lambda_S + self.control_b * self.lambda_B
        )

    def _precompute_skellam_probs(self) -> None:
        self.probs = np.zeros((self.n_controls, self.n_k), dtype=float)

        row_sums_before = np.zeros(self.n_controls, dtype=float)
        for c in range(self.n_controls):
            mu1 = self.lambda_B[c]
            mu2 = self.lambda_S[c]
            if mu1 == 0.0 and mu2 == 0.0:
                pmf = np.zeros(self.n_k, dtype=float)
                zero_idx = np.where(self.k_grid == 0)[0][0]
                pmf[zero_idx] = 1.0
            elif mu1 == 0.0:
                # K = X - Y with X~Pois(0), Y~Pois(mu2) => K = -Y (support k<=0)
                pmf = np.where(self.k_grid <= 0, poisson.pmf(-self.k_grid, mu2), 0.0)
            elif mu2 == 0.0:
                # K = X - Y with Y~Pois(0) => K = X (support k>=0)
                pmf = np.where(self.k_grid >= 0, poisson.pmf(self.k_grid, mu1), 0.0)
            else:
                pmf = skellam.pmf(self.k_grid, mu1, mu2)
            pmf = np.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0)
            pmf = np.where(pmf < 0.0, 0.0, pmf)
            row_sums_before[c] = pmf.sum()
            if row_sums_before[c] <= 0.0:
                raise ValueError(f"Skellam row sum is non-positive for control index {c}.")
            self.probs[c, :] = pmf / row_sums_before[c]

        row_sums_after = self.probs.sum(axis=1)
        omitted_tail_mass = 1.0 - row_sums_before
        self.dp_diagnostics.update(
            {
                "min_row_sum_before_norm": float(np.min(row_sums_before)),
                "max_row_sum_before_norm": float(np.max(row_sums_before)),
                "max_omitted_tail_mass": float(np.max(omitted_tail_mass)),
                "min_row_sum_after_norm": float(np.min(row_sums_after)),
                "max_row_sum_after_norm": float(np.max(row_sums_after)),
            }
        )

        print(
            "Skellam diagnostics - row sum before norm:"
            f" min={self.dp_diagnostics['min_row_sum_before_norm']:.8f},"
            f" max={self.dp_diagnostics['max_row_sum_before_norm']:.8f}"
        )
        print(
            "Skellam diagnostics - max omitted tail mass:"
            f" {self.dp_diagnostics['max_omitted_tail_mass']:.8f}"
        )
        print(
            "Skellam diagnostics - row sum after norm:"
            f" min={self.dp_diagnostics['min_row_sum_after_norm']:.8f},"
            f" max={self.dp_diagnostics['max_row_sum_after_norm']:.8f}"
        )

    def _precompute_transitions_and_risk(self) -> None:
        self.q_next_raw = self.q_grid[:, None] + self.params.Q * self.k_grid[None, :]
        q_next_clipped = np.clip(self.q_next_raw, self.params.q_min, self.params.q_max)
        idx = np.rint((q_next_clipped - self.params.q_min) / self.params.dq).astype(int)
        self.next_idx = np.clip(idx, 0, self.n_q - 1)

        self.risk_matrix = self.params.risk_coeff * (self.params.p * self.q_next_raw) ** 2

    def solve(self) -> dict[str, np.ndarray]:
        print("Building grids...")
        print(f"q0 on grid: {bool(np.min(np.abs(self.q_grid - self.params.q0)) < 1e-12)}")
        if len(self.q_grid) > 1:
            print(f"q-grid step == Q: {bool(abs((self.q_grid[1]-self.q_grid[0]) - self.params.Q) < 1e-12)}")
        print(f"Control grids nonempty: {len(self.a_grid) > 0 and len(self.b_grid) > 0}")

        print("Precomputing controls...")
        self._precompute_controls()
        print(
            "All lambdas nonnegative:"
            f" {bool(np.all(self.lambda_S >= -1e-12) and np.all(self.lambda_B >= -1e-12))}"
        )

        print("Precomputing Skellam transition probabilities...")
        self._precompute_skellam_probs()

        self._precompute_transitions_and_risk()

        print("Solving exact DP...")
        self.V[self.params.T, :] = 0.0
        for t in tqdm(range(self.params.T - 1, -1, -1), desc="Backward induction"):
            V_next = self.V[t + 1, :]
            continuation_matrix = V_next[self.next_idx]  # (n_q, n_k)
            value_plus_risk = self.risk_matrix + self.params.delta * continuation_matrix
            J = value_plus_risk @ self.probs.T - self.expected_fee[None, :]

            best_idx = np.argmin(J, axis=1)
            rows = np.arange(self.n_q)
            self.V[t, :] = J[rows, best_idx]
            self.policy_a[t, :] = self.control_a[best_idx]
            self.policy_b[t, :] = self.control_b[best_idx]
            self.policy_control_index[t, :] = best_idx

        q0_idx = int(np.argmin(np.abs(self.q_grid - self.params.q0)))
        a0 = float(self.policy_a[0, q0_idx])
        b0 = float(self.policy_b[0, q0_idx])
        v0 = float(self.V[0, q0_idx])
        spread0 = a0 + b0
        skew0 = a0 - b0
        self.dp_diagnostics.update(
            {
                "q0_index": q0_idx,
                "V0_at_q0": v0,
                "a0_at_q0": a0,
                "b0_at_q0": b0,
                "spread0_at_q0": spread0,
                "skew0_at_q0": skew0,
            }
        )

        print(f"V_0(q0): {v0:.6f}")
        print(f"a_0*(q0): {a0:.6f}")
        print(f"b_0*(q0): {b0:.6f}")
        print(f"spread:   {spread0:.6f}")
        print(f"skew:     {skew0:.6f}")

        neg_mask = self.q_grid < 0
        pos_mask = self.q_grid > 0
        if np.any(neg_mask) and np.any(pos_mask):
            avg_skew_neg = float(np.mean(self.policy_a[0, neg_mask] - self.policy_b[0, neg_mask]))
            avg_skew_pos = float(np.mean(self.policy_a[0, pos_mask] - self.policy_b[0, pos_mask]))
            print(f"Average skew t=0, q<0: {avg_skew_neg:.6f}")
            print(f"Average skew t=0, q>0: {avg_skew_pos:.6f}")
            if avg_skew_pos >= avg_skew_neg:
                print("WARNING: average skew for q>0 is not lower than for q<0.")
            self.dp_diagnostics["avg_skew_neg_t0"] = avg_skew_neg
            self.dp_diagnostics["avg_skew_pos_t0"] = avg_skew_pos

        return {
            "V": self.V,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "policy_control_index": self.policy_control_index,
            "q_grid": self.q_grid,
            "a_grid": self.a_grid,
            "b_grid": self.b_grid,
            "k_grid": self.k_grid,
            "control_a": self.control_a,
            "control_b": self.control_b,
            "skellam_probs": self.probs,
        }

    def _save_policy_csvs(self, output_dir: Path) -> None:
        q_vals = self.q_grid
        selected_times = [0, 63, 126, 189, 251]
        selected_times = [t for t in selected_times if 0 <= t < self.params.T]

        t0 = 0
        df0 = pd.DataFrame(
            {
                "t": t0,
                "q": q_vals,
                "ask_fee": self.policy_a[t0, :],
                "bid_fee": self.policy_b[t0, :],
                "spread": self.policy_a[t0, :] + self.policy_b[t0, :],
                "skew": self.policy_a[t0, :] - self.policy_b[t0, :],
                "value": self.V[t0, :],
            }
        )
        df0.to_csv(output_dir / "policy_t0.csv", index=False)

        rows: list[pd.DataFrame] = []
        for t in selected_times:
            rows.append(
                pd.DataFrame(
                    {
                        "t": t,
                        "q": q_vals,
                        "ask_fee": self.policy_a[t, :],
                        "bid_fee": self.policy_b[t, :],
                        "spread": self.policy_a[t, :] + self.policy_b[t, :],
                        "skew": self.policy_a[t, :] - self.policy_b[t, :],
                        "value": self.V[t, :],
                    }
                )
            )
        pd.concat(rows, ignore_index=True).to_csv(output_dir / "policy_selected_times.csv", index=False)

    def save_outputs(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print("Saving policies...")
        np.save(out / "V.npy", self.V)
        np.save(out / "policy_a.npy", self.policy_a)
        np.save(out / "policy_b.npy", self.policy_b)
        np.save(out / "policy_control_index.npy", self.policy_control_index)
        np.save(out / "q_grid.npy", self.q_grid)
        np.save(out / "a_grid.npy", self.a_grid)
        np.save(out / "b_grid.npy", self.b_grid)
        np.save(out / "k_grid.npy", self.k_grid)
        np.save(out / "control_a.npy", self.control_a)
        np.save(out / "control_b.npy", self.control_b)
        np.save(out / "skellam_probs.npy", self.probs)

        with (out / "params.json").open("w", encoding="utf-8") as f:
            json.dump(self.params.to_json_dict(), f, indent=2)
        with (out / "dp_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(self.dp_diagnostics, f, indent=2)

        self._save_policy_csvs(out)
