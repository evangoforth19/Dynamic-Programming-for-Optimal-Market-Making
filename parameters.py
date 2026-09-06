from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class DPParams:
    # Time
    T: int = 252
    annual_interest_rate: float = 0.06

    # Initial balance sheet
    p: float = 20.0
    F0: float = 800_000.0
    Y0: float = 1_000_000.0
    q0: int = 10_000

    # Order size
    Q: int = 100

    # Inventory grid
    q_min: int = -20_000
    q_max: int = 20_000
    dq: int = 100

    # Ask fee grid
    a_min: float = 0.0
    a_max: float = 0.05
    da: float = 0.005

    # Bid fee grid
    b_min: float = 0.0
    b_max: float = 0.05
    db: float = 0.005

    # Order flow parameters
    alpha_S: float = 25.0
    alpha_B: float = 25.0
    beta_S: float = 500.0
    beta_B: float = 500.0

    # Return and risk parameters
    mu_Y_ann: float = 0.12
    sigma_Y_ann: float = 0.15
    risk_free_rate_ann: float = 0.06
    sigma_S_ann: float = 0.30
    rho_SY: float = 0.50

    # Skellam truncation
    Kmax: int = 50

    # Simulation
    n_sims: int = 5000
    random_seed: int = 123

    # Derived parameters (filled in __post_init__)
    delta: float = 0.0
    W0: float = 0.0
    SR_Y_ann: float = 0.0
    sigma_perp_var_ann: float = 0.0
    sigma_perp_var_daily: float = 0.0
    risk_coeff: float = 0.0

    def __post_init__(self) -> None:
        self.compute_derived()
        self.validate_derived()

    def compute_derived(self) -> None:
        self.delta = (1.0 + self.annual_interest_rate) ** (-1.0 / 252.0)
        self.W0 = self.F0 + self.p * self.q0 + self.Y0
        self.SR_Y_ann = (self.mu_Y_ann - self.risk_free_rate_ann) / self.sigma_Y_ann
        self.sigma_perp_var_ann = self.sigma_S_ann**2 * (1.0 - self.rho_SY**2)
        self.sigma_perp_var_daily = self.sigma_perp_var_ann / 252.0
        self.risk_coeff = (
            self.SR_Y_ann / (2.0 * self.sigma_Y_ann * self.W0)
        ) * self.sigma_perp_var_daily

    def validate_derived(self) -> None:
        if self.W0 <= 0:
            raise ValueError("W0 must be positive.")
        if self.risk_coeff <= 0:
            raise ValueError("risk_coeff must be positive.")
        if self.dq != self.Q:
            raise ValueError("dq must equal Q for the requested inventory grid.")

    def lambda_S(self, a: np.ndarray | float) -> np.ndarray | float:
        return self.alpha_S - self.beta_S * a

    def lambda_B(self, b: np.ndarray | float) -> np.ndarray | float:
        return self.alpha_B - self.beta_B * b

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in data.items()}


def build_grids(params: DPParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_grid = np.round(np.arange(params.q_min, params.q_max + params.dq, params.dq), 10)
    a_grid = np.round(np.arange(params.a_min, params.a_max + params.da / 2.0, params.da), 10)
    b_grid = np.round(np.arange(params.b_min, params.b_max + params.db / 2.0, params.db), 10)
    k_grid = np.round(np.arange(-params.Kmax, params.Kmax + 1), 10)

    # Required validation checks
    if len(a_grid) == 0 or len(b_grid) == 0:
        raise ValueError("Control grids must be nonempty.")
    if np.min(np.abs(q_grid - params.q0)) > 1e-12:
        raise ValueError("q0 must lie on q_grid.")
    if len(q_grid) > 1:
        step = q_grid[1] - q_grid[0]
        if abs(step - params.Q) > 1e-12:
            raise ValueError("q_grid step must equal Q.")
    if np.any(params.lambda_S(a_grid) < -1e-12):
        raise ValueError("lambda_S(a_grid) must be nonnegative.")
    if np.any(params.lambda_B(b_grid) < -1e-12):
        raise ValueError("lambda_B(b_grid) must be nonnegative.")
    if params.W0 <= 0:
        raise ValueError("W0 must be positive.")
    if params.risk_coeff <= 0:
        raise ValueError("risk_coeff must be positive.")

    return q_grid, a_grid, b_grid, k_grid
