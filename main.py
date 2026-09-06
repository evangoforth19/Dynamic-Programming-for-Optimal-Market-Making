from __future__ import annotations

import json
import time
from pathlib import Path

from dp_solver import ExactDPSolver
from parameters import DPParams
from plots import create_all_plots
from simulate_policy import save_simulation_outputs, simulate_optimal_policy


def main() -> None:
    t_start = time.time()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = DPParams()

    print("Parameter summary:")
    print(f"  T={params.T}, delta={params.delta:.10f}, q0={params.q0}, Q={params.Q}")
    print(f"  W0={params.W0:.2f}, SR_Y_ann={params.SR_Y_ann:.6f}, risk_coeff={params.risk_coeff:.12e}")
    print(f"  Kmax={params.Kmax}, n_sims={params.n_sims}, seed={params.random_seed}")

    solver = ExactDPSolver(params)
    dp_results = solver.solve()
    solver.save_outputs(output_dir)

    print("Simulating optimal policy...")
    sim_results = simulate_optimal_policy(
        params=params,
        q_grid=dp_results["q_grid"],
        policy_a=dp_results["policy_a"],
        policy_b=dp_results["policy_b"],
        n_sims=params.n_sims,
        seed=params.random_seed,
    )
    save_simulation_outputs(sim_results, output_dir)

    print("Creating plots...")
    create_all_plots(
        output_dir=output_dir,
        q_grid=dp_results["q_grid"],
        V=dp_results["V"],
        policy_a=dp_results["policy_a"],
        policy_b=dp_results["policy_b"],
        sim_results=sim_results,
    )

    t_end = time.time()
    runtime_sec = t_end - t_start
    manifest = {
        "runtime_seconds": runtime_sec,
        "output_dir": str(output_dir),
        "n_q": int(len(dp_results["q_grid"])),
        "n_controls": int(len(dp_results["control_a"])),
        "n_k": int(len(dp_results["k_grid"])),
        "n_sims": params.n_sims,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"runtime: {runtime_sec:.2f} seconds")
    print("Done.")


if __name__ == "__main__":
    main()
