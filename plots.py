from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _selected_times(T: int) -> list[int]:
    times = [0, 63, 126, 189, 251]
    return [t for t in times if 0 <= t < T]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_all_plots(
    output_dir: str | Path,
    q_grid: np.ndarray,
    V: np.ndarray,
    policy_a: np.ndarray,
    policy_b: np.ndarray,
    sim_results: dict[str, np.ndarray | pd.DataFrame],
) -> None:
    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    T = policy_a.shape[0]
    times = _selected_times(T)

    # 1) Ask fee vs inventory
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in times:
        ax.plot(q_grid, policy_a[t, :], label=f"t={t}")
    ax.set_title("Optimal Ask Fee by Inventory")
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Ask fee a*(t,q)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "policy_ask_by_inventory.png")

    # 2) Bid fee vs inventory
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in times:
        ax.plot(q_grid, policy_b[t, :], label=f"t={t}")
    ax.set_title("Optimal Bid Fee by Inventory")
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Bid fee b*(t,q)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "policy_bid_by_inventory.png")

    # 3) Spread vs inventory
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in times:
        ax.plot(q_grid, policy_a[t, :] + policy_b[t, :], label=f"t={t}")
    ax.set_title("Optimal Spread by Inventory")
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Spread a+b")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "policy_spread_by_inventory.png")

    # 4) Skew vs inventory
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in times:
        ax.plot(q_grid, policy_a[t, :] - policy_b[t, :], label=f"t={t}")
    ax.set_title("Optimal Skew by Inventory")
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Skew a-b")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "policy_skew_by_inventory.png")

    # 5) Value function by inventory
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in times:
        ax.plot(q_grid, V[t, :], label=f"t={t}")
    ax.set_title("Value Function by Inventory")
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("V_t(q)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "value_function_by_inventory.png")

    q_path = sim_results["q_path"]  # type: ignore[index]
    a_path = sim_results["a_path"]  # type: ignore[index]
    b_path = sim_results["b_path"]  # type: ignore[index]
    spread_path = sim_results["spread_path"]  # type: ignore[index]
    skew_path = sim_results["skew_path"]  # type: ignore[index]
    final_outcomes: pd.DataFrame = sim_results["final_outcomes"]  # type: ignore[assignment]
    cumulative_profit = final_outcomes["cumulative_profit"].to_numpy()

    t_axis = np.arange(q_path.shape[1])

    # 6) Inventory sample paths
    fig, ax = plt.subplots(figsize=(9, 5))
    n_show = min(50, q_path.shape[0])
    for i in range(n_show):
        ax.plot(t_axis, q_path[i, :], color="tab:blue", alpha=0.2, linewidth=0.8)
    ax.set_title("Sample Inventory Paths")
    ax.set_xlabel("t")
    ax.set_ylabel("Inventory q_t")
    ax.grid(alpha=0.3)
    _save(fig, out / "inventory_sample_paths.png")

    # 7) Mean inventory path + 5/95 bands
    fig, ax = plt.subplots(figsize=(9, 5))
    q_mean = np.mean(q_path, axis=0)
    q_p05 = np.percentile(q_path, 5, axis=0)
    q_p95 = np.percentile(q_path, 95, axis=0)
    ax.plot(t_axis, q_mean, color="tab:blue", label="mean q_t")
    ax.fill_between(t_axis, q_p05, q_p95, color="tab:blue", alpha=0.2, label="5-95 pct")
    ax.set_title("Mean Inventory Path with Bands")
    ax.set_xlabel("t")
    ax.set_ylabel("Inventory")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "inventory_mean_with_bands.png")

    # 8) Histogram of terminal inventory
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(final_outcomes["terminal_inventory"], bins=50, color="tab:purple", alpha=0.8)
    ax.set_title("Terminal Inventory Distribution")
    ax.set_xlabel("q_T")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    _save(fig, out / "terminal_inventory_hist.png")

    # 9) Histogram of cumulative risk-adjusted profit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cumulative_profit, bins=50, color="tab:green", alpha=0.8)
    ax.set_title("Cumulative Risk-Adjusted Profit Distribution")
    ax.set_xlabel("Profit = - cumulative discounted cost")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    _save(fig, out / "cumulative_profit_hist.png")

    # 10) Average ask, bid, spread, skew over time
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(T), np.mean(a_path, axis=0), label="mean ask")
    ax.plot(np.arange(T), np.mean(b_path, axis=0), label="mean bid")
    ax.plot(np.arange(T), np.mean(spread_path, axis=0), label="mean spread")
    ax.plot(np.arange(T), np.mean(skew_path, axis=0), label="mean skew")
    ax.set_title("Average Controls over Time")
    ax.set_xlabel("t")
    ax.set_ylabel("Fee")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "average_controls_over_time.png")
