"""Run Project 8 Problem 3(c), changing only the dataset noise level."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from write_parameters import default_parameters
from plantga import (
    GeneticAlgorithmEstimator,
    generate_dataset,
    normalized_mse,
    simulate_heights,
)


NOISE_SIGMA = 0.20


def main() -> None:
    params = default_parameters()
    params["dataset"]["noise_sigma"] = NOISE_SIGMA
    data = generate_dataset(true_params=params["true_params"], **params["dataset"])

    estimator = GeneticAlgorithmEstimator(bounds=params["bounds"], **dict(params["ga"]))
    print(f"Running GA with noise_sigma = {NOISE_SIGMA} and otherwise default parameters...")
    result = estimator.run(data)
    best_params = result["best_params"]
    print(f"Converged in {result['generations_run']} generations.")

    time = data["time"]
    positions = data["positions"]
    h_pred_train = simulate_heights(
        params=best_params,
        initial_heights=data["initial_heights"],
        time=time,
        W=data["W"],
        F=data["F"],
        S=data["S"],
        positions=positions,
    )
    h_pred_test = simulate_heights(
        params=best_params,
        initial_heights=data["initial_heights_test"],
        time=time,
        W=data["W_test"],
        F=data["F_test"],
        S=data["S_test"],
        positions=positions,
    )

    train_mse = normalized_mse(data["h_obs"][data["train_idx"]], h_pred_train[data["train_idx"]])
    test_mse = normalized_mse(data["h_obs_test"], h_pred_test)

    print(f"\nTrain MSE: {train_mse:.4f}")
    print(f"Test  MSE: {test_mse:.4f}")
    print("\nParameter Recovery Table (high noise):")
    print(f"{'Parameter':<10} {'True':>10} {'Recovered':>12} {'% Error':>10}")
    print("-" * 46)
    for name in result["param_names"]:
        true_value = params["true_params"][name]
        recovered_value = best_params[name]
        pct_error = abs(recovered_value - true_value) / (abs(true_value) + 1e-12) * 100
        print(f"{name:<10} {true_value:>10.4f} {recovered_value:>12.4f} {pct_error:>9.1f}%")

    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, obs, pred, clean, label, color in [
        (axes[0], data["h_obs"], h_pred_train, data["h_clean"], "Training Season (high noise)", "steelblue"),
        (axes[1], data["h_obs_test"], h_pred_test, data["h_clean_test"], "Held-out Test Season (high noise)", "darkorange"),
    ]:
        ax.plot(time, obs.mean(axis=1), ".", color=color, alpha=0.35, markersize=3, label="Noisy observations")
        ax.plot(time, clean.mean(axis=1), "-", color=color, linewidth=1.7, alpha=0.8, label="True/noiseless mean")
        ax.plot(time, pred.mean(axis=1), "--", color="black", linewidth=1.9, label="GA prediction mean")
        ax.set_title(label)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean plant height")
    fig.suptitle(f"High Noise Parameter Recovery - Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig("figures/trajectories_3c.png", dpi=150, bbox_inches="tight")
    print("\nSaved figures/trajectories_3c.png")


if __name__ == "__main__":
    main()
