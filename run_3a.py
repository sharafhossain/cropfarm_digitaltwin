"""Run GA parameter recovery for Project 8, Problem 3a."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plantga import (
    generate_dataset,
    GeneticAlgorithmEstimator,
    normalized_mse,
    simulate_heights,
)

# ── 1. Generate data ──────────────────────────────────────────────────────────
data = generate_dataset(noise_sigma=0.03, seed=3)

true_params  = data["true_params"]
time         = data["time"]
positions    = data["positions"]

# ── 2. Define search bounds ───────────────────────────────────────────────────
bounds = {
    "G0":     (0.01, 0.30),
    "b_w":    (0.10, 1.50),
    "b_f":    (0.05, 1.00),
    "gamma":  (0.50, 2.00),
    "alpha":  (0.50, 2.00),
    "lambda": (0.01, 0.50),
    "rho":    (0.50, 3.00),
}

# ── 3. Run GA ─────────────────────────────────────────────────────────────────
ga = GeneticAlgorithmEstimator(
    bounds=bounds,
    population_size=60,
    num_generations=200,
    parent_fraction=0.4,
    mutation_rate=0.2,
    mutation_scale=0.08,
    elitism=2,
    patience=30,
    seed=42,
)

print("Running GA...")
result = ga.run(data)
best_params = result["best_params"]
print(f"Converged in {result['generations_run']} generations.")

# ── 4. Compute train and test MSE with best params ────────────────────────────
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

train_mse = normalized_mse(data["h_obs"], h_pred_train)
test_mse  = normalized_mse(data["h_obs_test"], h_pred_test)

print(f"\nTrain MSE: {train_mse:.4f}")
print(f"Test  MSE: {test_mse:.4f}")

# ── 5. Print parameter recovery table ────────────────────────────────────────
print("\nParameter Recovery Table:")
print(f"{'Parameter':<10} {'True':>10} {'Recovered':>12} {'% Error':>10}")
print("-" * 46)
for name in true_params:
    tv = true_params[name]
    rv = best_params[name]
    pct = abs(rv - tv) / (abs(tv) + 1e-12) * 100
    print(f"{name:<10} {tv:>10.4f} {rv:>12.4f} {pct:>9.1f}%")

# ── 6. Plot trajectories ──────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, (h_obs, h_pred, h_clean, label, color) in zip(axes, [
    (data["h_obs"],      h_pred_train, data["h_clean"],      "Training Season",   "steelblue"),
    (data["h_obs_test"], h_pred_test,  data["h_clean_test"],  "Test Season",       "darkorange"),
]):
    mean_obs   = h_obs.mean(axis=1)
    mean_pred  = h_pred.mean(axis=1)
    mean_clean = h_clean.mean(axis=1)

    ax.plot(time, mean_obs,   ".", color=color, alpha=0.4, markersize=3, label="Noisy observations")
    ax.plot(time, mean_clean, "-", color=color, linewidth=1.5, alpha=0.7, label="True (noiseless)")
    ax.plot(time, mean_pred,  "--", color="black", linewidth=1.8, label="GA prediction")

    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Mean plant height (m)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Parameter Recovery — Train MSE: {train_mse:.4f}  |  Test MSE: {test_mse:.4f}",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig("figures/trajectories_3a.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/trajectories_3a.png")
