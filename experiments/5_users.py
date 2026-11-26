
"""
Five-user severe localization test with identical center-array TOAs.

Reuses the original PRT-ROI-HRT, SIC, and joint Powell refinement
implemented in experiments/multi_user_final.py.

Displays detection results and localization geometry without saving files.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from config import C, SYSTEM, ARRAY
from signal_model.noise import add_awgn
from localization.pipeline import polar_to_xy
from experiments import multi_user_final as pipeline


# ==================== Configuration ====================

USER_POSITIONS = [
    (45.0, 75.0),
    (45.0, 125.0),
    (45.0, 375.0),
    (15.0, 350.0),
    (60.0, 300.0),
]

COMMON_TOA_NS = 1500.0
NOISE_SEED = 42

MAX_SIC_ITERATIONS = 6
HRT_STOP_THRESHOLD = 6e-6


# ==================== Scenario ====================

def create_users():
    users = []
    for theta, R in USER_POSITIONS:
        users.append({
            "range_m": R,
            "theta_deg": theta,
            "tau_ns": COMMON_TOA_NS,
            "tx_delay_ns": COMMON_TOA_NS - R / C * 1e9,
        })
    return users


# ==================== Matching ====================

def match_users(truth, detected):
    true_xy = np.array([
        polar_to_xy(u["range_m"], u["theta_deg"]) for u in truth
    ])

    estimated_xy = np.array([
        [u["x_m"], u["y_m"]] for u in detected
    ]).reshape(-1, 2)

    if len(detected) == 0:
        return [], true_xy, estimated_xy

    distances = np.linalg.norm(
        true_xy[:, None, :] - estimated_xy[None, :, :], axis=2
    )

    rows, cols = linear_sum_assignment(distances)
    matches = [(int(i), int(j), float(distances[i, j]))
               for i, j in zip(rows, cols)]

    return matches, true_xy, estimated_xy


# ==================== Visualization ====================

def plot_localization(truth, detected, matches, true_xy, estimated_xy):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(true_xy[:, 0], true_xy[:, 1], s=100, c="blue",
               marker="o", label="Ground truth", zorder=3)

    if len(detected):
        ax.scatter(estimated_xy[:, 0], estimated_xy[:, 1],
                   s=110, c="red", marker="x",
                   label="Estimated users", zorder=4)

    for i, j, error in matches:
        ax.plot([true_xy[i, 0], estimated_xy[j, 0]],
                [true_xy[i, 1], estimated_xy[j, 1]],
                "k--", alpha=0.5, lw=1)

    angles = np.deg2rad(np.linspace(-60, 60, 300))
    for R in (50, 400):
        ax.plot(R * np.sin(angles), R * np.cos(angles),
                "gray", ls=":", alpha=0.6)

    ax.set(xlabel="x (m)", ylabel="y (m)",
           title="Five-User Localization with Identical TOAs")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ==================== Main ====================

def main():
    pipeline.USERS = create_users()
    pipeline.MAX_SIC_ITERATIONS = MAX_SIC_ITERATIONS
    pipeline.HRT_STOP_THRESHOLD = HRT_STOP_THRESHOLD

    truth = pipeline.USERS
    x = ARRAY.positions(SYSTEM)
    t = pipeline.build_time_axis(x)

    clean_rx = pipeline.generate_signal(t, x)
    noisy_rx, _ = add_awgn(
        clean_rx, rng=np.random.default_rng(NOISE_SEED)
    )

    print("\n========== True Users ==========")
    for i, u in enumerate(truth, 1):
        print(f"User {i}: R={u['range_m']:.2f} m, "
              f"theta={u['theta_deg']:.3f} deg, "
              f"TOA={u['tau_ns']:.6f} ns")

    print("\n========== Running SIC ==========")

    detected, residual = pipeline.run_sic(noisy_rx, t, x)

    matches, true_xy, estimated_xy = match_users(truth, detected)

    print("\n========== Detection Results ==========")
    print(f"True users:     {len(truth)}")
    print(f"Detected users: {len(detected)}")
    print(f"Initial energy: {pipeline.energy(noisy_rx):.6e}")
    print(f"Residual energy: {pipeline.energy(residual):.6e}")

    print("\n========== Estimated Users ==========")
    for i, u in enumerate(detected, 1):
        print(f"Estimate {i}: R={u['range_m']:.3f} m, "
              f"theta={u['theta_deg']:.4f} deg, "
              f"TOA={u['tau_s']*1e9:.6f} ns")

    print("\n========== Localization Errors ==========")
    for i, j, error in matches:
        print(f"User {i+1} -> Estimate {j+1}: {error:.4f} m")

    if matches:
        errors = np.array([m[2] for m in matches])
        print(f"\nMatched users: {len(matches)}/{len(truth)}")
        print(f"ALE:  {np.mean(errors):.4f} m")
        print(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f} m")
        print(f"Max:  {np.max(errors):.4f} m")

    if len(detected) != len(truth):
        print("\nWARNING: Detected user count differs from ground truth.")
        print("Matching results are partial and do not imply full detection.")

    plot_localization(truth, detected, matches, true_xy, estimated_xy)


if __name__ == "__main__":
    main()
