from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import range_theta_to_pq, pq_to_range, p_to_theta_deg


# ============================================================
# Experiment settings
# ============================================================

SNR_DB = 10.0

R_VALUES = np.arange(100.0, 401.0, 25.0)
THETA_VALUES = np.arange(-50.0, 1.0, 10.0)

P_ANGLE_STEP_DEG = PRT.theta_step_deg
P_SEARCH_HALF_BINS = 2

TAU_SEARCH_HALF_SAMPLES = 2

DQ = 3e-14


def build_time_axis(range_m, theta_deg, antenna_x):
    dt, tau_true = PRT.tau_step, range_m / C
    delays = propagation_delays(range_m, theta_deg, antenna_x=antenna_x)

    margin = 8.0 * SYSTEM.gaussian_sigma
    t_min, t_max = delays.min() - margin, delays.max() + margin

    # Force tau_true exactly halfway between two time samples
    n_before = int(np.ceil((tau_true - t_min) / dt - 0.5))
    t_start = tau_true - (n_before + 0.5) * dt

    n_samples = int(np.ceil((t_max - t_start) / dt)) + 1
    return t_start + np.arange(n_samples) * dt


def build_local_p_grid(theta_true_deg):
    # Half-bin shift -> theta_true is exactly between two grid points
    offsets = np.arange(-P_SEARCH_HALF_BINS, P_SEARCH_HALF_BINS) + 0.5
    theta_grid = theta_true_deg + offsets * P_ANGLE_STEP_DEG
    p_grid = -np.sin(np.deg2rad(theta_grid)) / C

    return p_grid, theta_grid


def build_q_grid():
    R_min, R_max = np.min(R_VALUES), np.max(R_VALUES)
    theta_max = np.max(np.abs(THETA_VALUES))

    q_min = np.cos(np.deg2rad(theta_max))**2 / (2.0 * R_max * C)
    q_max = 1.0 / (2.0 * R_min * C)

    q_grid = np.arange(q_min, q_max + DQ, DQ)

    print(f"q_min   = {q_grid[0]:.3e}")
    print(f"q_max   = {q_grid[-1]:.3e}")
    print(f"Delta q = {DQ:.3e}")
    print(f"Nq      = {len(q_grid)}\n")

    return q_grid


def local_peak(prt, time_axis, p_grid, q_grid, tau_true):
    mask = np.abs(time_axis - tau_true) <= TAU_SEARCH_HALF_SAMPLES * PRT.tau_step
    tau_idx_global = np.where(mask)[0]

    local = np.abs(prt[tau_idx_global])
    idx = np.unravel_index(np.argmax(local), local.shape)

    tau_idx = tau_idx_global[idx[0]]
    p_idx, q_idx = idx[1], idx[2]

    return time_axis[tau_idx], p_grid[p_idx], q_grid[q_idx]


def run_position(range_true, theta_true, antenna_x, q_grid):
    tau_true = range_true / C

    p_true, q_true = range_theta_to_pq(range_true, theta_true)
    p_true, q_true = float(p_true), float(q_true)

    time_axis = build_time_axis(range_true, theta_true, antenna_x)
    p_grid, theta_grid = build_local_p_grid(theta_true)

    rx, _ = received_signal(
        time_axis,
        range_true,
        theta_true,
        antenna_x=antenna_x
    )

    rng = np.random.default_rng(SIM.rng_seed)
    noisy_rx, _ = add_awgn(rx, snr_db=SNR_DB, rng=rng)

    mf_rx = matched_filter(noisy_rx)

    prt = parabolic_radon_transform(
        mf_rx,
        antenna_x,
        p_grid,
        q_grid
    )

    tau_hat, p_hat, q_hat = local_peak(
        prt,
        time_axis,
        p_grid,
        q_grid,
        tau_true
    )

    range_hat = float(pq_to_range(p_hat, q_hat))
    theta_hat = float(p_to_theta_deg(p_hat))

    return {
        "R": range_true,
        "theta": theta_true,

        "R_hat": range_hat,
        "theta_hat": theta_hat,
        "tau_hat": tau_hat,

        "range_error": range_hat - range_true,
        "abs_range_error": abs(range_hat - range_true),

        "theta_error": theta_hat - theta_true,
        "abs_theta_error": abs(theta_hat - theta_true),

        "tau_error": tau_hat - tau_true,
        "abs_tau_error": abs(tau_hat - tau_true),

        "p_true": p_true,
        "p_hat": p_hat,

        "q_true": q_true,
        "q_hat": q_hat,

        "tau_true": tau_true,

        "theta_grid": theta_grid
    }


def main():
    antenna_x = ARRAY.positions(SYSTEM)
    q_grid = build_q_grid()

    results = []

    Np = 2 * P_SEARCH_HALF_BINS

    print(
        f"Positions: {len(R_VALUES)} ranges x {len(THETA_VALUES)} angles "
        f"= {len(R_VALUES) * len(THETA_VALUES)}"
    )

    print(
        f"Np={Np}, "
        f"Delta theta={P_ANGLE_STEP_DEG} deg, "
        f"theta offset=0.5 bin, "
        f"tau offset=0.5 sample, "
        f"tau window=±{TAU_SEARCH_HALF_SAMPLES} samples\n"
    )

    for theta in THETA_VALUES:
        for R in R_VALUES:
            result = run_position(R, theta, antenna_x, q_grid)
            results.append(result)

            print(
                f"R={R:6.1f} m, theta={theta:6.1f} deg | "
                f"R_hat={result['R_hat']:9.3f} m, "
                f"R_err={result['range_error']:+8.3f} m | "
                f"theta_hat={result['theta_hat']:7.3f} deg, "
                f"theta_err={result['theta_error']:+7.3f} deg | "
                f"tau_err={result['tau_error'] * 1e12:+8.3f} ps | "
                f"q_true={result['q_true']:.3e}, "
                f"q_hat={result['q_hat']:.3e}"
            )

    error_matrix = np.zeros((len(THETA_VALUES), len(R_VALUES)))

    for result in results:
        i = np.where(THETA_VALUES == result["theta"])[0][0]
        j = np.where(R_VALUES == result["R"])[0][0]
        error_matrix[i, j] = result["abs_range_error"]

    # ============================================================
    # Range error curves
    # ============================================================

    plt.figure(figsize=(9, 6))

    for i, theta in enumerate(THETA_VALUES):
        plt.plot(
            R_VALUES,
            error_matrix[i],
            marker="o",
            label=f"{theta:.0f}°"
        )

    plt.xlabel("True range R (m)")
    plt.ylabel("Absolute range error (m)")
    plt.title(f"Range estimation error (Δq = {DQ:.1e} s/m²)")
    plt.grid(True)
    plt.legend(title="Angle", ncol=2)
    plt.tight_layout()

    # ============================================================
    # Heatmap
    # ============================================================

    plt.figure(figsize=(9, 6))

    im = plt.imshow(
        error_matrix,
        origin="lower",
        aspect="auto",
        extent=[
            R_VALUES[0],
            R_VALUES[-1],
            THETA_VALUES[0],
            THETA_VALUES[-1]
        ]
    )

    plt.colorbar(im, label="Absolute range error (m)")
    plt.xlabel("True range R (m)")
    plt.ylabel("True angle θ (deg)")
    plt.title(f"Range error over user position (Δq = {DQ:.1e} s/m²)")
    plt.tight_layout()

    # ============================================================
    # Mean / maximum error over angle
    # ============================================================

    mean_error = np.mean(error_matrix, axis=0)
    max_error = np.max(error_matrix, axis=0)

    print("\n--- Error statistics over angle ---")

    for R, mean_e, max_e in zip(R_VALUES, mean_error, max_error):
        print(
            f"R={R:6.1f} m | "
            f"mean abs error={mean_e:8.3f} m | "
            f"max abs error={max_e:8.3f} m"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(R_VALUES, mean_error, marker="o", label="Mean over angles")
    plt.plot(R_VALUES, max_error, marker="s", label="Maximum over angles")

    plt.xlabel("True range R (m)")
    plt.ylabel("Absolute range error (m)")
    plt.title(f"Range error versus distance (Δq = {DQ:.1e} s/m²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()