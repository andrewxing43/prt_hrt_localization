"""
HRT accuracy experiment.

Purpose:
- Randomly sample continuous user positions.
- Use fixed HRT resolutions: 0.1 m, 0.1 deg, and dt/4 in tau.
- Keep true parameters naturally off-grid using fixed global lattices.
- Measure range, angle, tau, and 2-D Cartesian localization accuracy.

This experiment does NOT use PRT, ROI, or pipeline.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.hrt import hyperbolic_radon_transform, hrt_peak


# ============================================================
# Experiment settings
# ============================================================

SNR_DB = SIM.snr_db
N_POSITIONS = 250

R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0

CLOCK_OFFSET = 10e-9

HRT_DR = 1
HRT_DTHETA = PRT.theta_step_deg
HRT_DTAU = SYSTEM.dt/4

R_HALF_WIDTH = 50.0
THETA_HALF_WIDTH = 0.2
TAU_HALF_WIDTH = 2 * SYSTEM.dt


# ============================================================
# Fixed global HRT lattices
# ============================================================

R_GLOBAL = np.arange(0.0, 500.0 + HRT_DR, HRT_DR)
THETA_GLOBAL = np.arange(-65.0, 65.0 + 0.5 * HRT_DTHETA, HRT_DTHETA)


# ============================================================
# Helpers
# ============================================================

def build_time_axis(R, theta, clock_offset, antenna_x):
    delays = propagation_delays(R, theta, clock_offset=clock_offset, antenna_x=antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def local_grid(global_grid, value, half_width):
    mask = (global_grid >= value - half_width) & (global_grid <= value + half_width)
    return global_grid[mask]


def local_tau_grid(tau_true):
    n0 = int(np.ceil((tau_true - TAU_HALF_WIDTH) / HRT_DTAU))
    n1 = int(np.floor((tau_true + TAU_HALF_WIDTH) / HRT_DTAU))
    return np.arange(n0, n1 + 1) * HRT_DTAU


def summarize(x):
    x = np.asarray(x)
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "p95": np.percentile(x, 95),
        "p99": np.percentile(x, 99),
        "max": np.max(x),
    }


def polar_to_xy(R, theta_deg):
    theta = np.deg2rad(theta_deg)
    x = R * np.sin(theta)
    y = R * np.cos(theta)
    return x, y


# ============================================================
# Main
# ============================================================

def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    records = []

    print("=" * 80)
    print("HRT ACCURACY EXPERIMENT")
    print("=" * 80)
    print(f"Positions        : {N_POSITIONS}")
    print(f"SNR              : {SNR_DB:.1f} dB")
    print(f"R resolution     : {HRT_DR:.3f} m")
    print(f"theta resolution : {HRT_DTHETA:.3f} deg")
    print(f"tau resolution   : {HRT_DTAU * 1e12:.6f} ps")
    print()

    progress = tqdm(range(N_POSITIONS), desc="HRT positions", unit="pos")

    for _ in progress:
        R_true = rng_pos.uniform(R_MIN, R_MAX)
        theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
        tau_true = R_true / C + CLOCK_OFFSET

        time_axis = build_time_axis(R_true, theta_true, CLOCK_OFFSET, antenna_x)
        clean_rx, _ = received_signal(
            time_axis,
            R_true,
            theta_true,
            clock_offset=CLOCK_OFFSET,
            antenna_x=antenna_x,
        )
        noisy_rx, _ = add_awgn(clean_rx, snr_db=SNR_DB, rng=rng_noise)
        mf_rx = matched_filter(noisy_rx)

        range_grid = local_grid(R_GLOBAL, R_true, R_HALF_WIDTH)
        theta_grid = local_grid(THETA_GLOBAL, theta_true, THETA_HALF_WIDTH)
        tau_grid = local_tau_grid(tau_true)

        hrt = hyperbolic_radon_transform(
            mf_rx,
            time_axis,
            antenna_x,
            range_grid,
            theta_grid,
            tau_grid,
        )
        tau_hat, R_hat, theta_hat, _, _ = hrt_peak(
            hrt,
            tau_grid,
            range_grid,
            theta_grid,
        )

        x_true, y_true = polar_to_xy(R_true, theta_true)
        x_hat, y_hat = polar_to_xy(R_hat, theta_hat)

        dx = x_hat - x_true
        dy = y_hat - y_true
        xy_error = np.hypot(dx, dy)

        records.append({
            "R_true": R_true,
            "theta_true": theta_true,
            "tau_true": tau_true,
            "R_hat": R_hat,
            "theta_hat": theta_hat,
            "tau_hat": tau_hat,
            "R_error": abs(R_hat - R_true),
            "theta_error": abs(theta_hat - theta_true),
            "tau_error": abs(tau_hat - tau_true),
            "x_true": x_true,
            "y_true": y_true,
            "x_hat": x_hat,
            "y_hat": y_hat,
            "dx": dx,
            "dy": dy,
            "xy_error": xy_error,
        })

    R_true = np.array([x["R_true"] for x in records])
    theta_true = np.array([x["theta_true"] for x in records])

    R_error = np.array([x["R_error"] for x in records])
    theta_error = np.array([x["theta_error"] for x in records])
    tau_error = np.array([x["tau_error"] for x in records]) * 1e12

    dx = np.array([x["dx"] for x in records])
    dy = np.array([x["dy"] for x in records])
    xy_error = np.array([x["xy_error"] for x in records])

    R_stat = summarize(R_error)
    theta_stat = summarize(theta_error)
    tau_stat = summarize(tau_error)
    xy_stat = summarize(xy_error)

    average_localization_error = np.mean(xy_error)

    rmse_x = np.sqrt(np.mean(dx**2))
    rmse_y = np.sqrt(np.mean(dy**2))
    rmse_xy = np.sqrt(np.mean(dx**2 + dy**2))

    print("\n" + "=" * 80)
    print("RANGE ERROR")
    print("=" * 80)
    print(f"Mean   : {R_stat['mean']:.4f} m")
    print(f"Median : {R_stat['median']:.4f} m")
    print(f"P95    : {R_stat['p95']:.4f} m")
    print(f"P99    : {R_stat['p99']:.4f} m")
    print(f"Max    : {R_stat['max']:.4f} m")

    print("\nANGLE ERROR")
    print("=" * 80)
    print(f"Mean   : {theta_stat['mean']:.4f} deg")
    print(f"P95    : {theta_stat['p95']:.4f} deg")
    print(f"P99    : {theta_stat['p99']:.4f} deg")
    print(f"Max    : {theta_stat['max']:.4f} deg")

    print("\nTAU ERROR")
    print("=" * 80)
    print(f"Mean   : {tau_stat['mean']:.4f} ps")
    print(f"P95    : {tau_stat['p95']:.4f} ps")
    print(f"P99    : {tau_stat['p99']:.4f} ps")
    print(f"Max    : {tau_stat['max']:.4f} ps")

    print("\n2-D POSITION ERROR")
    print("=" * 80)
    print(f"Average localization error : {average_localization_error:.4f} m")
    print(f"Mean Euclidean error       : {xy_stat['mean']:.4f} m")
    print(f"Median                     : {xy_stat['median']:.4f} m")
    print(f"P95                        : {xy_stat['p95']:.4f} m")
    print(f"P99                        : {xy_stat['p99']:.4f} m")
    print(f"Max                        : {xy_stat['max']:.4f} m")
    print(f"RMSE x                     : {rmse_x:.4f} m")
    print(f"RMSE y                     : {rmse_y:.4f} m")
    print(f"2-D RMSE                   : {rmse_xy:.4f} m")


if __name__ == "__main__":
    main()