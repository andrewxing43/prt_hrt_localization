"""
Monte Carlo study of HRT tau-grid resolution.

Purpose:
- Fix one off-grid user position.
- Generate a new noise realization for each Monte Carlo trial.
- For the same noisy signal, run HRT with several tau-grid resolutions.
- Compare range-estimation statistics versus tau resolution.

This experiment does NOT use PRT, ROI, or pipeline.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.hrt import hyperbolic_radon_transform, hrt_peak


# ============================================================
# Experiment settings
# ============================================================

R_TRUE = 200.37
THETA_TRUE = 30.03
CLOCK_OFFSET = 10e-9

N_MC = 100

R_HALF_WIDTH = 20.0
HRT_DR = 0.1

THETA_HALF_WIDTH = 0.2
HRT_DTHETA = PRT.theta_step_deg

TAU_HALF_SAMPLES = 2
TAU_DIVISORS = [1, 2, 4, 8, 16]

R_GRID_OFFSET = 0.02
THETA_GRID_OFFSET = 0.02


# ============================================================
# Helpers
# ============================================================

def build_time_axis(R_true, theta_true, clock_offset, antenna_x):
    delays = propagation_delays(R_true, theta_true, clock_offset=clock_offset, antenna_x=antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def build_grid(start, stop, step):
    n = int(np.floor((stop - start) / step))
    grid = start + np.arange(n + 1) * step
    if grid[-1] < stop - 1e-15:
        grid = np.append(grid, stop)
    return grid


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
        "max": np.max(values),
    }


# ============================================================
# Main
# ============================================================

def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng = np.random.default_rng(SIM.rng_seed)

    tau_true = R_TRUE / C + CLOCK_OFFSET
    snr_db = SIM.reference_snr_db + 20.0 * np.log10(SIM.reference_range / R_TRUE)

    time_axis = build_time_axis(R_TRUE, THETA_TRUE, CLOCK_OFFSET, antenna_x)
    clean_rx, _ = received_signal(time_axis, R_TRUE, THETA_TRUE, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x)

    range_grid = build_grid(R_TRUE - R_HALF_WIDTH + R_GRID_OFFSET,
                            R_TRUE + R_HALF_WIDTH + R_GRID_OFFSET, HRT_DR)

    theta_grid = build_grid(THETA_TRUE - THETA_HALF_WIDTH + THETA_GRID_OFFSET,
                            THETA_TRUE + THETA_HALF_WIDTH + THETA_GRID_OFFSET, HRT_DTHETA)

    tau_grids = {}
    for divisor in TAU_DIVISORS:
        dtau = SYSTEM.dt / divisor
        tau_offset = 0.35 * dtau
        tau_half_width = TAU_HALF_SAMPLES * SYSTEM.dt
        tau_grids[divisor] = build_grid(tau_true - tau_half_width + tau_offset,
                                        tau_true + tau_half_width + tau_offset, dtau)

    results = {
        divisor: {"R_error": [], "theta_error": [], "tau_error": [], "R_hat": []}
        for divisor in TAU_DIVISORS
    }

    print("=" * 90)
    print("HRT TAU-RESOLUTION MONTE CARLO")
    print("=" * 90)
    print(f"R_true          : {R_TRUE:.6f} m")
    print(f"theta_true      : {THETA_TRUE:.6f} deg")
    print(f"tau_true        : {tau_true * 1e9:.6f} ns")
    print(f"SNR             : {snr_db:.1f} dB")
    print(f"Tx power        : {SIM.tx_power_dbm:.3f} dBm")
    print(f"Thermal noise   : {10.0 * np.log10(SIM.noise_power / 1e-3):.3f} dBm")
    print(f"Monte Carlo     : {N_MC}")
    print(f"R step          : {HRT_DR:.3f} m")
    print(f"theta step      : {HRT_DTHETA:.3f} deg")
    print(f"N_R             : {len(range_grid)}")
    print(f"N_theta         : {len(theta_grid)}")
    print()

    progress = tqdm(range(N_MC), desc="Monte Carlo", unit="trial")

    for _ in progress:
        noisy_rx, _ = add_awgn(clean_rx, rng=rng)
        mf_rx = matched_filter(noisy_rx)

        for divisor in TAU_DIVISORS:
            tau_grid = tau_grids[divisor]

            hrt = hyperbolic_radon_transform(
                mf_rx, time_axis, antenna_x, range_grid, theta_grid, tau_grid
            )

            tau_hat, R_hat, theta_hat, _, _ = hrt_peak(
                hrt, tau_grid, range_grid, theta_grid
            )

            results[divisor]["R_hat"].append(R_hat)
            results[divisor]["R_error"].append(abs(R_hat - R_TRUE))
            results[divisor]["theta_error"].append(abs(theta_hat - THETA_TRUE))
            results[divisor]["tau_error"].append(abs(tau_hat - tau_true))

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    header = (
        f"{'tau step':>12} {'ps':>10} {'N_tau':>8} "
        f"{'mean R err':>14} {'median':>12} {'P95':>12} "
        f"{'P99':>12} {'max':>12}"
    )
    print(header)

    stats = {}

    for divisor in TAU_DIVISORS:
        dtau = SYSTEM.dt / divisor
        R_stat = summarize(results[divisor]["R_error"])
        stats[divisor] = R_stat

        print(
            f"{'dt/' + str(divisor):>12} "
            f"{dtau * 1e12:10.4f} "
            f"{len(tau_grids[divisor]):8d} "
            f"{R_stat['mean']:14.4f} "
            f"{R_stat['median']:12.4f} "
            f"{R_stat['p95']:12.4f} "
            f"{R_stat['p99']:12.4f} "
            f"{R_stat['max']:12.4f}"
        )

    print("\nTau error statistics")
    print("-" * 90)

    for divisor in TAU_DIVISORS:
        dtau = SYSTEM.dt / divisor
        tau_err_ps = np.array(results[divisor]["tau_error"]) * 1e12
        tau_stat = summarize(tau_err_ps)

        print(
            f"dt/{divisor:<2d} ({dtau * 1e12:7.4f} ps) | "
            f"mean={tau_stat['mean']:7.4f} ps | "
            f"P95={tau_stat['p95']:7.4f} ps | "
            f"P99={tau_stat['p99']:7.4f} ps"
        )

    dtau_ps = np.array([SYSTEM.dt / d for d in TAU_DIVISORS]) * 1e12
    mean_R = np.array([stats[d]["mean"] for d in TAU_DIVISORS])
    p95_R = np.array([stats[d]["p95"] for d in TAU_DIVISORS])
    p99_R = np.array([stats[d]["p99"] for d in TAU_DIVISORS])

    plt.figure(figsize=(8, 5))
    plt.plot(dtau_ps, mean_R, marker="o", label="Mean")
    plt.plot(dtau_ps, p95_R, marker="o", label="P95")
    plt.plot(dtau_ps, p99_R, marker="o", label="P99")
    plt.xlabel("Tau grid step (ps)")
    plt.ylabel("Absolute range error (m)")
    plt.title("HRT Range Error versus Tau Resolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    for divisor in TAU_DIVISORS:
        plt.hist(results[divisor]["R_error"], bins=20, alpha=0.5,
                 label=f"dt/{divisor} = {SYSTEM.dt / divisor * 1e12:.3f} ps")

    plt.xlabel("Absolute range error (m)")
    plt.ylabel("Count")
    plt.title("HRT Range Error Distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()