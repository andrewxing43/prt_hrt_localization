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
THETA_TRUE = -53.5

R_VALUES = np.arange(200.0, 411.0, 20.0)

DQ_VALUES = [1e-11, 5e-12, 7e-13,5e-13, 1e-13]

P_ANGLE_STEP_DEG = PRT.theta_step_deg
P_SEARCH_HALF_BINS = 2
TAU_SEARCH_HALF_SAMPLES = 2

Q_MARGIN_FACTOR = 2.0


def build_time_axis(range_m, theta_deg, antenna_x):
    dt, tau_true = PRT.tau_step, range_m / C
    delays = propagation_delays(range_m, theta_deg, antenna_x=antenna_x)

    margin = 8.0 * SYSTEM.gaussian_sigma
    t_min, t_max = delays.min() - margin, delays.max() + margin

    # Force tau_true halfway between two time samples
    n_before = int(np.ceil((tau_true - t_min) / dt - 0.5))
    t_start = tau_true - (n_before + 0.5) * dt

    n_samples = int(np.ceil((t_max - t_start) / dt)) + 1
    return t_start + np.arange(n_samples) * dt


def build_local_p_grid(theta_true_deg):
    # Force theta_true halfway between two angle bins
    offsets = np.arange(-P_SEARCH_HALF_BINS, P_SEARCH_HALF_BINS) + 0.5
    theta_grid = theta_true_deg + offsets * P_ANGLE_STEP_DEG

    return -np.sin(np.deg2rad(theta_grid)) / C


def get_true_q_limits():
    q_values = []

    for R in R_VALUES:
        _, q = range_theta_to_pq(R, THETA_TRUE)
        q_values.append(float(q))

    return min(q_values), max(q_values)


def build_q_grid(dq):
    q_true_min, q_true_max = get_true_q_limits()

    # Leave margin outside the physical test region
    q_start = q_true_min - Q_MARGIN_FACTOR * dq
    q_stop = q_true_max + Q_MARGIN_FACTOR * dq

    # Half-bin offset so q_true_min is NOT exactly on-grid
    q_start += 0.5 * dq

    q_grid = np.arange(q_start, q_stop + dq, dq)

    return q_grid


def local_peak(prt, time_axis, p_grid, q_grid, tau_true):
    mask = np.abs(time_axis - tau_true) <= TAU_SEARCH_HALF_SAMPLES * PRT.tau_step
    tau_idx_global = np.where(mask)[0]

    local = np.abs(prt[tau_idx_global])
    idx = np.unravel_index(np.argmax(local), local.shape)

    tau_idx = tau_idx_global[idx[0]]
    p_idx = idx[1]
    q_idx = idx[2]

    return time_axis[tau_idx], p_grid[p_idx], q_grid[q_idx]


def generate_signal(range_true, antenna_x):
    time_axis = build_time_axis(range_true, THETA_TRUE, antenna_x)

    rx, _ = received_signal(
        time_axis,
        range_true,
        THETA_TRUE,
        antenna_x=antenna_x
    )

    rng = np.random.default_rng(SIM.rng_seed)

    noisy_rx, _ = add_awgn(
        rx,
        snr_db=SNR_DB,
        rng=rng
    )

    mf_rx = matched_filter(noisy_rx)

    return time_axis, mf_rx


def run_position(range_true, antenna_x, q_grid, time_axis, mf_rx):
    tau_true = range_true / C

    p_true, q_true = range_theta_to_pq(range_true, THETA_TRUE)
    p_true, q_true = float(p_true), float(q_true)

    p_grid = build_local_p_grid(THETA_TRUE)

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
        "R_hat": range_hat,
        "range_error": range_hat - range_true,
        "abs_range_error": abs(range_hat - range_true),

        "theta_hat": theta_hat,
        "theta_error": theta_hat - THETA_TRUE,

        "tau_hat": tau_hat,
        "tau_error": tau_hat - tau_true,

        "p_true": p_true,
        "p_hat": p_hat,

        "q_true": q_true,
        "q_hat": q_hat
    }


def main():
    antenna_x = ARRAY.positions(SYSTEM)

    signals = {}

    print("Generating signals...\n")

    for R in R_VALUES:
        signals[R] = generate_signal(R, antenna_x)

    all_errors = {}

    print(f"Fixed theta = {THETA_TRUE:.2f} deg")
    print(f"Range = {R_VALUES[0]:.0f} ~ {R_VALUES[-1]:.0f} m")
    print(f"Angle step = {P_ANGLE_STEP_DEG} deg")
    print("p: off-grid by 0.5 bin")
    print("tau: off-grid by 0.5 sample")
    print("q-grid: shifted by 0.5 bin to avoid boundary on-grid bias\n")

    for dq in DQ_VALUES:
        q_grid = build_q_grid(dq)
        errors = []

        print("=" * 90)
        print(
            f"Delta q = {dq:.1e} s/m^2 | "
            f"Nq = {len(q_grid)} | "
            f"q_min = {q_grid[0]:.3e} | "
            f"q_max = {q_grid[-1]:.3e}"
        )
        print("=" * 90)

        for R in R_VALUES:
            time_axis, mf_rx = signals[R]

            result = run_position(
                R,
                antenna_x,
                q_grid,
                time_axis,
                mf_rx
            )

            errors.append(result["abs_range_error"])

            q_offset_bins = (result["q_true"] - q_grid[0]) / dq
            q_fraction = q_offset_bins - np.floor(q_offset_bins)

            print(
                f"R={R:6.1f} m | "
                f"R_hat={result['R_hat']:9.3f} m | "
                f"R_err={result['range_error']:+9.3f} m | "
                f"theta_err={result['theta_error']:+7.3f} deg | "
                f"tau_err={result['tau_error'] * 1e12:+8.3f} ps | "
                f"q_true={result['q_true']:.3e} | "
                f"q_hat={result['q_hat']:.3e} | "
                f"q_frac={q_fraction:.3f}"
            )

        all_errors[dq] = np.array(errors)

    # ============================================================
    # Plot
    # ============================================================

    plt.figure(figsize=(10, 6))

    for dq in DQ_VALUES:
        plt.plot(
            R_VALUES,
            all_errors[dq],
            marker="o",
            markersize=4,
            label=f"Δq={dq:.0e}"
        )

    plt.xlabel("True range R (m)")
    plt.ylabel("Absolute range error |R̂ - R| (m)")
    plt.title(
        f"Effect of q Resolution on Range Estimation\n"
        f"θ={THETA_TRUE:.1f}°, p, τ and q-boundary off-grid"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for dq in DQ_VALUES:
        errors = all_errors[dq]

        print(
            f"Delta q={dq:.1e} | "
            f"mean={np.mean(errors):8.3f} m | "
            f"max={np.max(errors):8.3f} m"
        )

    plt.show()


if __name__ == "__main__":
    main()