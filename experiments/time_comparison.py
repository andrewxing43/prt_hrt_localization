"""No-hint comparison: global HRT vs PRT -> ROI -> HRT."""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time import perf_counter
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.hrt import hyperbolic_radon_transform
from localization.roi import build_roi
from localization.pipeline import (
    run_prt, fixed_grid, DP,
    PRT_R_MIN, PRT_R_MAX, PRT_THETA_MIN, PRT_THETA_MAX,
    HRT_DR, HRT_DTHETA, HRT_DTAU
)

N_MC = 1
R_MIN, R_MAX = PRT_R_MIN, PRT_R_MAX
THETA_MIN, THETA_MAX = PRT_THETA_MIN, PRT_THETA_MAX
CLOCK_OFFSET = 10e-9

# Shared HRT memory settings for both methods.
R_BLOCK = 16
THETA_BLOCK = 128
HRT_BATCH_SIZE = 16

METHODS = ("Global HRT", "PRT + ROI + HRT")


def build_time_axis(R, theta, antenna_x):
    delays = propagation_delays(
        R, theta, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x
    )
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def polar_to_xy(R, theta):
    theta = np.deg2rad(theta)
    return R * np.sin(theta), R * np.cos(theta)


def hrt_search(mf_rx, time_axis, antenna_x, r_bounds, th_bounds, tau_bounds):
    r_grid = fixed_grid(*r_bounds, HRT_DR)
    th_grid = fixed_grid(*th_bounds, HRT_DTHETA)
    tau_grid = fixed_grid(*tau_bounds, HRT_DTAU)

    best_peak = -np.inf
    best_index = None
    best_order = None

    is_global = (
        tuple(r_bounds) == (R_MIN, R_MAX)
        and tuple(th_bounds) == (THETA_MIN, THETA_MAX)
        and tuple(tau_bounds)
        == (float(time_axis[0]), float(time_axis[-1]))
    )
    total_blocks = (
        (len(r_grid) + R_BLOCK - 1) // R_BLOCK
        * ((len(th_grid) + THETA_BLOCK - 1) // THETA_BLOCK)
    )

    with trange(
        total_blocks, desc="Global HRT", unit="block",
        position=1, leave=False, disable=not is_global
    ) as progress:
        for r0 in range(0, len(r_grid), R_BLOCK):
            ranges = r_grid[r0:r0 + R_BLOCK]
            for th0 in range(0, len(th_grid), THETA_BLOCK):
                angles = th_grid[th0:th0 + THETA_BLOCK]
                hrt = hyperbolic_radon_transform(
                    mf_rx, time_axis, antenna_x,
                    ranges, angles, tau_grid,
                    batch_size=HRT_BATCH_SIZE
                )
                score = np.abs(hrt)
                it, ir, ith = np.unravel_index(
                    np.argmax(score), score.shape
                )
                peak = float(score[it, ir, ith])
                index = (int(it), r0 + int(ir), th0 + int(ith))
                order = (
                    (index[0] * len(r_grid) + index[1]) * len(th_grid)
                    + index[2]
                )
                if peak > best_peak or (
                    peak == best_peak
                    and (best_order is None or order < best_order)
                ):
                    best_peak = peak
                    best_index = index
                    best_order = order

                del hrt, score
                progress.update(1)

    it, ir, ith = best_index
    return (
        float(r_grid[ir]), float(th_grid[ith]),
        float(tau_grid[it]), best_peak
    )


def global_hrt(rx, time_axis, antenna_x):
    mf_rx = matched_filter(rx)
    R, theta, tau, peak = hrt_search(
        mf_rx, time_axis, antenna_x,
        (R_MIN, R_MAX), (THETA_MIN, THETA_MAX),
        (float(time_axis[0]), float(time_axis[-1]))
    )
    return polar_to_xy(R, theta)


def prt_roi_hrt(rx, time_axis, antenna_x):
    mf_rx = matched_filter(rx)

    # No theta_hint or tau_hint: global PRT search.
    R_prt, theta_prt, tau_prt, peak = run_prt(
        mf_rx, time_axis, antenna_x
    )
    roi = build_roi(
        R_prt, theta_prt, tau_prt, dp=DP,
        range_bounds=(R_MIN, R_MAX),
        theta_bounds=(THETA_MIN, THETA_MAX),
        tau_bounds=(float(time_axis[0]), float(time_axis[-1]))
    )
    R, theta, tau, peak = hrt_search(
        mf_rx, time_axis, antenna_x,
        roi.range_bounds, roi.theta_bounds, roi.tau_bounds
    )

    covered = (
        roi.R_min <= TRUE_R <= roi.R_max
        and roi.theta_min <= TRUE_THETA <= roi.theta_max
        and roi.tau_min <= TRUE_TAU <= roi.tau_max
    )
    return polar_to_xy(R, theta), covered


def timed_call(func, *args):
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    result = func(*args)
    cp.cuda.runtime.deviceSynchronize()
    return result, perf_counter() - start


def warm_up(antenna_x):
    # Small workload initializes CUDA/FFT/interpolation paths.
    R, theta = 200.0, 0.0
    time_axis = build_time_axis(R, theta, antenna_x)
    rx, _ = received_signal(
        time_axis, R, theta,
        clock_offset=CLOCK_OFFSET, antenna_x=antenna_x
    )
    mf_rx = matched_filter(rx)
    run_prt(mf_rx, time_axis, antenna_x)
    tau = R / C + CLOCK_OFFSET
    hyperbolic_radon_transform(
        mf_rx, time_axis, antenna_x,
        np.array([R]), np.array([theta]), np.array([tau]),
        batch_size=HRT_BATCH_SIZE
    )
    cp.cuda.runtime.deviceSynchronize()


def print_results(name, errors, elapsed, snrs):
    ALE = np.mean(errors)
    RMSE = np.sqrt(np.mean(errors**2))
    print("\n" + "=" * 50)
    print(name)
    print(f"N        : {N_MC}")
    print(f"Tx power : {SIM.tx_power_dbm:.3f} dBm")
    print(f"SNR range: {snrs.min():.3f} to {snrs.max():.3f} dB")
    print(f"ALE      : {ALE:.6f} m")
    print(f"RMSE     : {RMSE:.6f} m")
    print(f"Max error: {errors.max():.6f} m")
    print(f"Total time : {elapsed.sum():.6f} s")
    print(f"Mean time  : {elapsed.mean() * 1e3:.6f} ms")
    print(f"Median time: {np.median(elapsed) * 1e3:.6f} ms")
    print("=" * 50)


def plot_results(name, errors, x_true, y_true):
    ALE = np.mean(errors)
    RMSE = np.sqrt(np.mean(errors**2))

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.sort(errors), np.arange(1, N_MC + 1) / N_MC,
        linewidth=2
    )
    plt.xlabel("2-D Localization Error (m)")
    plt.ylabel("CDF")
    plt.title(f"Localization Error Distribution — {name}")
    plt.grid(True)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    sc = ax.scatter(
        x_true, y_true, c=errors, s=42,
        cmap="viridis", edgecolors="none"
    )
    xmax = R_MAX * np.sin(
        np.deg2rad(max(abs(THETA_MIN), abs(THETA_MAX)))
    )
    ax.set_xlim(-xmax - 25, xmax + 25)
    ax.set_ylim(0, R_MAX + 25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"True Positions Colored by Localization Error\n{name}")
    ax.grid(True, alpha=0.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    fig.colorbar(sc, cax=cax).set_label("Absolute Error (m)")
    ax.text(
        0.02, 0.98,
        f"N={N_MC}\nALE={ALE:.3f} m\nRMSE={RMSE:.3f} m",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(
            boxstyle="round", facecolor="white",
            alpha=0.9, edgecolor="gray"
        )
    )
    plt.tight_layout()


def main():
    global TRUE_R, TRUE_THETA, TRUE_TAU

    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)
    errors = np.zeros((2, N_MC))
    elapsed = np.zeros((2, N_MC))
    coverage = np.zeros(N_MC, dtype=bool)
    snrs = np.zeros(N_MC)
    x_true_all, y_true_all = np.zeros(N_MC), np.zeros(N_MC)

    print("Warming up GPU...")
    warm_up(antenna_x)

    for i in trange(N_MC, desc="Monte Carlo"):
        TRUE_R = rng_pos.uniform(R_MIN, R_MAX)
        TRUE_THETA = rng_pos.uniform(THETA_MIN, THETA_MAX)
        TRUE_TAU = TRUE_R / C + CLOCK_OFFSET
        x_true, y_true = polar_to_xy(TRUE_R, TRUE_THETA)

        time_axis = build_time_axis(TRUE_R, TRUE_THETA, antenna_x)
        clean_rx, _ = received_signal(
            time_axis, TRUE_R, TRUE_THETA,
            clock_offset=CLOCK_OFFSET, antenna_x=antenna_x
        )
        noisy_rx, _ = add_awgn(clean_rx, rng=rng_noise)
        args = (noisy_rx, time_axis, antenna_x)

        order = (0, 1) if i % 2 == 0 else (1, 0)
        for k in order:
            if k == 0:
                xy, elapsed[k, i] = timed_call(global_hrt, *args)
            else:
                result, elapsed[k, i] = timed_call(prt_roi_hrt, *args)
                xy, coverage[i] = result
            errors[k, i] = np.hypot(
                xy[0] - x_true, xy[1] - y_true
            )

        x_true_all[i], y_true_all[i] = x_true, y_true
        snrs[i] = (
            SIM.reference_snr_db
            + 20.0 * np.log10(SIM.reference_range / TRUE_R)
        )

    for k, name in enumerate(METHODS):
        print_results(name, errors[k], elapsed[k], snrs)
        plot_results(name, errors[k], x_true_all, y_true_all)

    print(f"\nROI coverage: {100.0 * coverage.mean():.3f}%")
    print(f"ROI misses  : {np.count_nonzero(~coverage)}")
    ratio = elapsed[0].sum() / elapsed[1].sum()
    print(f"Global HRT / Pipeline time ratio: {ratio:.6f}")
    print("Ratio > 1: pipeline faster; ratio < 1: global HRT faster.")
    plt.show()


if __name__ == "__main__":
    main()