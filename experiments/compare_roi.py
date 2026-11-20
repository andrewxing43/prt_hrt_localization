"""Compare both ROIs using a hint-assisted fast PRT and the same fast HRT with same post hoc oversamp"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from localization.pipeline import Q_GRID, fixed_grid, nearest_indices, N_LOCAL_P, N_LOCAL_TAU, HRT_DR, HRT_DTHETA, PRT_R_MIN, PRT_R_MAX, PRT_THETA_MIN, PRT_THETA_MAX
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import build_roi
from localization.analytical_roi import get_np, get_error_margins, get_roi
from transforms.fast_prt import run_parabolic_radon_transform
from transforms.fast_hrt import run_hyperbolic_radon_transform, resample_hrt

N_MC = 250
R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
SNR_DB = 10.0
CLOCK_OFFSET = 10e-9
K_INTERP = 4
HRT_RANGE_BATCH = 32
RESULTS_PATH = ROOT / "results" / "compare_roi_hinted.csv"

def build_time_axis(R, theta, clock_offset, antenna_x):
    delays = propagation_delays(R, theta, clock_offset=clock_offset, antenna_x=antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt

def polar_to_xy(R, theta_deg):
    t = np.deg2rad(theta_deg)
    return R * np.sin(t), R * np.cos(t)

# run and find PRT peak without hints
def shared_prt_peak(mf_rx, time_axis, antenna_x, px, qx, f0):
    # run PRT
    prt = run_parabolic_radon_transform(mf_rx, antenna_x, SYSTEM.dt, f0, px, qx, True)

    # map PQ back to physical coordinates
    P, Q = np.meshgrid(px, qx, indexing="ij")
    ranges = pq_to_range(P, Q)
    angles = p_to_theta_deg(P)

    score = np.abs(prt)**2
    idx = np.unravel_index(np.argmax(score), score.shape)

    return float(px[idx[1]]), float(qx[idx[2]]), float(time_axis[idx[0]])

# apply hints and find PRT peak using fast transform
def shared_prt_peak_hints(mf_rx, time_axis, antenna_x, px, qx, f0, theta_hint, tau_hint):

    # construct the local p grid using theta hint
    p_hint = -np.sin(np.deg2rad(theta_hint)) / C
    p_grid = px[nearest_indices(px, p_hint, N_LOCAL_P)]

    # construct tau search spaace using time hint
    tau_idx = nearest_indices(time_axis, tau_hint, N_LOCAL_TAU)

    # run PRT
    prt = run_parabolic_radon_transform(mf_rx, antenna_x, SYSTEM.dt, f0, p_grid, qx, True)

    # map all P/Q back to physical coordinates
    P, Q = np.meshgrid(p_grid, qx, indexing="ij")
    ranges = pq_to_range(P, Q)
    angles = p_to_theta_deg(P)

    score = np.abs(prt[tau_idx])**2
    idx = np.unravel_index(np.argmax(score), score.shape)

    return float(p_grid[idx[1]]), float(qx[idx[2]]), float(time_axis[tau_idx[idx[0]]])

# find HRT peak using fast transform using either ROI object
def run_shared_hrt(mf_rx, time_axis, antenna_x, f0, roi):

    # create range and theta grids from ROI
    range_vals = fixed_grid(roi.R_min, roi.R_max, HRT_DR)
    theta_vals = np.deg2rad(fixed_grid(roi.theta_min, roi.theta_max, HRT_DTHETA))
    tau_vals = time_axis[0] + np.arange((len(time_axis) - 1) * K_INTERP + 1) * (SYSTEM.dt / K_INTERP)
    tau_idx = np.flatnonzero((tau_vals >= roi.tau_min) & (tau_vals <= roi.tau_max))

    best_power = -np.inf
    best = None

    # chunk ranges and run, find the peak over all chunks
    for start in range(0, len(range_vals), HRT_RANGE_BATCH):
        ranges = range_vals[start:start + HRT_RANGE_BATCH]

        # run HRT
        hrt = run_hyperbolic_radon_transform(mf_rx, antenna_x, SYSTEM.dt, f0, ranges, theta_vals, True)

        # resample 4x
        hrt_fine = resample_hrt(hrt, K_INTERP)

        # keep this peak in this range batch if its the best, discard otherwise
        score = np.abs(hrt_fine[tau_idx])**2
        idx = np.unravel_index(np.argmax(score), score.shape)
        if score[idx] > best_power:
            best_power = float(score[idx])
            best = (float(ranges[idx[1]]), float(np.rad2deg(theta_vals[idx[2]])), float(tau_vals[tau_idx[idx[0]]]))
    return best, (len(range_vals), len(theta_vals), len(tau_idx))

def roi_coverage(roi, R, theta, tau):
    return (roi.R_min <= R <= roi.R_max, roi.theta_min <= theta <= roi.theta_max, roi.tau_min <= tau <= roi.tau_max)

def main():
    antenna_x = ARRAY.positions(SYSTEM)
    L = np.max(antenna_x) - np.min(antenna_x)
    f0 = SYSTEM.fc
    B = SYSTEM.bandwidth
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    Np = get_np(L, f0, B)
    E_q_cont, dp, dt = get_error_margins(L, f0, B, Np, oversampling=SYSTEM.oversamp)

    px = np.linspace(-1.0 / C, 1.0 / C, Np)
    qx = np.asarray(Q_GRID)
    dq = float(qx[1] - qx[0])
    E_q = E_q_cont + dq / 2.0
    print(f"Shared grid: Np={Np}, Nq={len(qx)}, hinted p bins={N_LOCAL_P}, hinted tau bins={N_LOCAL_TAU}, dq={dq:.6e}")
    print(f"Analytical E_q={E_q:.6e} s/m^2 including dq/2; HRT interpolation={K_INTERP}x")

    # save results
    names = ("Original ROI", "Analytical ROI")
    rows = []
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trial", "method", "R_true", "theta_true", "tau_true", "p_prt", "q_prt", "tau_prt", "R_hat", "theta_hat", "tau_hat", "position_error", "range_error", "range_covered", "theta_covered", "tau_covered", "joint_covered", "R_min", "R_max", "theta_min", "theta_max", "tau_min", "tau_max", "range_width", "theta_width", "tau_width", "Nr", "Ntheta", "Ntau", "search_nodes", "shared_seconds", "roi_hrt_seconds", "total_seconds", "status"]

    with RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in trange(N_MC, desc="Paired ROI comparison"):

            # generate random source positions
            R_true = rng_pos.uniform(R_MIN, R_MAX)
            theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
            tau_true = R_true / C + CLOCK_OFFSET

            x_true, y_true = polar_to_xy(R_true, theta_true)
            time_axis = build_time_axis(R_true, theta_true, CLOCK_OFFSET, antenna_x)
            clean_rx, _ = received_signal(time_axis, R_true, theta_true, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x)
            noisy_rx, _ = add_awgn(clean_rx, snr_db=SNR_DB, rng=rng_noise)

            start = time.perf_counter()

            # matched filter
            mf_rx = matched_filter(noisy_rx)

            # run PRT
            p_peak, q_peak, tau_peak = shared_prt_peak_hints(mf_rx, time_axis, antenna_x, px, qx, f0, theta_true, tau_true)
            shared_seconds = time.perf_counter() - start
            R_peak = float(pq_to_range(p_peak, q_peak))
            theta_peak = float(p_to_theta_deg(p_peak))

            # build both ROIs
            for method in ((0, 1) if trial % 2 == 0 else (1, 0)):
                row = dict.fromkeys(fields, np.nan)
                row.update(trial=trial, method=names[method], R_true=R_true, theta_true=theta_true, tau_true=tau_true, p_prt=p_peak, q_prt=q_peak, tau_prt=tau_peak, shared_seconds=shared_seconds, range_covered=False, theta_covered=False, tau_covered=False, joint_covered=False, position_error=np.inf, range_error=np.inf, status="ok")
                start = time.perf_counter()
                try:
                    if method == 0:
                        roi = build_roi(R_peak, theta_peak, tau_peak, dq=dq, range_bounds=(PRT_R_MIN, PRT_R_MAX), theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX))
                    else:
                        roi = get_roi(p_peak, q_peak, tau_peak, dp, E_q, dt, range_bounds=(PRT_R_MIN, PRT_R_MAX), theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX))
                    cover = roi_coverage(roi, R_true, theta_true, tau_true)
                    row.update(range_covered=cover[0], theta_covered=cover[1], tau_covered=cover[2], joint_covered=all(cover), R_min=roi.R_min, R_max=roi.R_max, theta_min=roi.theta_min, theta_max=roi.theta_max, tau_min=roi.tau_min, tau_max=roi.tau_max, range_width=roi.R_max - roi.R_min, theta_width=roi.theta_max - roi.theta_min, tau_width=roi.tau_max - roi.tau_min)

                    # run HRT on each ROI and save info
                    (R_hat, theta_hat, tau_hat), shape = run_shared_hrt(mf_rx, time_axis, antenna_x, f0, roi)
                    x_hat, y_hat = polar_to_xy(R_hat, theta_hat)
                    row.update(R_hat=R_hat, theta_hat=theta_hat, tau_hat=tau_hat, position_error=float(np.hypot(x_hat - x_true, y_hat - y_true)), range_error=abs(R_hat - R_true), Nr=shape[0], Ntheta=shape[1], Ntau=shape[2], search_nodes=int(np.prod(shape)))
                except ValueError as error:
                    row["status"] = str(error)
                row["roi_hrt_seconds"] = time.perf_counter() - start
                row["total_seconds"] = shared_seconds + row["roi_hrt_seconds"]
                rows.append(row)
                writer.writerow(row)
            handle.flush()

    plt.figure(figsize=(8, 5))
    for name in names:
        selected = [row for row in rows if row["method"] == name]
        errors = np.asarray([row["position_error"] for row in selected])
        covered = np.asarray([row["joint_covered"] for row in selected], dtype=bool)
        valid = np.isfinite(errors)
        print(f"\n{name}: completed={np.sum(valid)}/{N_MC}, joint ROI coverage={np.mean(covered):.1%}")
        print(f"ALE={np.mean(errors):.4f} m, RMSE={np.sqrt(np.mean(errors**2)):.4f} m, P95={np.percentile(errors, 95) if np.all(valid) else 'see CSV; failed trials retained'}")
        print(f"Coverage R/theta/tau: {np.mean([row['range_covered'] for row in selected]):.1%} / {np.mean([row['theta_covered'] for row in selected]):.1%} / {np.mean([row['tau_covered'] for row in selected]):.1%}")
        print(f"Average total processing time={np.mean([row['total_seconds'] for row in selected]):.4f} s/run")
        if np.any(covered & valid):
            print(f"ALE when truth is inside ROI={np.mean(errors[covered & valid]):.4f} m")
        plt.plot(np.sort(errors[valid]), np.arange(1, np.sum(valid) + 1) / N_MC, linewidth=2, label=name)
    plt.xlabel("2-D Localization Error (m)")
    plt.ylabel("Fraction of all trials")
    plt.title("ROI Comparison: Shared Hinted PRT and Fast HRT")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5.5))
    bins = np.linspace(R_MIN, R_MAX, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for name in names:
        selected = [row for row in rows if row["method"] == name and np.isfinite(row["range_error"])]
        ranges = np.asarray([row["R_true"] for row in selected])
        errors = np.asarray([row["range_error"] for row in selected])
        plt.scatter(ranges, errors, alpha=0.2, s=15, label=f"{name} trials")
        means = np.full(10, np.nan)
        for b in range(10):
            mask = (ranges >= bins[b]) & (ranges < bins[b + 1])
            if np.any(mask): means[b] = np.mean(errors[mask])
        plt.plot(centers, means, marker="o", linewidth=2, label=f"{name} mean")
    plt.xlabel("True range (m)")
    plt.ylabel("Absolute range error (m)")
    plt.title("Range Error: Completed Trials")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(f"Per-trial results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
