"""Paired SNR sweep with shared hinted PRT, identical HRT, coverage metrics, and position error maps."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import argparse
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from localization.pipeline import Q_GRID, fixed_grid, nearest_indices, find_hrt_peak, polar_to_xy, N_LOCAL_P, N_LOCAL_TAU, HRT_DR, HRT_DTHETA, PRT_R_MIN, PRT_R_MAX, PRT_THETA_MIN, PRT_THETA_MAX
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import build_roi
from localization.analytical_roi import get_np, get_error_margins, get_roi
from transforms.fast_prt import run_parabolic_radon_transform
from transforms.fast_hrt import run_hyperbolic_radon_transform, resample_hrt

N_MC = 250
R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
SNR_VALUES = (0.0, 5.0, 10.0, 15.0, 20.0, np.inf)
CLOCK_OFFSET = 10e-9
K_INTERP = 4
HRT_RANGE_BATCH = 32
OUTPUT_DIR = ROOT / "results" / "roi_snr"
METHODS = ("Original ROI", "Analytical ROI")

def build_time_axis(R, theta, clock_offset, antenna_x):
    delays = propagation_delays(R, theta, clock_offset=clock_offset, antenna_x=antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt

# apply hints and find PRT peak using fast transform
def shared_prt_peak(mf_rx, time_axis, antenna_x, px, f0, theta_hint, tau_hint):
    # construct the local p grid using theta hint
    p_hint = -np.sin(np.deg2rad(theta_hint)) / C
    p_grid = px[nearest_indices(px, p_hint, N_LOCAL_P)]

    # construct tau search spaace using time hint
    tau_idx = nearest_indices(time_axis, tau_hint, N_LOCAL_TAU)

    # run PRT
    prt = run_parabolic_radon_transform(mf_rx, antenna_x, SYSTEM.dt, f0, p_grid, Q_GRID, True)
    score = np.abs(prt[tau_idx])**2
    idx = np.unravel_index(np.argmax(score), score.shape)
    return float(p_grid[idx[1]]), float(Q_GRID[idx[2]]), float(time_axis[tau_idx[idx[0]]])

# find HRT peak using fast transform using either ROI object
def run_shared_hrt(mf_rx, time_axis, antenna_x, f0, roi):
    # create range and theta grids from ROI
    range_vals = fixed_grid(roi.R_min, roi.R_max, HRT_DR)
    theta_grid = fixed_grid(roi.theta_min, roi.theta_max, HRT_DTHETA)
    theta_vals = np.deg2rad(theta_grid)

    # set up tau grid post interp
    tau_vals = time_axis[0] + np.arange((len(time_axis) - 1) * K_INTERP + 1) * (SYSTEM.dt / K_INTERP)
    tau_idx = np.flatnonzero((tau_vals >= roi.tau_min) & (tau_vals <= roi.tau_max))

    best_peak = -np.inf
    best = None

    # chunk ranges and run, find the peak over all chunks
    for start in range(0, len(range_vals), HRT_RANGE_BATCH):
        ranges = range_vals[start:start + HRT_RANGE_BATCH]

        # run HRT
        hrt = run_hyperbolic_radon_transform(mf_rx, antenna_x, SYSTEM.dt, f0, ranges, theta_vals, True)

        # resample 4x
        hrt_fine = resample_hrt(hrt, K_INTERP)

        # keep this peak in this range batch if its the best, discard otherwise
        R_hat, theta_hat, tau_hat, peak = find_hrt_peak(hrt_fine[tau_idx], tau_vals[tau_idx], ranges, theta_grid)
        if peak > best_peak:
            best_peak = peak
            best = (R_hat, theta_hat, tau_hat)
    return best, (len(range_vals), len(theta_vals), len(tau_idx))

def roi_coverage(roi, R, theta, tau):
    return (roi.R_min <= R <= roi.R_max, roi.theta_min <= theta <= roi.theta_max, roi.tau_min <= tau <= roi.tau_max)

def run_snr(snr_db, calibration):
    antenna_x = ARRAY.positions(SYSTEM)
    L = np.max(antenna_x) - np.min(antenna_x)
    f0 = SYSTEM.fc
    B = SYSTEM.bandwidth

    # reset seed per SNR to reuse positions and scaled versions of the same noise
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    Np, E_q_cont, dp, dt = calibration
    px = np.linspace(-1.0 / C, 1.0 / C, Np)
    qx = np.asarray(Q_GRID)
    dq = float(qx[1] - qx[0])
    E_q = E_q_cont + dq / 2.0
    print(f"Shared grid: Np={Np}, Nq={len(qx)}, hinted p bins={N_LOCAL_P}, hinted tau bins={N_LOCAL_TAU}, dq={dq:.6e}")
    print(f"Analytical E_q={E_q:.6e} s/m^2 including dq/2; HRT interpolation={K_INTERP}x")

    # save results
    names = METHODS
    rows = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"trials_{snr_tag(snr_db)}.csv"
    fields = ["snr_db", "Np", "Nq", "dp", "dq", "dt", "E_q", "p_true", "q_true", "p_error", "q_error", "tau_error", "p_bound_exceeded", "q_bound_exceeded", "tau_bound_exceeded", "spatial_nodes", "trial", "method", "R_true", "theta_true", "tau_true", "p_prt", "q_prt", "tau_prt", "R_hat", "theta_hat", "tau_hat", "position_error", "range_error", "range_covered", "theta_covered", "tau_covered", "joint_covered", "R_min", "R_max", "theta_min", "theta_max", "tau_min", "tau_max", "range_width", "theta_width", "tau_width", "Nr", "Ntheta", "Ntau", "search_nodes", "shared_seconds", "roi_hrt_seconds", "total_seconds", "status"]

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in trange(N_MC, desc=f"SNR {snr_tag(snr_db)}"):

            # generate random source positions
            R_true = rng_pos.uniform(R_MIN, R_MAX)
            theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
            tau_true = R_true / C + CLOCK_OFFSET
            x_true, y_true = polar_to_xy(R_true, theta_true)
            time_axis = build_time_axis(R_true, theta_true, CLOCK_OFFSET, antenna_x)
            clean_rx, _ = received_signal(time_axis, R_true, theta_true, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x)

            # use the clean signal for noiseless baseline, add noise otherwise
            if np.isinf(snr_db):
                noisy_rx = clean_rx
            else:
                noisy_rx, _ = add_awgn(clean_rx, snr_db=snr_db, rng=rng_noise)

            start = time.perf_counter()

            # matched filter
            mf_rx = matched_filter(noisy_rx)

            # run PRT
            p_peak, q_peak, tau_peak = shared_prt_peak(mf_rx, time_axis, antenna_x, px, f0, theta_true, tau_true)
            shared_seconds = time.perf_counter() - start
            R_peak = float(pq_to_range(p_peak, q_peak))
            theta_peak = float(p_to_theta_deg(p_peak))

            # record which PRT error margins are exceeded
            p_true = -np.sin(np.deg2rad(theta_true)) / C
            q_true = np.cos(np.deg2rad(theta_true))**2 / (2.0 * C * R_true)
            p_error, q_error, tau_error = abs(p_peak - p_true), abs(q_peak - q_true), abs(tau_peak - tau_true)

            # build both ROIs
            for method in ((0, 1) if trial % 2 == 0 else (1, 0)):
                row = dict.fromkeys(fields, np.nan)
                row.update(snr_db=snr_db, Np=Np, Nq=len(qx), dp=dp, dq=dq, dt=dt, E_q=E_q, p_true=p_true, q_true=q_true, p_error=p_error, q_error=q_error, tau_error=tau_error, p_bound_exceeded=p_error > dp / 2.0, q_bound_exceeded=q_error > E_q, tau_bound_exceeded=tau_error > dt / 2.0, trial=trial, method=names[method], R_true=R_true, theta_true=theta_true, tau_true=tau_true, p_prt=p_peak, q_prt=q_peak, tau_prt=tau_peak, shared_seconds=shared_seconds, range_covered=False, theta_covered=False, tau_covered=False, joint_covered=False, position_error=np.inf, range_error=np.inf, status="ok")
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
                    row.update(R_hat=R_hat, theta_hat=theta_hat, tau_hat=tau_hat, position_error=float(np.hypot(x_hat - x_true, y_hat - y_true)), range_error=abs(R_hat - R_true), Nr=shape[0], Ntheta=shape[1], Ntau=shape[2], search_nodes=int(np.prod(shape)), spatial_nodes=shape[0] * shape[1])
                except ValueError as error:
                    row["status"] = str(error)
                row["roi_hrt_seconds"] = time.perf_counter() - start
                row["total_seconds"] = shared_seconds + row["roi_hrt_seconds"]
                rows.append(row)
                writer.writerow(row)
            handle.flush()

    print(f"Saved {results_path}")
    return rows

# for replotting
def snr_tag(snr_db):
    return "noiseless" if np.isinf(snr_db) else f"{snr_db:g}dB"

def read_trials(paths):
    rows = []
    text_fields = {"method", "status"}
    bool_fields = {"range_covered", "theta_covered", "tau_covered", "joint_covered", "p_bound_exceeded", "q_bound_exceeded", "tau_bound_exceeded"}
    for path in paths:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {key: value if key in text_fields else value == "True" if key in bool_fields else float(value) for key, value in raw.items()}
                rows.append(row)
    return rows

def summarize(rows):
    summary = []
    for snr_db in sorted({row["snr_db"] for row in rows}):
        for name in METHODS:
            selected = [row for row in rows if row["snr_db"] == snr_db and row["method"] == name]
            if not selected:
                continue
            errors = np.asarray([row["position_error"] for row in selected])
            finite = np.isfinite(errors)
            ordered = np.sort(errors)

            # compare errors on the same trials for both ROIs
            paired = []
            for row in selected:
                other = next((r for r in rows if r["snr_db"] == snr_db and r["trial"] == row["trial"] and r["method"] != name), None)
                if other is not None and row["joint_covered"] and other["joint_covered"] and np.isfinite(row["position_error"]) and np.isfinite(other["position_error"]):
                    paired.append(row["position_error"])
            entry = dict(snr_db=snr_db, method=name, trials=len(selected), completed=int(np.sum(finite)), ALE=float(np.mean(errors)), RMSE=float(np.sqrt(np.mean(errors**2))), P95=float(ordered[int(np.ceil(0.95 * len(ordered))) - 1]), paired_inside_ALE=float(np.mean(paired)) if paired else np.nan, paired_inside_count=len(paired))
            for key in ("range_covered", "theta_covered", "tau_covered", "joint_covered", "p_bound_exceeded", "q_bound_exceeded", "tau_bound_exceeded", "shared_seconds", "roi_hrt_seconds", "total_seconds"):
                entry[key] = float(np.mean([row[key] for row in selected]))
            for key in ("range_width", "theta_width", "tau_width", "spatial_nodes", "search_nodes"):
                values = np.asarray([row[key] for row in selected])
                entry[key] = float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else np.nan
            summary.append(entry)
            print(f"{snr_tag(snr_db):>9} | {name:14} | coverage={entry['joint_covered']:.1%} | RMSE={entry['RMSE']:.3f} m | time={entry['total_seconds']:.3f} s | completed={entry['completed']}/{entry['trials']}")
    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary

# create the spatial error map for both ROIs
def plot_results(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    snrs = sorted({row["snr_db"] for row in rows})
    finite_errors = [row["position_error"] for row in rows if np.isfinite(row["position_error"])]
    vmax = max(max(finite_errors, default=1.0), 1e-12)
    norm = plt.Normalize(0.0, vmax)
    angles = np.deg2rad(np.linspace(THETA_MIN, THETA_MAX, 301))
    for snr_db in snrs:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True, constrained_layout=True)
        for ax, name in zip(axes, METHODS):
            selected = [row for row in rows if row["snr_db"] == snr_db and row["method"] == name]
            radii = np.asarray([row["R_true"] for row in selected])
            theta = np.deg2rad([row["theta_true"] for row in selected])
            errors = np.asarray([row["position_error"] for row in selected])
            covered = np.asarray([row["joint_covered"] for row in selected], dtype=bool)
            finite = np.isfinite(errors)
            x, y = radii * np.sin(theta), radii * np.cos(theta)
            sc = ax.scatter(x[finite], y[finite], c=errors[finite], cmap="turbo", norm=norm, s=24, edgecolors="none")
            if np.any(~covered & finite):
                ax.scatter(x[~covered & finite], y[~covered & finite], facecolors="none", edgecolors="black", s=48, linewidths=0.8, label="Truth outside ROI")
            if np.any(~finite):
                ax.scatter(x[~finite], y[~finite], marker="x", color="red", s=45, label="Localization failed")
            for radius in (R_MIN, R_MAX):
                ax.plot(radius * np.sin(angles), radius * np.cos(angles), color="black", linewidth=1.2)
            for angle in np.deg2rad([THETA_MIN, THETA_MAX]):
                ax.plot(np.array([R_MIN, R_MAX]) * np.sin(angle), np.array([R_MIN, R_MAX]) * np.cos(angle), color="black", linestyle="--", linewidth=1.0)
            stats = next(s for s in summary if s["snr_db"] == snr_db and s["method"] == name)
            text = f"N={len(selected)}, completed={np.sum(finite)}\nALE={stats['ALE']:.2f} m; RMSE={stats['RMSE']:.2f} m\nJoint coverage={stats['joint_covered']:.1%}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7"))
            ax.set_title(name)
            ax.set_xlabel("x (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_ylim(0.0, R_MAX * 1.2)
            ax.grid(True, alpha=0.25)
            if np.any(~covered | ~finite):
                ax.legend(loc="lower center", fontsize=8)
        axes[0].set_ylabel("y (m)")
        fig.colorbar(sc, ax=axes, label="2-D localization error (m)", shrink=0.8)
        fig.suptitle(f"True source positions — {snr_tag(snr_db)} — shared hinted PRT")
        fig.savefig(OUTPUT_DIR / f"positions_{snr_tag(snr_db)}.png", dpi=180)
        plt.close(fig)

    # plot coverage, error, runtime, and search size across the SNRs
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    metrics = (("joint_covered", "Joint ROI coverage", "%"), ("RMSE", "Localization RMSE", "m"), ("P95", "95th percentile error", "m"), ("total_seconds", "Total processing time", "s/run"), ("spatial_nodes", "HRT range × angle candidates", "Mean count"), ("range_width", "Range ROI width", "m"))
    for ax, (key, title, units) in zip(axes.flat, metrics):
        for name in METHODS:
            values = [next(s[key] for s in summary if s["snr_db"] == snr and s["method"] == name) for snr in snrs]
            values = np.asarray(values) * (100.0 if key == "joint_covered" else 1.0)
            ax.plot(np.arange(len(snrs)), np.where(np.isfinite(values), values, np.nan), marker="o", label=name)
            for index in np.flatnonzero(~np.isfinite(values)):
                ax.text(index, 0.95 if name == METHODS[0] else 0.85, "unavailable/∞", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(snrs)), [snr_tag(snr) for snr in snrs], rotation=25)
        ax.set_title(title)
        ax.set_ylabel(units)
        ax.set_xlabel("Pulse-energy SNR")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0.0, 105.0)
    axes[0, 0].legend()
    fig.savefig(OUTPUT_DIR / "metrics_vs_snr.png", dpi=180)
    plt.close(fig)

    # show the parameter margin violations
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ("p_bound_exceeded", "q_bound_exceeded", "tau_bound_exceeded"), ("p error > dp/2", "q error > E_q", "Time error > dt/2")):
        values = [100.0 * next(s[key] for s in summary if s["snr_db"] == snr and s["method"] == METHODS[0]) for snr in snrs]
        ax.plot(np.arange(len(snrs)), values, marker="o")
        ax.set_xticks(np.arange(len(snrs)), [snr_tag(snr) for snr in snrs], rotation=25)
        ax.set_title(title)
        ax.set_ylabel("Shared PRT trials (%)")
        ax.set_ylim(0.0, 105.0)
        ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_DIR / "prt_margin_violations.png", dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Rebuild plots and summary from trials_*.csv in OUTPUT_DIR")
    args = parser.parse_args()
    if args.plot_only:
        paths = sorted(OUTPUT_DIR.glob("trials_*.csv"))
        if not paths:
            raise FileNotFoundError(f"No trial CSVs in {OUTPUT_DIR}")
        plot_results(read_trials(paths))
        return
    if any(OUTPUT_DIR.glob("trials_*.csv")):
        raise FileExistsError(f"Existing trial results in {OUTPUT_DIR}; choose a new OUTPUT_DIR or use --plot-only")
    
    antenna_x = ARRAY.positions(SYSTEM)
    L = float(np.ptp(antenna_x))
    Np = get_np(L, SYSTEM.fc, SYSTEM.bandwidth)
    E_q_cont, dp, dt = get_error_margins(L, SYSTEM.fc, SYSTEM.bandwidth, Np, oversampling=SYSTEM.oversamp)

    rows = []
    for snr_db in SNR_VALUES:
        rows.extend(run_snr(snr_db, (Np, E_q_cont, dp, dt)))
        plot_results(rows)
    print(f"Results and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
