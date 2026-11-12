"""Regenerate the overnight bandwidth/separation figures without repeating MC.

Place this file in the repository's visualizations/ directory. It expects:

    experiments/overnight_bw_sep_run.py
    experiments/overnight_bw_separation_results/

Run from anywhere with ``python visualizations/replot_bw_separation.py``.
The detection curve is calculated again from sweep_trials.csv. The saved
folder contains PNGs but not the numerical HRT maps, so the six illustrative
map cases are replayed with the original experiment's generator, seed, and
HRT implementation. No calibration or Monte Carlo trials are rerun.
"""

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import resample


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "overnight_bw_sep_run.py"
RESULT_DIR = ROOT / "experiments" / "overnight_bw_separation_results"
OUTPUT_DIR = RESULT_DIR / "replotted_figures"
REMAKE_MAPS = True  # Set False to copy the original map PNGs instead.
MAKE_QAM_TRUE_TAU_MAPS = True
ANGLE_MIN_DEG, ANGLE_MAX_DEG, ANGLE_STEP_DEG = -8.0, 8.0, 0.2
SLICE_INTERPOLATION = 8  # Fourier evaluation of the diagnostic tau slices.

# Edit all figure titles, axis labels, panel titles, legend text, and captions
# here. Templates use {bandwidth:g} and/or {source_index} where applicable.
TEXT = {
    "combined_title": "Raw HRT, same received energy per source; recorded time samples only",
    "combined_gaussian_panel": "Gaussian: B={bandwidth:g} GHz",
    "combined_qam_panel": "RRC-QAM: {bandwidth:g} GHz symbol rate",
    "combined_x": "Candidate range (m)",
    "combined_y": "Arrival relative to source 1 (ps)",
    "combined_colorbar": "HRT magnitude (dB below each panel maximum)",
    "gaussian_title": (
        "Equal-energy Gaussian Pulses, Varied Bandwidth\n"
        r"HRT $R$–$\tau$ Slices at True $\theta=0^\circ$"
    ),
    "qam_title": (
        "Equal-energy RRC-QAM Bursts, Fixed Duration\n"
        r"HRT $R$–$\tau$ Slices at True $\theta=0^\circ$"
    ),
    "bandwidth_panel": "{bandwidth:g} GHz",
    "bandwidth_x": "Candidate range (m)",
    "bandwidth_y": "Arrival relative to source 1 (ps)",
    "bandwidth_colorbar": "HRT magnitude (dB below panel maximum)",
    "qam_caption": "Crosses mark burst centers; QAM peaks depend on symbols",
    "detection_title": "Blind Peak Detection of Two 8 GHz Gaussian Pulses",
    "detection_x": "True center-arrival gap (ps)",
    "detection_y": "Fraction of trials",
    "detection_two": "Exactly two detected",
    "detection_correct": "Both within range/time tolerances",
    "qam_slice_title": "RRC-QAM: Range–Angle HRT at Each Nominal Source Arrival",
    "qam_slice_panel": "{bandwidth:g} GHz; τ{source_index} (source {source_index} arrival)",
    "qam_slice_inset": "{bandwidth:g} GHz",
    "qam_slice_x": "Candidate range (m)",
    "qam_slice_y": "Candidate angle (deg)",
    "qam_slice_colorbar": "HRT power (dB below this row's maximum)",
    "true_locations": "True locations",
}


def read_sweep(folder):
    with (folder / "sweep_trials.csv").open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    if not trials:
        raise ValueError("sweep_trials.csv is empty")

    grouped = defaultdict(list)
    for row in trials:
        grouped[float(row["gap_ps"])].append(row)
    rows = []
    for gap, group in sorted(grouped.items()):
        rows.append(dict(
            gap_ps=gap,
            n=len(group),
            exact_two_rate=np.mean([int(r["exact_two"]) for r in group]),
            both_correct_rate=np.mean([int(r["both_correct"]) for r in group]),
        ))
    return rows


def plot_detection(rows, destination):
    gaps = [r["gap_ps"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(gaps, [r["exact_two_rate"] for r in rows], "o-", lw=2,
            label=TEXT["detection_two"])
    ax.plot(gaps, [r["both_correct_rate"] for r in rows], "s-", lw=2,
            label=TEXT["detection_correct"])
    ax.set(xlabel=TEXT["detection_x"],
           ylabel=TEXT["detection_y"], ylim=(-0.04, 1.04),
           xlim=(min(gaps), max(gaps)),
           title=TEXT["detection_title"])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)


def load_original_experiment():
    if not EXPERIMENT.is_file():
        raise FileNotFoundError(
            f"Cannot regenerate heatmaps without {EXPERIMENT}. "
            "The result folder stores PNGs, not HRT score grids. "
            "Set REMAKE_MAPS=False to reuse its existing PNGs."
        )
    # The original experiment imports config and signal_model from the repo.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("original_bw_experiment", EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_bandwidth_maps(maps, module, output_dir):
    """Make the original combined map and separate slide-sized map figures."""
    ranges = module.RANGES_M
    tau0 = module.R1_M/module.C
    gap = module.ILLUSTRATION_GAP_PS
    families = ("Gaussian", "RRC-QAM")
    bandwidths = module.BANDWIDTHS_GHZ

    fig, axes = plt.subplots(2, len(bandwidths), figsize=(15.6, 8.4),
                             layout="constrained")
    for row, family in enumerate(families):
        for col, bandwidth in enumerate(bandwidths):
            score, t, _ = maps[(family, bandwidth)]
            tau_ps = (t-tau0)*1e12
            db = 10*np.log10(np.maximum(score/score.max(), 1e-15))
            ax = axes[row, col]
            im = ax.imshow(db.T, origin="lower", aspect="auto", cmap="magma",
                           extent=(ranges[0], ranges[-1], tau_ps[0], tau_ps[-1]),
                           vmin=-15, vmax=0)
            ax.plot([module.R1_M, module.R2_M], [0, gap], "x",
                    color="#23d7b0", ms=8, mew=2)
            template = (TEXT["combined_gaussian_panel"] if row == 0 else
                        TEXT["combined_qam_panel"])
            ax.set_title(template.format(bandwidth=bandwidth), fontsize=25)
            ax.set_xlabel(TEXT["combined_x"], fontsize = 20)
            if col == 0:
                ax.set_ylabel(TEXT["combined_y"], fontsize=20)
            ax.set_ylim((-450, 600) if row == 0 else (-2200, 2400))
    fig.colorbar(im, ax=axes, shrink=0.8, label=TEXT["combined_colorbar"])
    fig.suptitle(TEXT["combined_title"], fontsize=25)
    fig.savefig(output_dir / "bandwidth_hrt_maps.png", dpi=170,
                facecolor="white")
    plt.close(fig)

    for row, (family, filename, title_key) in enumerate((
        ("Gaussian", "gaussian_bandwidth_hrt.png", "gaussian_title"),
        ("RRC-QAM", "rrc_qam_bandwidth_hrt.png", "qam_title"),
    )):
        figure, panels = plt.subplots(
            1, len(bandwidths), figsize=(15, 6),
            sharex=True, sharey=True, layout="constrained")
        for panel, bandwidth in zip(panels, bandwidths):
            score, t, _ = maps[(family, bandwidth)]
            tau_ps = (t-tau0)*1e12
            db = 10*np.log10(np.maximum(score/score.max(), 1e-15))
            im = panel.imshow(
                db.T, origin="lower", aspect="auto", cmap="magma",
                extent=(ranges[0], ranges[-1], tau_ps[0], tau_ps[-1]),
                vmin=-15, vmax=0)
            panel.plot(module.R1_M, 0, "x", color="#2be654", ms=9, mew=2, label=f"Source 1: {module.R1_M:g} m")
            panel.plot(module.R2_M, gap, "x", color="#3330f8", ms=9, mew=2, label=f"Source 2: {module.R2_M:g} m")
            panel.set_title(TEXT["bandwidth_panel"].format(bandwidth=bandwidth),
                            fontsize=25)
            panel.set_xlabel(TEXT["bandwidth_x"], fontsize=20)
            panel.set_ylim((-450, 600) if row == 0 else (-2200, 2400))
            panel.tick_params(labelsize=12)
        panels[0].set_ylabel(TEXT["bandwidth_y"], fontsize=20)
        figure.colorbar(im, ax=panels, shrink=0.85,
                        label=TEXT["bandwidth_colorbar"])
        # if row == 1:
        #     figure.supxlabel(TEXT["qam_caption"], fontsize=20)
        figure.suptitle(TEXT[title_key], fontsize=25)
        panels[0].legend(loc="upper left", fontsize=15)
        figure.savefig(output_dir / filename, dpi=190, facecolor="white")
        plt.close(figure)


def score_rtheta_slices(data, time_axis, antenna_x, ranges, angles_deg,
                        arrivals, f0, c):
    """Spherical delay-and-sum at specified output times and spatial points.

    The padded raw antenna records are Fourier-interpolated once, and each
    candidate then samples its trace at tau + (distance - range)/c. This is
    solely for these known-tau diagnostic slices, not the MC detector.
    """
    dt = float(time_axis[1] - time_axis[0])
    r0 = float(np.min(ranges))
    max_delay = 0.0
    for angle in (float(np.min(angles_deg)), float(np.max(angles_deg))):
        th = np.deg2rad(angle)
        distances = np.hypot(antenna_x - r0*np.sin(th), r0*np.cos(th))
        max_delay = max(max_delay, float(np.max(abs(distances-r0))/c))
    guard = int(np.ceil(max_delay/dt)) + 16
    nfft = next_fast_len(data.shape[1] + 2*guard)
    padded = np.pad(data, ((0, 0),
                           (guard, nfft-data.shape[1]-guard)))
    traces = resample(padded, nfft*SLICE_INTERPOLATION, axis=1)
    first_time = time_axis[0] - guard*dt
    fine_dt = dt/SLICE_INTERPOLATION
    antenna_indices = np.arange(len(antenna_x))[None, :]

    rr, tt = np.meshgrid(ranges, np.deg2rad(angles_deg), indexing="ij")
    rr = rr.ravel()
    tt = tt.ravel()
    scores = [np.empty(len(rr)) for _ in arrivals]
    for begin in range(0, len(rr), 128):
        end = min(begin+128, len(rr))
        r = rr[begin:end, None]
        th = tt[begin:end, None]
        dist = np.hypot(antenna_x[None, :] - r*np.sin(th), r*np.cos(th))
        delay = (dist-r)/c
        amp = r/dist
        weight = amp/np.sqrt(np.sum(amp**2, axis=1, keepdims=True))
        weight = weight * np.exp(2j*np.pi*f0*delay)
        for k, tau in enumerate(arrivals):
            index_float = (tau+delay-first_time)/fine_dt
            index = np.floor(index_float).astype(np.int64)
            fraction = index_float-index
            valid = (index >= 0) & (index+1 < traces.shape[1])
            index = np.clip(index, 0, traces.shape[1]-2)
            value = ((1-fraction)*traces[antenna_indices, index]
                     + fraction*traces[antenna_indices, index+1])
            value *= valid
            scores[k][begin:end] = abs(np.sum(weight*value, axis=1))**2
    return [score.reshape(len(ranges), len(angles_deg)) for score in scores]


def plot_qam_rtheta_slices(slices_by_bandwidth, module, destination):
    """Two true-arrival slices per symbol rate; common scale within each row."""
    bandwidths = module.BANDWIDTHS_GHZ
    ranges = module.RANGES_M
    angles = np.arange(ANGLE_MIN_DEG,
                       ANGLE_MAX_DEG+ANGLE_STEP_DEG/2, ANGLE_STEP_DEG)
    fig, axes = plt.subplots(len(bandwidths), 2, figsize=(10.8, 11.5),
                             sharex=True, sharey=True, layout="constrained")
    for row, bandwidth in enumerate(bandwidths):
        two_slices = slices_by_bandwidth[bandwidth]
        common_peak = max(float(np.max(s)) for s in two_slices)
        for col, score in enumerate(two_slices):
            ax = axes[row, col]
            db = 10*np.log10(np.maximum(score/max(common_peak, 1e-30), 1e-15))
            im = ax.imshow(db.T, origin="lower", aspect="auto", cmap="magma",
                           extent=(ranges[0], ranges[-1], angles[0], angles[-1]),
                           vmin=-20, vmax=0)
            ax.plot([module.R1_M, module.R2_M], [0, 0], "x",
                    color="#23d7b0", ms=9, mew=2,
                    label=TEXT["true_locations"])
            ax.set_title(TEXT["qam_slice_panel"].format(
                bandwidth=bandwidth, source_index=col+1), fontsize=25)
            ax.set_xlabel(TEXT["qam_slice_x"])
            if col == 0:
                ax.set_ylabel(TEXT["qam_slice_y"])
            ax.text(0.02, 0.97,
                    TEXT["qam_slice_inset"].format(bandwidth=bandwidth),
                    transform=ax.transAxes,
                    va="top", color="white", fontsize=15)
            ax.tick_params(labelsize=11)
    fig.colorbar(im, ax=axes, shrink=0.82,
                 label=TEXT["qam_slice_colorbar"])
    fig.suptitle(TEXT["qam_slice_title"], fontsize=25)
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)


def remake_maps(module, summary):
    # Reproduce only the map-generating portion of the original main().
    from config import ARRAY
    x = np.asarray(ARRAY.positions(module.SYSTEM), float)
    dt = float(module.SYSTEM.dt)
    if int(summary["seed"]) != module.SEED:
        raise ValueError("Experiment seed differs from the saved result")
    if int(summary["antennas"]) != len(x):
        raise ValueError("Array size differs from the saved result")
    if not np.isclose(float(summary["sample_period_ps"]), dt * 1e12):
        raise ValueError("Sample interval differs from the saved result")
    variance, _, _ = module.reference_noise(x)
    if not np.isclose(variance, float(summary["complex_noise_variance"]),
                      rtol=1e-8, atol=0):
        raise ValueError("Receiver noise variance differs from the saved result")

    tau1 = module.R1_M / module.C
    rng_demo = np.random.default_rng(module.SEED + 1)
    maps = {}
    qam_records = {}
    gaussian_margin = (8 * module.bw_sigma(min(module.BANDWIDTHS_GHZ))
                       + 0.4e-9)
    gaussian_time = module.axis(
        tau1, gaussian_margin,
        gaussian_margin + module.ILLUSTRATION_GAP_PS * 1e-12, dt)
    qam_margin = 1.5 * module.QAM_DURATION_S + 0.4e-9
    qam_time = module.axis(
        tau1, qam_margin,
        qam_margin + module.ILLUSTRATION_GAP_PS * 1e-12, dt)

    for kind, time_axis in (("Gaussian", gaussian_time),
                            ("RRC-QAM", qam_time)):
        hrt = module.BroadsideHRT(time_axis, x, module.RANGES_M)
        # One shared noise realization per waveform family, in original order.
        noise = module.add_noise(
            np.zeros((len(x), len(time_axis)), complex), rng_demo, variance)
        for bandwidth in module.BANDWIDTHS_GHZ:
            if kind == "Gaussian":
                clean = module.gaussian_source(
                    time_axis, x, module.R1_M, tau1, bandwidth)
                clean += module.gaussian_source(
                    time_axis, x, module.R2_M,
                    tau1 + module.ILLUSTRATION_GAP_PS * 1e-12, bandwidth)
                info = None
            else:
                clean, info = module.qam_pair(
                    time_axis, x, bandwidth, module.ILLUSTRATION_GAP_PS)
            received = clean + noise
            maps[(kind, bandwidth)] = (
                hrt.scores(received), time_axis, info)
            if kind == "RRC-QAM" and MAKE_QAM_TRUE_TAU_MAPS:
                qam_records[bandwidth] = (received, time_axis)
            print(f"Recomputed {kind} {bandwidth:g} GHz map", flush=True)

    plot_bandwidth_maps(maps, module, OUTPUT_DIR)
    print("Saved the three bandwidth figures first", flush=True)
    if MAKE_QAM_TRUE_TAU_MAPS:
        slices_by_bandwidth = {}
        angles = np.arange(ANGLE_MIN_DEG,
                           ANGLE_MAX_DEG+ANGLE_STEP_DEG/2,
                           ANGLE_STEP_DEG)
        for bandwidth, (received, time_axis) in qam_records.items():
            slices_by_bandwidth[bandwidth] = score_rtheta_slices(
                received, time_axis, x, module.RANGES_M, angles,
                (tau1, tau1+module.ILLUSTRATION_GAP_PS*1e-12),
                float(getattr(module.SYSTEM, "f0", module.SYSTEM.fc)),
                module.C)
        plot_qam_rtheta_slices(
            slices_by_bandwidth, module,
            OUTPUT_DIR / "qam_range_angle_at_true_arrivals.png")
        print("Saved QAM true-arrival diagnostic", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "summary.json"
    with summary_path.open() as handle:
        summary = json.load(handle)

    filenames = ("bandwidth_hrt_maps.png", "gaussian_bandwidth_hrt.png",
                 "rrc_qam_bandwidth_hrt.png")
    if REMAKE_MAPS:
        remake_maps(load_original_experiment(), summary)
    else:
        import shutil
        for name in filenames:
            shutil.copy2(RESULT_DIR / name, OUTPUT_DIR / name)
        print("Reused saved map PNGs; the CSVs do not contain HRT map values")

    # All Monte Carlo work was completed by the original experiment. These
    # are read-only calculations from its saved rows, performed last.
    rows = read_sweep(RESULT_DIR)
    # if any(r["n"] != int(summary["n_mc_per_gap"]) for r in rows):
    #     raise ValueError("Incomplete sweep_trials.csv: trial counts differ by gap")
    plot_detection(rows, OUTPUT_DIR / "separation_detection.png")
    print("Redrew separation_detection.png from saved sweep_trials.csv", flush=True)
    count = 4 + int(REMAKE_MAPS and MAKE_QAM_TRUE_TAU_MAPS)
    print(f"{count} figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
