"""Single-panel LRT response for the 50 m, broadside source.

Place in experiments/ of the localization repository and run:
    python experiments/slide_lrt_50m.py

Uses the same source, noise calibration, waveform options, LRT grid, and
display limits as the two-source Radon comparison script. This is a single
source scene, not a source-separation experiment.
"""

from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.qam_source import MultiSourceGen
from transforms.lrt import run_linear_rt_cupy


RANGE_M = 50.0
THETA_DEG = 0.0
WAVEFORM = "gaussian"  # "gaussian" or "qam"
NOISELESS = False
MF = False
QAM_PW_SEC = 2e-9
QAM_M = 16
CLOCK_OFFSET = 0.0
CPO = True

LRT_N_P = 601
RECORD_HALF_WINDOW_S = 8e-9
PLOT_TIME_NS = (-5.0, 5.0)
PLOT_PC_HALF_WIDTH = 0.35
DB_MIN, DB_MAX = -35.0, 0.0
CMAP = "magma"
TRUTH_COLOR = "#4DEEEA"


def reference_peak_power(x, f0, seed, reference_range):
    """Fixed-reference peak-SNR convention from the comparison script."""
    center = int(np.argmin(np.abs(x)))
    if WAVEFORM == "gaussian":
        delay = propagation_delays(
            reference_range, 0.0, clock_offset=CLOCK_OFFSET, antenna_x=x
        )[center]
        clean, _ = received_signal(
            np.array([delay]), reference_range, 0.0,
            clock_offset=CLOCK_OFFSET, antenna_x=x,
        )
        return float(abs(clean[center, 0]) ** 2)

    qam_cal = MultiSourceGen(
        pw_sec=QAM_PW_SEC, M=QAM_M, antenna_x=x[center:center + 1],
        f0=f0, seed=seed + 3,
    )
    clean, _, _ = qam_cal.generate([dict(
        range_m=reference_range, theta_deg=0.0, clock_offset=CLOCK_OFFSET
    )])
    return float(np.max(np.abs(clean[0]) ** 2))


def make_record(x, f0, seed, variance):
    center_arrival = RANGE_M / C + CLOCK_OFFSET
    delays = propagation_delays(
        RANGE_M, THETA_DEG, clock_offset=CLOCK_OFFSET, antenna_x=x
    )
    pulse_margin = (
        8 * SYSTEM.gaussian_sigma if WAVEFORM == "gaussian"
        else 1.5 * QAM_PW_SEC + 4 * SYSTEM.dt
    )
    half_window = max(
        RECORD_HALF_WINDOW_S,
        float(np.max(np.abs(delays - center_arrival))) + pulse_margin,
    )
    n0 = int(np.floor((center_arrival - half_window) / SYSTEM.dt))
    n1 = int(np.ceil((center_arrival + half_window) / SYSTEM.dt))
    time_axis = np.arange(n0, n1 + 1) * SYSTEM.dt

    if WAVEFORM == "gaussian":
        clean, _ = received_signal(
            time_axis, RANGE_M, THETA_DEG,
            clock_offset=CLOCK_OFFSET, antenna_x=x,
        )
        info = None
    else:
        qam_gen = MultiSourceGen(
            pw_sec=QAM_PW_SEC, M=QAM_M, antenna_x=x,
            f0=f0, seed=seed + 2,
        )
        clean, info = qam_gen.received_signal(
            time_axis, [dict(
                range_m=RANGE_M, theta_deg=THETA_DEG,
                clock_offset=CLOCK_OFFSET,
            )]
        )

    rng_noise = np.random.default_rng(seed + 1)
    noise_scale = np.sqrt(variance / 2.0)
    noisy = clean + noise_scale * (
        rng_noise.standard_normal(clean.shape)
        + 1j * rng_noise.standard_normal(clean.shape)
    )

    if MF:
        if WAVEFORM == "gaussian":
            half = int(np.ceil(6 * SYSTEM.gaussian_sigma / SYSTEM.dt))
            lag = np.arange(-half, half + 1) * SYSTEM.dt
            pulse = np.exp(-0.5 * (lag / SYSTEM.gaussian_sigma) ** 2)
            pulse /= np.linalg.norm(pulse)
        else:
            reference = info[0]["waveform_reference"]
            pulse = np.conj(reference[::-1]) / np.linalg.norm(reference)
        data = fftconvolve(noisy, pulse[None, :], mode="same", axes=1)
    else:
        data = noisy
    return data, time_axis


def main():
    if WAVEFORM not in ("gaussian", "qam"):
        raise ValueError("WAVEFORM must be 'gaussian' or 'qam'")

    x = np.asarray(ARRAY.positions(SYSTEM), dtype=float)
    f0 = float(SYSTEM.fc)
    seed = int(getattr(SIM, "rng_seed", 0))
    reference_range = float(getattr(SIM, "reference_range", 300.0))
    reference_snr_db = float(getattr(SIM, "reference_snr_db", 10.0))
    peak_power = reference_peak_power(x, f0, seed, reference_range)
    variance = (
        0.0 if NOISELESS
        else peak_power / 10 ** (reference_snr_db / 10)
    )
    data, time_axis = make_record(x, f0, seed, variance)
    time_ns = (time_axis - RANGE_M / C - CLOCK_OFFSET) * 1e9

    # Relative linear delay p*x, p = -sin(theta)/c.
    p_lrt = np.linspace(-1.1 / C, 1.1 / C, LRT_N_P)
    lrt = run_linear_rt_cupy(data, x, SYSTEM.dt, p_lrt, f0, cpo=CPO)
    power = np.abs(lrt) ** 2
    lrt_db = 10 * np.log10(np.maximum(power / (np.max(power) + 1e-30), 1e-4))

    fig, ax = plt.subplots(figsize=(8, 5.3), layout="constrained")
    im = ax.imshow(
        lrt_db.T, extent=[time_ns[0], time_ns[-1],
                          p_lrt[0] * C, p_lrt[-1] * C],
        origin="lower", aspect="auto", cmap=CMAP,
        vmin=DB_MIN, vmax=DB_MAX,
    )
    pc_true = -np.sin(np.deg2rad(THETA_DEG))
    ax.axhline(
        pc_true, color=TRUTH_COLOR, linestyle="--", linewidth=1.0,
        label=f"Source: R={RANGE_M:g} m, θ={THETA_DEG:g}°",
    )
    ax.set_xlim(*PLOT_TIME_NS)
    ax.set_ylim(pc_true - PLOT_PC_HALF_WIDTH,
                pc_true + PLOT_PC_HALF_WIDTH)
    ax.set_xlabel("Relative arrival time (ns)", fontsize=20)
    ax.set_ylabel(r"Normalized slowness $pc$", fontsize=20)
    ax.set_title("Linear Radon Transform", fontsize=25)
    ax.legend(loc="upper right", fontsize=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Magnitude normalized to panel peak (dB)")

    output_dir = HERE / (
        f"slide_lrt_50m_{WAVEFORM}_{'mf' if MF else 'raw'}_"
        f"{'noiseless' if NOISELESS else 'noisy'}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "radon_lrt_50m.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    print(f"Config: {ROOT / 'config.py'}; {data.shape[0]} antennas, "
          f"{data.shape[1]} samples; noise variance={variance:.6g}")
    plt.show()


if __name__ == "__main__":
    main()
