from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
if not (ROOT / "config.py").is_file():
    ROOT = HERE / "sic2_integration_fixture"
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal


R1_M, R2_M = 40.0, 60.0
THETA1_DEG = THETA2_DEG = 0.0
CENTER_ARRIVAL_SEPARATION_PS = 160.0
R_GRID = np.arange(30.0, 75.01, 0.5)
THETA_GRID_DEG = np.arange(-2.0, 2.001, 0.1)
CHUNK = 1024

OUTPUT_BROADBAND = HERE / "two_source_broadband_phase_slide.png"
OUTPUT_NARROWBAND = HERE / "two_source_narrowband_phase_slide.png"


def noise_variance_at_reference(x):
    reference_range = float(getattr(SIM, "reference_range", 300.0))
    reference_snr_db = float(getattr(SIM, "reference_snr_db", 10.0))
    center = int(np.argmin(abs(x)))
    arrival = propagation_delays(reference_range, 0.0, antenna_x=x)[center]
    reference, _ = received_signal(
        np.array([arrival]), reference_range, 0.0, antenna_x=x
    )
    return float(abs(reference[center, 0])**2 / 10**(reference_snr_db / 10))


def broadband_phase_score(data, x, dt, ranges, angles_deg):
    """Incoherent sum over frequency of spherical spatial Bartlett power.

    For each frequency, retain its own geometric delay phase (f0 + f).
    The FFT has no time gate around either source's arrival.
    """
    spectrum = np.fft.fft(data, axis=1)
    frequencies = np.fft.fftfreq(data.shape[1], d=dt)
    occupied = abs(frequencies) <= SYSTEM.bandwidth
    frequencies = frequencies[occupied]
    spectrum = spectrum[:, occupied]

    rr, tt = np.meshgrid(ranges, angles_deg, indexing="ij")
    candidates = np.column_stack((rr.ravel(), tt.ravel()))
    scores = np.empty(len(candidates))
    f0 = float(getattr(SYSTEM, "f0", SYSTEM.fc))

    for start in range(0, len(candidates), 16):
        candidates_tile = candidates[start:start + 16]
        r = candidates_tile[:, 0, None]
        theta = np.deg2rad(candidates_tile[:, 1, None])
        dist = np.sqrt(r*r + x[None, :]**2 - 2*r*x[None, :]*np.sin(theta))
        relative_delay = (dist - r) / C
        amplitude = r / dist
        correction = np.exp(
            2j*np.pi*(f0 + frequencies[None, None, :])
            * relative_delay[:, :, None]
        )
        projections = np.einsum(
            "kmf,mf->kf", amplitude[:, :, None]*correction, spectrum,
            optimize=True,
        )
        scores[start:start + len(candidates_tile)] = (
            np.sum(abs(projections)**2, axis=1)
            / np.sum(amplitude**2, axis=1)
        )
    return scores.reshape(len(ranges), len(angles_deg)), len(frequencies)


def phase_maps(x, f0, ranges, angles_deg, data):
    """Normalized spherical carrier steering at center frequency f0."""
    rr, th = np.meshgrid(ranges, np.deg2rad(angles_deg), indexing="ij")
    rs, angles = rr.ravel(), th.ravel()
    scores = np.empty((len(rs), data.shape[1]))
    xx = x[:, None]
    for start in range(0, len(rs), CHUNK):
        stop = min(start + CHUNK, len(rs))
        r = rs[None, start:stop]
        a = angles[None, start:stop]
        dist = np.sqrt(r*r + xx*xx - 2*r*xx*np.sin(a))
        delta = (xx*xx - 2*r*xx*np.sin(a)) / (dist + r)
        steering = np.exp(-2j*np.pi*f0*delta/C) / np.sqrt(len(x))
        scores[start:stop] = np.abs(steering.conj().T @ data)**2
    return scores.reshape(len(ranges), len(angles_deg), -1)


def plot_two_source_figure(
    observation, peak, time, center_time, x, arrivals, sources,
    score_db, phase_title, output_path
):
    """Render and save the two-panel observation + phase-score slide figure."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), layout="constrained")
    t_ns = (time - center_time) * 1e9
    axes[0].imshow(
        abs(observation) / peak, cmap="Blues", origin="lower", aspect="auto",
        interpolation="nearest", vmin=0, vmax=1.7,
        extent=(t_ns[0], t_ns[-1], x[0], x[-1]),
    )
    colors = ("#2be654", "#3330f8")
    for j, delay in enumerate(arrivals):
        axes[0].plot(
            (delay - center_time) * 1e9, x, lw=2.2,
            color=colors[j], alpha=0.6,
            label=f"Source {j+1}: {sources[j][0]:g} m, {sources[j][1]:g}°",
        )
    axes[0].set_title("Received antenna–time record", fontsize=24)
    axes[0].set_xlabel("Time relative to source 1 arrival (ns)", fontsize=18)
    axes[0].set_ylabel("Antenna position (m)", fontsize=18)
    axes[0].legend(loc="lower left", fontsize=14, framealpha=0.94)

    # x axis = range, y axis = angle; phase-map values include both sources.
    heat = axes[1].imshow(
        score_db.T, origin="lower", aspect="auto", cmap="magma",
        vmin=-10, vmax=0, interpolation="nearest",
        extent=(R_GRID[0], R_GRID[-1],
                THETA_GRID_DEG[0], THETA_GRID_DEG[-1]),
    )
    for j, (r, theta, _) in enumerate(sources):
        axes[1].scatter(
            r, theta, marker=("x", "x")[j], s=150,
            linewidths=2.5, c=colors[j],
            edgecolors="white" if j == 0 else None,
            label=f"True source {j+1}: {r:g} m",
        )
    axes[1].set_title(phase_title, fontsize=24)
    axes[1].set_xlabel("Candidate range (m)", fontsize=18)
    axes[1].set_ylabel("Candidate angle (deg)", fontsize=18)
    axes[1].legend(loc="upper right", fontsize=13, framealpha=0.94)
    fig.colorbar(
        heat, ax=axes[1], label="Phase correlation magnitude (dB below max)",
        pad=0.02, fraction=0.047,
    )
    for ax in axes:
        ax.tick_params(labelsize=14)

    fig.savefig(output_path, dpi=220, facecolor="white")
    return fig


def main():
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    center_time = R1_M / C
    gap = CENTER_ARRIVAL_SEPARATION_PS * 1e-12
    offset2 = (R1_M - R2_M) / C + gap
    sources = [(R1_M, THETA1_DEG, 0.0), (R2_M, THETA2_DEG, offset2)]

    half_width = 0.62e-9
    n0 = int(np.floor((center_time - half_width) / SYSTEM.dt))
    n1 = int(np.ceil((center_time + half_width) / SYSTEM.dt))
    time = np.arange(n0, n1 + 1) * SYSTEM.dt
    clean = np.zeros((len(x), len(time)), complex)
    arrivals = []
    for r, theta, offset in sources:
        signal, delay = received_signal(
            time, r, theta, clock_offset=offset, antenna_x=x
        )
        clean += signal
        arrivals.append(delay)

    variance = noise_variance_at_reference(x)
    rng = np.random.default_rng(int(getattr(SIM, "rng_seed", 0)) + 1)
    noise = np.sqrt(variance / 2) * (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    )
    observation = clean + noise

    center = int(np.argmin(abs(x)))
    ref, _ = received_signal(
        np.array([arrivals[0][center]]), R1_M, THETA1_DEG, antenna_x=x
    )
    peak = float(abs(ref[center, 0]))

    # 1. Original broadband phase matching across occupied FFT frequencies
    score_bb, n_frequency_bins = broadband_phase_score(
        observation, x, SYSTEM.dt, R_GRID, THETA_GRID_DEG
    )
    score_bb_db = 10 * np.log10(np.maximum(score_bb, 1e-30) / score_bb.max())
    winner_bb = np.unravel_index(np.argmax(score_bb), score_bb.shape)

    plot_two_source_figure(
        observation, peak, time, center_time, x, arrivals, sources,
        score_bb_db, "Broadband phase matching", OUTPUT_BROADBAND,
    )
    print(f"Saved {OUTPUT_BROADBAND}; frequency bins={n_frequency_bins}; "
          f"gap={CENTER_ARRIVAL_SEPARATION_PS:g} ps")
    print(f"Broadband map peak: R={R_GRID[winner_bb[0]]:g} m, "
          f"theta={THETA_GRID_DEG[winner_bb[1]]:g} deg")
    print("Broadband spatial score at truth (dB below map max): ",
          [round(score_bb_db[np.argmin(abs(R_GRID - r)),
                             np.argmin(abs(THETA_GRID_DEG - theta))], 2)
           for r, theta, _ in sources])

    # 2. Center-frequency (fc) narrowband phase matching with normalized time sum
    f0 = float(getattr(SYSTEM, "f0", SYSTEM.fc))
    z_raw = observation.sum(axis=1) / np.sqrt(len(time))
    score_fc = phase_maps(
        x, f0, R_GRID, THETA_GRID_DEG, z_raw[:, None]
    )[:, :, 0]
    score_fc_db = 10 * np.log10(np.maximum(score_fc, 1e-30) / score_fc.max())
    winner_fc = np.unravel_index(np.argmax(score_fc), score_fc.shape)

    plot_two_source_figure(
        observation, peak, time, center_time, x, arrivals, sources,
        score_fc_db, "Narrowband Phase Matching", OUTPUT_NARROWBAND,
    )
    print(f"Saved {OUTPUT_NARROWBAND}; f0={f0/1e9:g} GHz; "
          f"gap={CENTER_ARRIVAL_SEPARATION_PS:g} ps")
    print(f"Narrowband (fc) map peak: R={R_GRID[winner_fc[0]]:g} m, "
          f"theta={THETA_GRID_DEG[winner_fc[1]]:g} deg")
    print("Narrowband (fc) spatial score at truth (dB below map max): ",
          [round(score_fc_db[np.argmin(abs(R_GRID - r)),
                             np.argmin(abs(THETA_GRID_DEG - theta))], 2)
           for r, theta, _ in sources])

    plt.show()


if __name__ == "__main__":
    main()