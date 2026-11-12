"""Same observation as two_source_broadband_phase_slide.py: retain HRT time.

Run alongside that script in experiments/. This is a diagnostic at the known
common angle, not a blind three-dimensional search or two-source estimator.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys

from slide_13_dual_source import (
    ARRAY, C, R1_M, R2_M, R_GRID, SIM, SYSTEM,
    CENTER_ARRIVAL_SEPARATION_PS, THETA1_DEG, THETA2_DEG,
    broadband_phase_score, noise_variance_at_reference,
    propagation_delays, received_signal,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from slide_13_dual_source import (
    ARRAY, C, R1_M, R2_M, R_GRID, SIM, SYSTEM,
    CENTER_ARRIVAL_SEPARATION_PS, THETA1_DEG, THETA2_DEG,
    broadband_phase_score, noise_variance_at_reference,
    propagation_delays, received_signal,
)


OUTPUT = HERE / "two_source_hrt_time_slice.png"
OUTPUT_STANDALONE = HERE / "two_source_hrt_only.png"


def hrt_slice(data, antenna_x, ranges, theta_deg):
    """Spatially align each frequency, then IFFT before taking power."""
    spectrum = np.fft.fft(data, axis=1)
    freq = np.fft.fftfreq(data.shape[1], d=SYSTEM.dt)
    occupied = abs(freq) <= SYSTEM.bandwidth
    spectrum[:, ~occupied] = 0
    f0 = float(getattr(SYSTEM, "f0", SYSTEM.fc))
    theta = np.deg2rad(theta_deg)
    power = np.empty((len(ranges), data.shape[1]))

    for i, r in enumerate(ranges):
        distance = np.sqrt(r*r + antenna_x**2 - 2*r*antenna_x*np.sin(theta))
        delta = (distance - r) / C
        amplitude = r / distance
        correction = np.exp(2j * np.pi * (f0 + freq[None, :]) * delta[:, None])
        aligned_spectrum = np.sum(
            amplitude[:, None] * correction * spectrum, axis=0
        ) / np.sqrt(np.sum(amplitude**2))
        trace = np.fft.ifft(aligned_spectrum)
        power[i] = abs(trace)**2
    return power


def main():
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    center_time = R1_M / C
    gap = CENTER_ARRIVAL_SEPARATION_PS * 1e-12
    offset2 = (R1_M - R2_M) / C + gap
    sources = [(R1_M, THETA1_DEG, 0.0), (R2_M, THETA2_DEG, offset2)]

    # Same time axis, source order, and receiver-noise draw as the phase slide.
    half_width = 0.62e-9
    n0 = int(np.floor((center_time - half_width) / SYSTEM.dt))
    n1 = int(np.ceil((center_time + half_width) / SYSTEM.dt))
    time = np.arange(n0, n1 + 1) * SYSTEM.dt
    clean = np.zeros((len(x), len(time)), complex)
    for r, theta, offset in sources:
        rx, _ = received_signal(time, r, theta, clock_offset=offset, antenna_x=x)
        clean += rx
    variance = noise_variance_at_reference(x)
    rng = np.random.default_rng(int(getattr(SIM, "rng_seed", 0)) + 1)
    noisy = clean + np.sqrt(variance / 2) * (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    )

    angle_grid = np.arange(-2.0, 2.001, 0.1)
    spatial, n_bins = broadband_phase_score(
        noisy, x, SYSTEM.dt, R_GRID, angle_grid,
    )
    time_power = hrt_slice(noisy, x, R_GRID, THETA1_DEG)
    time_ns = (time - center_time) * 1e9

    # Parseval: the phase-map score at broadside is the HRT time energy,
    # apart from the inverse FFT's normalization.
    spatial_at_broadside = spatial[:, np.argmin(abs(angle_grid - THETA1_DEG))]
    relative_parseval_error = np.max(
        abs(time_power.sum(axis=1) * len(time) - spatial_at_broadside)
        / np.maximum(spatial_at_broadside, 1e-30)
    )
    assert relative_parseval_error < 1e-10

    spat_db = 10 * np.log10(np.maximum(spatial, 1e-30) / spatial.max())
    temporal_db = 10 * np.log10(np.maximum(time_power, 1e-30) / time_power.max())
    colors = ("#2be654", "#3330f8")

    # -------------------------------------------------------------------------
    # Figure 1: Standalone single-panel two-source HRT score figure
    # -------------------------------------------------------------------------
    fig_hrt, ax_hrt = plt.subplots(figsize=(10.2, 5.6), layout="constrained")
    heat_hrt = ax_hrt.imshow(
        temporal_db.T, origin="lower", aspect="auto", cmap="magma",
        vmin=-15, vmax=0, interpolation="nearest",
        extent=(R_GRID[0], R_GRID[-1], time_ns[0], time_ns[-1]),
    )
    for j, (r, theta, _) in enumerate(sources):
        t_true_ns = (0.0, gap * 1e9)[j]
        ax_hrt.scatter(
            r, t_true_ns, marker="x", s=150,
            linewidths=2.5, c=colors[j],
            label=f"True source {j+1}: {r:g} m",
            zorder=5,
        )
    ax_hrt.set_title(r"HRT Range–$\tau$ Slice at True $\theta=0^\circ$", fontsize=25)
    ax_hrt.set_xlabel("Candidate range (m)", fontsize=20)
    ax_hrt.set_ylabel("Center arrival time\nrelative to source 1 (ns)", fontsize=20)
    ax_hrt.set_ylim(-0.1, 0.3)
    ax_hrt.tick_params(labelsize=14)
    ax_hrt.legend(loc="upper right", fontsize=15, framealpha=0.94)
    fig_hrt.colorbar(
        heat_hrt, ax=ax_hrt, label="HRT magnitude (dB below max)",
        pad=0.02, fraction=0.047,
    )
    fig_hrt.savefig(OUTPUT_STANDALONE, dpi=220, facecolor="white")
    print(f"Saved standalone HRT figure to {OUTPUT_STANDALONE}")

    # -------------------------------------------------------------------------
    # Figure 2: Original two-panel comparison figure (with source legends)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.1), layout="constrained")
    im0 = axes[0].imshow(
        spat_db.T, origin="lower", aspect="auto", cmap="magma", vmin=-10, vmax=0,
        extent=(R_GRID[0], R_GRID[-1], angle_grid[0], angle_grid[-1]),
        interpolation="nearest",
    )
    axes[0].set_title("Broadband spatial score", fontsize=22)
    axes[0].set(xlabel="Candidate range (m)", ylabel="Candidate angle (deg)")
    fig.colorbar(im0, ax=axes[0], label="dB below spatial maximum")

    im1 = axes[1].imshow(
        temporal_db.T, origin="lower", aspect="auto", cmap="magma", vmin=-15, vmax=0,
        extent=(R_GRID[0], R_GRID[-1], time_ns[0], time_ns[-1]),
        interpolation="nearest",
    )
    axes[1].set_title(r"HRT score at $\theta=0^\circ$", fontsize=22)
    axes[1].set(
        xlabel="Candidate range (m)",
        ylabel="Center arrival time relative to source 1 (ns)",
    )
    axes[1].set_ylim(-0.1, 0.3)
    fig.colorbar(im1, ax=axes[1], label="dB below HRT maximum")

    for j, (r, theta, _) in enumerate(sources):
        axes[0].scatter(
            r, theta, s=155, marker="x", c=colors[j],
            linewidths=2.5, label=f"True source {j+1}: {r:g} m", zorder=5,
        )
        axes[1].scatter(
            r, (0.0, gap * 1e9)[j], s=155, marker="x", c=colors[j],
            linewidths=2.5, label=f"True source {j+1}: {r:g} m", zorder=5,
        )
    for ax in axes:
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.legend(loc="upper right", fontsize=13, framealpha=0.94)

    fig.savefig(OUTPUT, dpi=220, facecolor="white")
    print(f"Saved {OUTPUT}; {n_bins} baseband bins; gap={gap*1e12:g} ps")
    print(f"Broadside Parseval relative error={relative_parseval_error:.3g}")
    for j, (r, theta, _) in enumerate(sources):
        i = np.argmin(abs(R_GRID - r))
        k = np.argmin(abs(time_ns - (0.0, gap * 1e9)[j]))
        print(f"Truth {j+1}: HRT score {temporal_db[i, k]:.2f} dB; "
              f"spatial score {spat_db[i, np.argmin(abs(angle_grid - theta))]:.2f} dB")
    plt.show()

if __name__ == "__main__":
    main()
