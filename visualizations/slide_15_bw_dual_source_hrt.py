"""Bandwidth sweep for two unknown-time sources using raw, template-free HRT.

Copy into experiments/ of the localization repository and run with
``python experiments/bandwidth_two_source_hrt.py``. This standalone copy also
uses the included sic2_integration_fixture when no repository config is found.

Only the Gaussian pulse width and its energy-normalizing amplitude change
between panels. The geometry, arrival times, sampling rate, noise variance,
and noise realization stay fixed. HRT does not receive the pulse waveform.
"""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
if not (ROOT / "config.py").is_file():
    ROOT = HERE / "sic2_integration_fixture"  # Standalone demonstration only.
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import received_signal, propagation_delays, propagation_distances


BANDWIDTHS_GHZ = (1.0, 4.0, 8.0)
R1_M, R2_M = 40.0, 60.0
THETA_DEG = 0.0
CENTER_ARRIVAL_GAP_PS = 160.0
RANGES_M = np.arange(30.0, 75.0 + 0.25, 0.5)
TIME_OVERSAMPLE = 4
SEED = int(getattr(SIM, "rng_seed", 0)) + 1
NOISELESS = False
OUTPUT = HERE / "bandwidth_two_source_hrt.png"


def signal_at_bandwidth(time_axis, antenna_x, bandwidth_ghz, tau1):
    """Two sources, exact spherical propagation, equal energy per source.

The repo Gaussian convention is sigma = 1/(sqrt(2)*pi*B). Multiplying the
unit-peak Gaussian by sqrt(sigma_8/sigma_B) preserves its continuous-time
energy. The usual 1/distance free-space amplitude remains in place.
    """
    bandwidth = bandwidth_ghz * 1e9
    sigma = 1.0 / (np.sqrt(2.0) * np.pi * bandwidth)
    sigma_ref = 1.0 / (np.sqrt(2.0) * np.pi * SYSTEM.bandwidth)
    amplitude = np.sqrt(sigma_ref / sigma)
    offset2 = tau1 + CENTER_ARRIVAL_GAP_PS * 1e-12 - R2_M / C
    signal = np.zeros((len(antenna_x), len(time_axis)), dtype=np.complex128)
    for r, offset in ((R1_M, 0.0), (R2_M, offset2)):
        arrivals = propagation_delays(
            r, THETA_DEG, clock_offset=offset, antenna_x=antenna_x
        )
        # Evaluate the existing received_signal model at each antenna's own
        # arrival. Its peak contains the repo's path loss and carrier phase.
        peak_record, _ = received_signal(
            arrivals, r, THETA_DEG,
            clock_offset=offset, antenna_x=antenna_x,
        )
        peak = peak_record[np.arange(len(antenna_x)), np.arange(len(antenna_x))]
        envelope = np.exp(-0.5 * ((time_axis[None, :] - arrivals[:, None]) / sigma) ** 2)
        signal += amplitude * peak[:, None] * envelope
    return signal


def hrt_range_time_map(data, time_axis, antenna_x, ranges):
    """Exact spherical HRT with frequency-domain fractional delays.

Output time is the array-center arrival tau. This is a single theta=0 slice
of HRT; it does not use either source's pulse shape or arrival-time hint.
    """
    dt = float(time_axis[1] - time_axis[0])
    delta_max = max(
        np.max(np.abs((propagation_distances(r, THETA_DEG, antenna_x) - r) / C))
        for r in (float(ranges[0]), float(ranges[-1]))
    )
    guard = int(np.ceil(delta_max / dt)) + 12
    nt = len(time_axis)
    nfft = next_fast_len(nt + 2 * guard)
    padded = np.pad(data, ((0, 0), (guard, nfft - nt - guard)))
    spectrum = np.fft.fft(padded, axis=1)
    freq = np.fft.fftfreq(nfft, d=dt)
    f0 = float(getattr(SYSTEM, "f0", SYSTEM.fc))
    scores = np.empty((len(ranges), (nt - 1) * TIME_OVERSAMPLE + 1))

    for index, r in enumerate(ranges):
        distance = propagation_distances(float(r), THETA_DEG, antenna_x)
        delta = (distance - r) / C
        relative_amplitude = r / distance
        weight = relative_amplitude / np.linalg.norm(relative_amplitude)
        aligned_spectrum = np.sum(
            weight[:, None] * spectrum
            * np.exp(2j * np.pi * (f0 + freq[None, :]) * delta[:, None]),
            axis=0,
        )
        if TIME_OVERSAMPLE == 1:
            aligned = np.fft.ifft(aligned_spectrum)
        else:
            centered = np.fft.fftshift(aligned_spectrum)
            left = (nfft * TIME_OVERSAMPLE) // 2 - nfft // 2
            right = nfft * (TIME_OVERSAMPLE - 1) - left
            enlarged = np.pad(centered, (left, right))
            aligned = TIME_OVERSAMPLE * np.fft.ifft(np.fft.ifftshift(enlarged))
        start = guard * TIME_OVERSAMPLE
        scores[index] = np.abs(
            aligned[start:start + scores.shape[1]]
        ) ** 2

    tau_axis = time_axis[0] + np.arange(scores.shape[1]) * dt / TIME_OVERSAMPLE
    return scores, tau_axis


def main():
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    dt = float(SYSTEM.dt)  # Same receiver clock and sample rate for every B.
    tau1 = R1_M / C
    sigma_widest = 1.0 / (np.sqrt(2.0) * np.pi * min(BANDWIDTHS_GHZ) * 1e9)
    margin = 7.0 * sigma_widest
    n0 = int(np.floor((tau1 - margin) / dt))
    n1 = int(np.ceil((tau1 + CENTER_ARRIVAL_GAP_PS * 1e-12 + margin) / dt))
    time_axis = np.arange(n0, n1 + 1) * dt

    # Fixed per-sample complex noise variance: 8 GHz unit-peak Gaussian at
    # the reference range has the repository's reference peak SNR.
    ref_r = float(getattr(SIM, "reference_range", 300.0))
    ref_snr_db = float(getattr(SIM, "reference_snr_db", 10.0))
    ref_arrival = propagation_delays(
        ref_r, 0.0, antenna_x=x
    )[len(x) // 2]
    reference, _ = received_signal(
        np.array([ref_arrival]), ref_r, 0.0, antenna_x=x
    )
    variance = float(abs(reference[len(x) // 2, 0]) ** 2) / 10.0 ** (ref_snr_db / 10.0)
    rng = np.random.default_rng(SEED)
    noise = np.zeros((len(x), len(time_axis)), complex) if NOISELESS else (
        np.sqrt(variance / 2.0)
        * (rng.standard_normal((len(x), len(time_axis)))
           + 1j * rng.standard_normal((len(x), len(time_axis))))
    )

    fig, axes = plt.subplots(
        1, len(BANDWIDTHS_GHZ), figsize=(16.5, 5.2),
        sharex=True, sharey=True, layout="constrained",
    )
    for ax, bandwidth_ghz in zip(axes, BANDWIDTHS_GHZ):
        started = time.perf_counter()
        raw = signal_at_bandwidth(time_axis, x, bandwidth_ghz, tau1) + noise
        score, tau = hrt_range_time_map(raw, time_axis, x, RANGES_M)
        tau_ps = (tau - tau1) * 1e12
        keep = (tau_ps >= -150.0) & (tau_ps <= 340.0)
        relative_db = 10.0 * np.log10(
            np.maximum(score[:, keep] / score.max(), 1e-15)
        )
        im = ax.imshow(
            relative_db.T, origin="lower", aspect="auto", cmap="magma",
            vmin=-15, vmax=0,
            extent=(RANGES_M[0], RANGES_M[-1], tau_ps[keep][0], tau_ps[keep][-1]),
        )
        ax.plot(R1_M, 0.0, "x", ms=10, mew=2.5, color="#22dd99",
                label=f"Source 1 truth: {R1_M:g} m, 0 ps")
        ax.plot(R2_M, CENTER_ARRIVAL_GAP_PS, "x", ms=10, mew=2.5,
                color="#33aaff",
                label=f"Source 2 truth: {R2_M:g} m, {CENTER_ARRIVAL_GAP_PS:g} ps")
        ax.set_title(f"{bandwidth_ghz:g} GHz", fontsize=25)
        ax.set_xlabel("Candidate range (m)", fontsize=20)
        ax.tick_params(labelsize=13)
        ax.set_xlim(30, 75)
        ax.set_ylim(-150, 340)
        print(f"B={bandwidth_ghz:g} GHz: {time.perf_counter()-started:.2f} s", flush=True)

    axes[0].set_ylabel("Center arrival relative to source 1 (ps)", fontsize=20)
    colorbar = fig.colorbar(im, ax=axes, pad=0.015, shrink=0.89)
    colorbar.set_label("dB below each panel's HRT maximum", fontsize=15)
    colorbar.ax.tick_params(labelsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=2, frameon=False, fontsize=15)
    fig.suptitle(r"Effect of Pulse Bandwidth on HRT Source Separation (Slice at $\theta=0^\circ$)", fontsize=25)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    print(f"Saved {OUTPUT}")
    print(f"Config: {ROOT / 'config.py'}; antennas={len(x)}, dt={dt*1e12:.3f} ps; "
          f"complex noise variance={variance:.3e}")
    print("Equal continuous-time pulse energy; both sources have the same "
          "positions and center arrival times in every panel.")
    plt.show()


if __name__ == "__main__":
    main()
