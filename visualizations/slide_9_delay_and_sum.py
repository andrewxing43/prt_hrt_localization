"""Slide visuals for candidate-based delay and sum without a pulse template.

1. Original unaligned baseband signals across antennas
2. Signals after fractional delay (envelope aligned, carrier phase mismatched)
3. Signals after carrier phase correction (coherent phase alignment)
4. Summed and normalized array output
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import resample


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / 'config.py').is_file() else HERE.parent
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal


RANGE_M = 300.0
THETA_DEG = 50.0
CLOCK_OFFSET_S = 0.0
OUTPUT_DIR = HERE / 'slide_delay_alignment_steps'
SHOW_PLOTS = True

# Visual clarity settings for single-antenna slide traces:
# - PLOT_SNR_DB: single-antenna SNR in dB (set to None to use SIM.reference_snr_db;
#   20 dB keeps realistic noise without burying individual antenna pulses in +-0.6 spikes)
# - UPSAMPLE_FACTOR: Fourier interpolation factor so narrow pulses sampled at SYSTEM.dt
#   render as smooth curves rather than 3-point triangles
PLOT_SNR_DB = None  # 20.0
UPSAMPLE_FACTOR = 4


def shift_traces(data, relative_delays):
    """Evaluate y_n(t + delta_n) by shifting each received trace in time."""
    guard = int(np.ceil(np.max(np.abs(relative_delays)) / SYSTEM.dt)) + 8
    nt = data.shape[1]
    nfft = next_fast_len(nt + 2 * guard)
    padded = np.zeros((len(relative_delays), nfft), dtype=np.complex128)
    padded[:, guard:guard + nt] = data
    frequencies = np.fft.fftfreq(nfft, d=SYSTEM.dt)
    shifted = np.fft.ifft(
        np.fft.fft(padded, axis=1)
        * np.exp(2j * np.pi * frequencies[None, :]
                 * relative_delays[:, None]),
        axis=1,
    )
    return shifted[:, guard:guard + nt]


def noise_variance_at_reference(antenna_x, snr_db_override=None):
    reference_range = float(getattr(SIM, 'reference_range', 300.0))
    if snr_db_override is not None:
        reference_snr_db = float(snr_db_override)
    else:
        reference_snr_db = float(getattr(SIM, 'reference_snr_db', 10.0))
    center = int(np.argmin(np.abs(antenna_x)))
    delays = propagation_delays(
        reference_range, 0.0, clock_offset=CLOCK_OFFSET_S,
        antenna_x=antenna_x,
    )
    reference, _ = received_signal(
        np.array([delays[center]]), reference_range, 0.0,
        clock_offset=CLOCK_OFFSET_S, antenna_x=antenna_x,
    )
    return float(abs(reference[center, 0])**2 / 10**(reference_snr_db / 10))


def select_illustrative_antennas(antenna_x, center, relative_delays):
    """Pick 5 antennas near the array center with clear arrival separation and strong I-channel contrast.

    Avoids two pitfalls of uniform full-array spacing:
    1. Outer antennas arrive at +-4 ns, outside a tight zoom window around the narrow pulse.
    2. Some antennas land near carrier phase +-90 deg (cos(phi) ~ 0), hiding their pulse
       energy in the imaginary (Q) channel when plotting the real part.
    For offsets with positive arrival delay (t > 0, under the upper-right legend), negative
    real peaks are preferred so pulses point downward and never collide with the legend box.
    """
    nx = len(antenna_x)
    if center < 28 or center + 28 >= nx:
        step = max(1, nx // 6)
        ants = [
            max(0, center - 2 * step),
            max(0, center - step),
            center,
            min(nx - 1, center + step),
            min(nx - 1, center + 2 * step),
        ]
        return sorted(list(dict.fromkeys(ants)))

    target_offsets = [-24, -12, 0, 12, 24]
    target_cos = [-0.95, -0.75, 1.0, -0.55, 0.85]
    selected = []
    fc = float(SYSTEM.fc)
    for off, t_c in zip(target_offsets, target_cos):
        if off == 0:
            selected.append(center)
        else:
            candidates = np.arange(max(0, center + off - 3), min(nx, center + off + 4))
            phases = np.cos(2.0 * np.pi * fc * relative_delays[candidates])
            best = candidates[int(np.argmin(np.abs(phases - t_c)))]
            selected.append(int(best))
    return selected


def save(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220, facecolor='white')
    print(f'Saved {path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    antenna_x = np.asarray(ARRAY.positions(SYSTEM), dtype=float)
    center = int(np.argmin(np.abs(antenna_x)))
    center_arrival = RANGE_M / C + CLOCK_OFFSET_S
    arrivals = propagation_delays(
        RANGE_M, THETA_DEG, clock_offset=CLOCK_OFFSET_S,
        antenna_x=antenna_x,
    )
    margin = 8 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((arrivals.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((arrivals.max() + margin) / SYSTEM.dt))
    time = np.arange(n0, n1 + 1) * SYSTEM.dt
    # Record extra noise-only samples beyond the displayed time window so
    # time shifts do not leave artificial zero-filled wedges.
    extra = int(np.ceil(np.max(np.abs(arrivals - center_arrival))
                        / SYSTEM.dt)) + 16
    extended_time = np.arange(n0 - extra, n1 + extra + 1) * SYSTEM.dt
    extended_clean, _ = received_signal(
        extended_time, RANGE_M, THETA_DEG,
        clock_offset=CLOCK_OFFSET_S, antenna_x=antenna_x,
    )
    variance = noise_variance_at_reference(antenna_x, snr_db_override=PLOT_SNR_DB)
    rng = np.random.default_rng(int(getattr(SIM, 'rng_seed', 0)) + 1)
    extended_noisy = extended_clean + np.sqrt(variance / 2) * (
        rng.standard_normal(extended_clean.shape)
        + 1j * rng.standard_normal(extended_clean.shape)
    )

    # Candidate delays relative to center antenna arrival
    relative_delays = arrivals - center_arrival
    display = slice(extra, extra + len(time))
    clean = extended_clean[:, display]
    noisy = extended_noisy[:, display]
    shifted = shift_traces(extended_noisy, relative_delays)[:, display]
    phase_correction = np.exp(
        2j * np.pi * float(SYSTEM.fc) * relative_delays[:, None]
    )
    phase_only = noisy * phase_correction
    aligned = shifted * phase_correction
    t_ns = (time - center_arrival) * 1e9
    display_scale = float(np.max(np.abs(clean[center])))

    # Select 5 antennas near the center and auto-scale the time window to their arrival spread
    ants_to_plot = select_illustrative_antennas(antenna_x, center, relative_delays)
    max_sel_delay_ns = float(np.max(np.abs(relative_delays[ants_to_plot]))) * 1e9
    pulse_margin_ns = 6.0 * float(SYSTEM.gaussian_sigma) * 1e9
    xlim_ns = (-(max_sel_delay_ns + pulse_margin_ns), +(max_sel_delay_ns + pulse_margin_ns))

    # Align center antenna carrier phase at pulse peak to positive real for clean presentation
    center_peak = int(np.argmax(abs(aligned[center])))
    common_rotation = np.exp(-1j * np.angle(aligned[center, center_peak]))

    # Extract window and Fourier-upsample so narrow pulses look smooth rather than jagged
    mask = (t_ns >= xlim_ns[0] - 0.1) & (t_ns <= xlim_ns[1] + 0.1)
    t_win = t_ns[mask]
    n_up = len(t_win) * UPSAMPLE_FACTOR
    t_up = np.linspace(t_win[0], t_win[-1], n_up)

    # Cohesive color map for tracking the same antennas across all steps
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(ants_to_plot)))

    # -------------------------------------------------------------------------
    # Step 1: Original Unaligned Baseband Signals (Real Part)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    for a, color in zip(ants_to_plot, colors):
        trace_up = resample(np.real(noisy[a, mask] * common_rotation) / display_scale, n_up)
        ax.plot(
            t_up,
            trace_up,
            color=color,
            lw=2.0,
            label=f'Ant {a} (x = {antenna_x[a]:.2f} m)',
        )
    ax.set_title('1. Receive Unaligned Baseband Signals (Real Part)', fontsize=25)
    ax.set_xlabel('Relative Time (ns)', fontsize=20)
    ax.set_ylabel('Real part / peak amplitude', fontsize=20)
    ax.set_xlim(xlim_ns)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=15, framealpha=0.92)
    save(fig, 'step_1_unaligned_signals.png')

    # -------------------------------------------------------------------------
    # Step 2: Signals After Shifting Traces in Time (Real Part)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12,5), layout='constrained')
    for a, color in zip(ants_to_plot, colors):
        trace_up = resample(np.real(shifted[a, mask] * common_rotation) / display_scale, n_up)
        ax.plot(
            t_up,
            trace_up,
            color=color,
            lw=2.0,
            label=f'Ant {a} (x = {antenna_x[a]:.2f} m)',
        )
    ax.set_title('2. Apply Candidate Shift in Time (Real Part)', fontsize=25)
    ax.set_xlabel('Relative Time (ns)', fontsize=20)
    ax.set_ylabel('Real part / peak amplitude', fontsize=20)
    ax.set_xlim(xlim_ns)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=15, framealpha=0.92)
    save(fig, 'step_2_fractional_delay.png')

    # -------------------------------------------------------------------------
    # Step 3: Signals After Carrier Phase Correction (Real Part)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12,5), layout='constrained')
    for a, color in zip(ants_to_plot, colors):
        trace_up = resample(np.real(aligned[a, mask] * common_rotation) / display_scale, n_up)
        ax.plot(
            t_up,
            trace_up,
            color=color,
            lw=2.0,
            label=f'Ant {a} (x = {antenna_x[a]:.2f} m)',
        )
    ax.set_title('3. Apply Carrier Phase Correction (Real Part)', fontsize=25)
    ax.set_xlabel('Relative Time (ns)', fontsize=20)
    ax.set_ylabel('Real part / peak amplitude', fontsize=20)
    ax.set_xlim(xlim_ns)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=15, framealpha=0.92)
    save(fig, 'step_3_carrier_phase.png')

    # -------------------------------------------------------------------------
    # Step 4: Summed and Normalized Array Output (Real Part)
    # -------------------------------------------------------------------------
    sum_shifted_up = resample(
        np.real(shifted[:, mask].mean(axis=0) * common_rotation) / display_scale, n_up
    )
    sum_phase_only_up = resample(
        np.real(phase_only[:, mask].mean(axis=0) * common_rotation) / display_scale, n_up
    )
    sum_aligned_up = resample(
        np.real(aligned[:, mask].mean(axis=0) * common_rotation) / display_scale, n_up
    )

    fig, ax = plt.subplots(figsize=(12,5), layout='constrained')
    ax.plot(
        t_up,
        sum_shifted_up,
        color='#617f99',
        lw=1.8,
        ls='--',
        label='Time shift only',
    )
    ax.plot(
        t_up,
        sum_phase_only_up,
        color='#2a9d8f',
        lw=2.0,
        ls='-.',
        label='Phase correction only',
    )
    ax.plot(
        t_up,
        sum_aligned_up,
        color='#d65539',
        lw=2.2,
        label='Time shift + phase correction',
    )
    ax.set_title('4. Average Over Antennas and Search Time (Real Part)', fontsize=25)
    ax.set_xlabel('Relative Time (ns)', fontsize=20)
    ax.set_ylabel('Real part / peak amplitude', fontsize=20)
    ax.set_xlim(xlim_ns)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=15, framealpha=0.92)
    save(fig, 'step_4_array_sum.png')

    print(f'Generated slide figures successfully in {OUTPUT_DIR}')
    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()