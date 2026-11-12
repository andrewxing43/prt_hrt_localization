"""Slide figure: array curvature and frequency-dependent relative phase.

Run from the repository:
    python experiments/slide2_array_frequency_phase.py

The plot shows the ideal propagation phase relative to the center antenna.
It is independent of the transmitted waveform when all antennas observe the
same waveform and the occupied frequencies have nonzero received energy.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY
from signal_model.propagation import propagation_delays


RANGE_M = 50.0
THETA_DEG = 0.0
BANDWIDTH_HZ = 8e9
N_FREQUENCIES = 401
NARROW_BAND_HZ = 1e9  # Illustrative subband around f0.
OUTPUT = HERE / "slide2_array_frequency_phase.png"


def main():
    x = np.asarray(ARRAY.positions(SYSTEM), dtype=float)
    f0 = float(SYSTEM.fc)
    frequencies = np.linspace(f0 - BANDWIDTH_HZ / 2,
                              f0 + BANDWIDTH_HZ / 2, N_FREQUENCIES)
    delays = propagation_delays(RANGE_M, THETA_DEG, antenna_x=x)
    center = int(np.argmin(np.abs(x)))
    relative_delays = delays - delays[center]

    # The waveform and any common transmit-time phase cancel in Y_n(f)Y_0*(f).
    # Express phase in cycles; this continuous model is already unwrapped.
    phase_cycles = -frequencies[:, None] * relative_delays[None, :]
    frequencies_ghz = frequencies / 1e9
    f0_ghz = f0 / 1e9
    narrow_half_ghz = NARROW_BAND_HZ / 2e9

    plt.figure(figsize=(10,8))

    # plt.axvspan(f0_ghz - narrow_half_ghz,
    #                  f0_ghz + narrow_half_ghz, color="#e1e3e7",
    #                  label="Illustrative 1 GHz subband")
    for target_x, color, edge in ((0.8, "#167ab4", False), (float(x[-1]), "#dd6234", True)):
        antenna = int(np.argmin(np.abs(x - target_x)))
        label = rf"$x={x[antenna]:.2f}$ m (edge element)" if edge else rf"$x={x[antenna]:.2f}$ m"
        plt.plot(frequencies_ghz, phase_cycles[:, antenna], lw=2.7,
                      color=color, label=label)
    plt.axvline(f0_ghz, color="black", ls="--", lw=1.4, label="f0 = 24 GHz")
    plt.title(f"Spatial Phase Relative to Array Center\n"
              rf"($R={RANGE_M:g}$ m, $\theta={THETA_DEG:g}^\circ$)", fontsize=20)
    plt.xlabel("Frequency (GHz)", fontsize=15)
    plt.ylabel("Relative phase (cycles)", fontsize=15)
    plt.grid(True, alpha=0.22)
    plt.legend(loc="lower left", fontsize=15, framealpha=0.96)

    plt.savefig(OUTPUT, dpi=220, facecolor="white")
    print(f"Saved {OUTPUT}")
    print(f"Aperture={np.ptp(x):.3f} m; center-to-edge delay="
          f"{relative_delays[-1]*1e12:.2f} ps; "
          f"edge phase change over band="
          f"{BANDWIDTH_HZ*relative_delays[-1]:.3f} cycles")
    plt.show()


if __name__ == "__main__":
    main()
