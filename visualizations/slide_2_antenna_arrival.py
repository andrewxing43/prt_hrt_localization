"""Slide 2 visual: one wideband pulse sampled across the receiving ULA.

Run from the repository with:
    python experiments/slide2_system_model_visual.py

The main panel shows the magnitude of the noiseless *raw* complex received
signal, before averaging or matched filtering. The inset uses the same data
at three array positions. The finer time axis is only for drawing the figure;
it does not change the propagation model or an estimator's sample rate.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY
from signal_model.propagation import propagation_delays, received_signal


# A near source at broadside makes spherical curvature easy to see. This is a
# pedagogical noiseless example, not a Monte Carlo localization result.
RANGE_M = 50.0
THETA_DEG = 0.0
CLOCK_OFFSET_S = 0.0
VISUAL_TIME_STEP_PS = 1.0
OUTPUT = HERE / "slide2_antenna_time_record.png"


def main():
    antenna_x = np.asarray(ARRAY.positions(SYSTEM), dtype=float)
    delays = propagation_delays(
        RANGE_M, THETA_DEG,
        clock_offset=CLOCK_OFFSET_S, antenna_x=antenna_x,
    )
    center_arrival = RANGE_M / C + CLOCK_OFFSET_S
    sigma = SYSTEM.gaussian_sigma

    # Exact continuous-time model sampled densely for visualization only.
    t0 = delays.min() - 4.5 * sigma
    t1 = delays.max() + 4.5 * sigma
    time = np.arange(t0, t1 + VISUAL_TIME_STEP_PS * 1e-12,
                     VISUAL_TIME_STEP_PS * 1e-12)
    clean, _ = received_signal(
        time, RANGE_M, THETA_DEG,
        clock_offset=CLOCK_OFFSET_S, antenna_x=antenna_x,
    )
    magnitude = np.abs(clean)
    magnitude /= np.maximum(magnitude.max(axis=1, keepdims=True), 1e-30)
    relative_time_ps = (time - center_arrival) * 1e12
    relative_delay_ps = (delays - center_arrival) * 1e12

    plt.figure(figsize=(12,10))
    im = plt.imshow(
        magnitude, origin="lower", aspect="auto", cmap="Blues",
        interpolation="nearest",
        extent=(relative_time_ps[0], relative_time_ps[-1],
                antenna_x[0], antenna_x[-1]),
        vmin=0, vmax=1,
    )
    plt.plot(relative_delay_ps, antenna_x, color="#e65f2b",
                lw=2.3, label="Spherical Arrival Time")
    plt.axvline(0, color="#57466e", ls="--", lw=2,
                   label="Plane-Wave Approx. \nat Broadside")
    plt.xlabel("Time relative to center arrival (ps)", fontsize=20)
    plt.ylabel("Antenna position $x$ (m)", fontsize=20)

    plt.legend(loc="center right", fontsize=15, framealpha=0.95)
    plt.colorbar(im, ax=im.axes, label="Normalized envelope magnitude",
             pad=0.02, fraction=0.048)

    plt.suptitle(
        f"Antenna Position vs. Source Waveform Arrival Time\n",
        fontsize=25,
    )
    plt.title(rf"($R={RANGE_M:g}\,\mathrm{{m}},\ \theta={THETA_DEG:g}^\circ$, Noiseless)",fontsize=25)

    plt.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT}")
    print(f"Aperture = {np.ptp(antenna_x):.3f} m; "
          f"center-to-edge arrival difference = "
          f"{relative_delay_ps[-1]:.2f} ps; "
          f"pulse sigma = {sigma * 1e12:.2f} ps")
    plt.show()


if __name__ == "__main__":
    main()
