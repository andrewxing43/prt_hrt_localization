"""
250-run Monte Carlo test for PRT -> ROI -> HRT pipeline.

Each run:
- Random off-grid user
- R ~ U(50, 400) m
- theta ~ U(-60, 60) deg
- SNR = 10 dB
- Run current truth-assisted pipeline
- Record final 2-D localization error

Outputs:
- Progress bar
- Error CDF
- ALE
- RMSE
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from localization.pipeline import localize


N_MC = 250
R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
SNR_DB = 10.0
CLOCK_OFFSET = 10e-9


def build_time_axis(R, theta, clock_offset, antenna_x):
    delays = propagation_delays(R, theta, clock_offset=clock_offset, antenna_x=antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def polar_to_xy(R, theta_deg):
    t = np.deg2rad(theta_deg)
    return R * np.sin(t), R * np.cos(t)


def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)
    errors = np.zeros(N_MC)

    for i in trange(N_MC, desc="Monte Carlo"):
        R_true = rng_pos.uniform(R_MIN, R_MAX)
        theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
        tau_true = R_true / C + CLOCK_OFFSET
        x_true, y_true = polar_to_xy(R_true, theta_true)

        time_axis = build_time_axis(R_true, theta_true, CLOCK_OFFSET, antenna_x)
        clean_rx, _ = received_signal(time_axis, R_true, theta_true, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x)
        noisy_rx, _ = add_awgn(clean_rx, snr_db=SNR_DB, rng=rng_noise)

        result = localize(noisy_rx, time_axis, antenna_x, theta_true, tau_true)
        errors[i] = np.hypot(result.x - x_true, result.y - y_true)

    ALE = np.mean(errors)
    RMSE = np.sqrt(np.mean(errors**2))

    print("\n" + "=" * 50)
    print(f"N    : {N_MC}")
    print(f"ALE  : {ALE:.6f} m")
    print(f"RMSE : {RMSE:.6f} m")
    print("=" * 50)

    x = np.sort(errors)
    y = np.arange(1, N_MC + 1) / N_MC

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("2-D Localization Error (m)")
    plt.ylabel("CDF")
    plt.title("Localization Error Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()