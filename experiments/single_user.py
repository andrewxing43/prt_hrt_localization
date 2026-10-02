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
- True user position map with color = absolute error
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


N_MC = 500
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


def draw_user_region(ax, rmin=R_MIN, rmax=R_MAX, thmin=THETA_MIN, thmax=THETA_MAX):
    th = np.deg2rad(np.linspace(thmin, thmax, 600))
    for r, lw, ls, a in [(rmin, 1.5, "--", 0.9), (rmax, 2.0, "-", 1.0)]:
        ax.plot(r * np.sin(th), r * np.cos(th), color="k", lw=lw, ls=ls, alpha=a)

    for th_deg, lw, ls, a in [(thmin, 1.5, "--", 0.9), (thmax, 1.5, "--", 0.9), (0.0, 1.0, ":", 0.7)]:
        t = np.deg2rad(th_deg)
        rr = np.linspace(rmin, rmax, 300)
        ax.plot(rr * np.sin(t), rr * np.cos(t), color="k", lw=lw, ls=ls, alpha=a)

    for r in [100, 200, 300]:
        ax.plot(r * np.sin(th), r * np.cos(th), color="gray", lw=0.8, ls=":", alpha=0.5)

    ax.text(rmax * np.sin(np.deg2rad(thmin)) - 18, rmax * np.cos(np.deg2rad(thmin)) - 5, f"{thmin:.0f}°", fontsize=10)
    ax.text(rmax * np.sin(np.deg2rad(thmax)) + 4,  rmax * np.cos(np.deg2rad(thmax)) - 5, f"{thmax:.0f}°", fontsize=10)
    ax.text(0, rmax + 8, "0°", ha="center", fontsize=10)
    ax.text(8, rmin + 2, f"R={int(rmin)} m", fontsize=9)
    ax.text(8, rmax - 8, f"R={int(rmax)} m", fontsize=9)


def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    errors = np.zeros(N_MC)
    x_true_all = np.zeros(N_MC)
    y_true_all = np.zeros(N_MC)

    for i in trange(N_MC, desc="Monte Carlo"):
        R_true = rng_pos.uniform(R_MIN, R_MAX)
        theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
        tau_true = R_true / C + CLOCK_OFFSET
        x_true, y_true = polar_to_xy(R_true, theta_true)

        time_axis = build_time_axis(R_true, theta_true, CLOCK_OFFSET, antenna_x)
        clean_rx, _ = received_signal(time_axis, R_true, theta_true, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x)
        noisy_rx, _ = add_awgn(clean_rx, snr_db=SNR_DB, rng=rng_noise)

        result = localize(noisy_rx, time_axis, antenna_x, theta_true, tau_true)

        x_true_all[i], y_true_all[i] = x_true, y_true
        errors[i] = np.hypot(result.x - x_true, result.y - y_true)

    ALE = np.mean(errors)
    RMSE = np.sqrt(np.mean(errors**2))

    print("\n" + "=" * 50)
    print(f"N    : {N_MC}")
    print(f"ALE  : {ALE:.6f} m")
    print(f"RMSE : {RMSE:.6f} m")
    print("=" * 50)

    # CDF
    x = np.sort(errors)
    y = np.arange(1, N_MC + 1) / N_MC

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("2-D Localization Error (m)")
    plt.ylabel("CDF")
    plt.title("Localization Error Distribution")
    plt.grid(True)
    plt.tight_layout()

    # Position-error map
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    sc = ax.scatter(x_true_all, y_true_all, c=errors, s=42, cmap="turbo", edgecolors="k", linewidths=0.25)
    draw_user_region(ax)

    xmax = R_MAX * np.sin(np.deg2rad(max(abs(THETA_MIN), abs(THETA_MAX))))
    ax.set_xlim(-xmax - 25, xmax + 25)
    ax.set_ylim(0, R_MAX + 25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("True User Positions Colored by Absolute Localization Error")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Absolute Error (m)")

    txt = f"N={N_MC}\nALE={ALE:.3f} m\nRMSE={RMSE:.3f} m"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()