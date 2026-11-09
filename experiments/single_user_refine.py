
"""Random single-user evaluation using the exact multi_user_final SIC algorithm."""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from experiments import multi_user_final as multi_pipeline
from tqdm import trange
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn

# ==================== Settings ====================
N_MC = 500
R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
CLOCK_OFFSET = 10e-9

def polar_to_xy(R, theta):
    t = np.deg2rad(theta)
    return R * np.sin(t), R * np.cos(t)


def build_time_axis(R, theta, antenna_x):
    delays = propagation_delays(
        R, theta, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x
    )
    margin = 8 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


# ==================== Monte Carlo ====================
def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    errors = np.full(N_MC, np.nan)
    snrs = np.zeros(N_MC)
    positions = np.zeros((N_MC, 2))
    matched_trials = 0

    for i in trange(N_MC, desc="Single user: multi-user pipeline"):

        # 1. Random user
        R_true = rng_pos.uniform(R_MIN, R_MAX)
        theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
        tau_true = R_true / C + CLOCK_OFFSET
        x_true, y_true = polar_to_xy(R_true, theta_true)

        # 2. Received signal + AWGN
        t = build_time_axis(R_true, theta_true, antenna_x)

        clean_rx, _ = received_signal(
            t, R_true, theta_true,
            clock_offset=CLOCK_OFFSET, antenna_x=antenna_x
        )
        noisy_rx, _ = add_awgn(clean_rx, rng=rng_noise)

        # Use the same assisted search, candidates, joint Powell and SIC stopping
        # as multi_user_final, without informing the detector to stop at one user.
        previous_users = multi_pipeline.USERS
        try:
            multi_pipeline.USERS = [{
                "range_m": R_true, "theta_deg": theta_true,
                "tau_ns": tau_true * 1e9,
            }]
            detected, _ = multi_pipeline.run_sic(noisy_rx, t, antenna_x)
        finally:
            multi_pipeline.USERS = previous_users

        positions[i] = [x_true, y_true]
        if len(detected) == 1:
            errors[i] = np.hypot(
                detected[0]["x_m"] - x_true, detected[0]["y_m"] - y_true
            )
            matched_trials += 1
        snrs[i] = SIM.reference_snr_db + 20 * np.log10(
            SIM.reference_range / R_true
        )

    # ==================== Statistics ====================
    valid = np.isfinite(errors)
    errors = errors[valid]
    positions = positions[valid]
    ALE = float(np.mean(errors)) if errors.size else np.nan
    RMSE = float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan
    MAX_ERROR = float(np.max(errors)) if errors.size else np.nan

    print("\n" + "=" * 55)
    print("Single User Localization: multi_user_final algorithm")
    print("=" * 55)
    print(f"Monte Carlo trials : {N_MC}")
    print(f"Antenna stride     : {multi_pipeline.ANTENNA_STRIDE}")
    print(f"Time stride        : {multi_pipeline.TIME_STRIDE}")
    print("Amplitude gain     : Fixed (alpha = 1)")
    print(f"Error samples      : {matched_trials}")
    print(f"Tx power           : {SIM.tx_power_dbm:.3f} dBm")
    print(f"SNR range          : {snrs.min():.3f} ~ {snrs.max():.3f} dB")
    print("-" * 55)
    print(f"ALE                : {ALE:.6f} m")
    print(f"RMSE               : {RMSE:.6f} m")
    print(f"Max error          : {MAX_ERROR:.6f} m")
    print("=" * 55)

    # ==================== Figure 1: CDF ====================
    x = np.sort(errors)
    y = np.arange(1, errors.size + 1) / errors.size if errors.size else np.array([])

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, lw=2, color="tab:blue")
    plt.xlabel("2-D Localization Error (m)")
    plt.ylabel("CDF")
    plt.title("Localization Error CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ==================== Figure 2: Spatial Error ====================
    fig, ax = plt.subplots(figsize=(8.6, 8.2))

    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=errors, s=42, cmap="viridis",
        edgecolors="none"
    )

    xmax = R_MAX * np.sin(
        np.deg2rad(max(abs(THETA_MIN), abs(THETA_MAX)))
    )
    ax.set_xlim(-xmax - 25, xmax + 25)
    ax.set_ylim(0, R_MAX + 25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Spatial Localization Error: multi_user_final algorithm")
    ax.grid(True, alpha=0.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Absolute Error (m)")

    ax.text(
        0.02, 0.98,
        f"Samples={matched_trials}\nALE={ALE:.3f} m\nRMSE={RMSE:.3f} m",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(
            boxstyle="round", facecolor="white",
            alpha=0.9, edgecolor="gray"
        )
    )

    plt.show()


if __name__ == "__main__":
    main()
