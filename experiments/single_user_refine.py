
"""Single-user localization: PRT -> ROI -> HRT -> Downsampled Powell."""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import trange
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from localization.pipeline import localize

# ==================== Settings ====================
N_MC = 2000
R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
CLOCK_OFFSET = 10e-9

ANTENNA_STRIDE = 2
TIME_STRIDE = 2

POWELL_DR = 2.0
POWELL_DTHETA = 0.02
POWELL_DTAU = SYSTEM.dt
POWELL_MAXITER = 100
POWELL_FTOL = 1e-10
POWELL_XTOL = 1e-6


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


def reconstruct(t, x, R, theta, tau):
    rx, _ = received_signal(
        t, float(R), float(theta),
        clock_offset=float(tau) - float(R) / C,
        antenna_x=x
    )
    return rx


def residual_energy(rx, t, x, R, theta, tau):
    e = rx - reconstruct(t, x, R, theta, tau)
    return float(np.vdot(e, e).real)


# ==================== Powell Refinement ====================
def powell_refine(rx, t, antenna_x, R0, theta0, tau0):

    # Downsampling for optimization
    x_sub = antenna_x[::ANTENNA_STRIDE]
    t_sub = t[::TIME_STRIDE]
    rx_sub = rx[::ANTENNA_STRIDE, ::TIME_STRIDE]

    bounds = [
        (max(R_MIN, R0 - POWELL_DR),
         min(R_MAX, R0 + POWELL_DR)),
        (max(THETA_MIN, theta0 - POWELL_DTHETA),
         min(THETA_MAX, theta0 + POWELL_DTHETA)),
        (-POWELL_DTAU * 1e9, POWELL_DTAU * 1e9)
    ]

    x0 = np.array([
        np.clip(R0, *bounds[0]),
        np.clip(theta0, *bounds[1]),
        0.0
    ])

    scale = max(np.vdot(rx_sub, rx_sub).real, 1e-30)

    def objective(z):
        R, theta, dtau_ns = z
        tau = tau0 + dtau_ns * 1e-9
        return residual_energy(
            rx_sub, t_sub, x_sub, R, theta, tau
        ) / scale

    result = minimize(
        objective, x0,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": POWELL_MAXITER,
            "ftol": POWELL_FTOL,
            "xtol": POWELL_XTOL
        }
    )

    # Full-resolution validation
    E0 = residual_energy(rx, t, antenna_x, R0, theta0, tau0)

    if np.all(np.isfinite(result.x)):
        R1, theta1, dtau_ns = result.x
        tau1 = tau0 + dtau_ns * 1e-9
        E1 = residual_energy(rx, t, antenna_x, R1, theta1, tau1)

        if np.isfinite(E1) and E1 < E0:
            return float(R1), float(theta1), float(tau1), True

    return float(R0), float(theta0), float(tau0), False


# ==================== Monte Carlo ====================
def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    errors = np.zeros(N_MC)
    snrs = np.zeros(N_MC)
    positions = np.zeros((N_MC, 2))
    accepted = 0

    for i in trange(N_MC, desc="PRT -> ROI -> HRT -> Powell"):

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

        # 3. PRT -> ROI -> HRT
        result = localize(
            noisy_rx, t, antenna_x, theta_true, tau_true
        )

        # 4. Downsampled Powell (fixed amplitude)
        R_hat, theta_hat, tau_hat, success = powell_refine(
            noisy_rx, t, antenna_x,
            result.R, result.theta, result.tau
        )
        accepted += int(success)

        # 5. Final localization error
        x_hat, y_hat = polar_to_xy(R_hat, theta_hat)
        errors[i] = np.hypot(x_hat - x_true, y_hat - y_true)
        positions[i] = [x_true, y_true]

        snrs[i] = SIM.reference_snr_db + 20 * np.log10(
            SIM.reference_range / R_true
        )

    # ==================== Statistics ====================
    ALE = np.mean(errors)
    RMSE = np.sqrt(np.mean(errors**2))
    MAX_ERROR = np.max(errors)

    print("\n" + "=" * 55)
    print("Single User Localization: PRT + ROI + HRT + Powell")
    print("=" * 55)
    print(f"Monte Carlo trials : {N_MC}")
    print(f"Antenna stride     : {ANTENNA_STRIDE}")
    print(f"Time stride        : {TIME_STRIDE}")
    print("Amplitude gain     : Fixed (alpha = 1)")
    print(f"Powell accepted    : {accepted}/{N_MC}")
    print(f"Tx power           : {SIM.tx_power_dbm:.3f} dBm")
    print(f"SNR range          : {snrs.min():.3f} ~ {snrs.max():.3f} dB")
    print("-" * 55)
    print(f"ALE                : {ALE:.6f} m")
    print(f"RMSE               : {RMSE:.6f} m")
    print(f"Max error          : {MAX_ERROR:.6f} m")
    print("=" * 55)

    # ==================== Figure 1: CDF ====================
    x = np.sort(errors)
    y = np.arange(1, N_MC + 1) / N_MC

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
    ax.set_title("Spatial Localization Error: PRT + HRT + Powell")
    ax.grid(True, alpha=0.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Absolute Error (m)")

    ax.text(
        0.02, 0.98,
        f"N={N_MC}\nALE={ALE:.3f} m\nRMSE={RMSE:.3f} m",
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
