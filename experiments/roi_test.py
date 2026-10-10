from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import range_theta_to_pq, pq_to_range, p_to_theta_deg

N_POSITIONS = 500

R_MIN, R_MAX = 50.0, 1000.0
THETA_MIN, THETA_MAX = -60.0, 60.0

P_STEP_DEG, N_LOCAL_P, N_LOCAL_TAU = PRT.theta_step_deg, 4, 4
DQ = PRT.dq
K = 6 #k=4.5 if R_max=400
DELTA_THETA_MAX_DEG = 0.2

P_GRID_MIN_DEG, P_GRID_MAX_DEG = -65.0, 65.0
Q_SEARCH_R_MIN, Q_SEARCH_R_MAX, Q_SEARCH_THETA_MAX = 40.0, 1100.0, 65.0

THETA_GRID = np.arange(P_GRID_MIN_DEG, P_GRID_MAX_DEG + 0.5 * P_STEP_DEG, P_STEP_DEG)
P_GRID = -np.sin(np.deg2rad(THETA_GRID)) / C

Q_GLOBAL_MIN = np.cos(np.deg2rad(Q_SEARCH_THETA_MAX))**2 / (2 * Q_SEARCH_R_MAX * C)
Q_GLOBAL_MAX = 1.0 / (2 * Q_SEARCH_R_MIN * C)
Q_GRID = np.arange(Q_GLOBAL_MIN, Q_GLOBAL_MAX + DQ, DQ)


def build_time_axis(R, theta, antenna_x):
    dt = PRT.tau_step
    delays = propagation_delays(R, theta, antenna_x=antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / dt))
    n1 = int(np.ceil((delays.max() + margin) / dt))
    return np.arange(n0, n1 + 1) * dt


def get_local_p_grid(theta_true):
    idx = np.searchsorted(THETA_GRID, theta_true)
    start = np.clip(idx - N_LOCAL_P // 2, 0, len(THETA_GRID) - N_LOCAL_P)
    indices = np.arange(start, start + N_LOCAL_P)
    return THETA_GRID[indices], P_GRID[indices]


def get_local_q_grid(theta_local):
    cos2 = np.cos(np.deg2rad(theta_local))**2
    q_min = np.min(cos2) / (2 * R_MAX * C) - DQ
    q_max = np.max(cos2) / (2 * R_MIN * C) + DQ
    return Q_GRID[(Q_GRID >= q_min) & (Q_GRID <= q_max)]


def get_tau_indices(time_axis, tau_true):
    idx = np.searchsorted(time_axis, tau_true)
    start = np.clip(idx - N_LOCAL_TAU // 2, 0, len(time_axis) - N_LOCAL_TAU)
    return np.arange(start, start + N_LOCAL_TAU)


def find_peak(prt, time_axis, p_local, q_local, tau_true):
    tau_idx = get_tau_indices(time_axis, tau_true)
    local = np.abs(prt[tau_idx])
    i = np.unravel_index(np.argmax(local), local.shape)
    return time_axis[tau_idx[i[0]]], p_local[i[1]], q_local[i[2]]


def roi_radius_R(R_hat, theta_hat):
    theta_rad = np.deg2rad(theta_hat)
    dtheta = np.deg2rad(DELTA_THETA_MAX_DEG)
    dq_eff = K * DQ
    w_q = 2 * C * R_hat**2 / np.cos(theta_rad)**2 * dq_eff
    w_theta = 2 * R_hat * abs(np.tan(theta_rad)) * dtheta
    return w_q + w_theta


def main():
    antenna_x = ARRAY.positions(SYSTEM)

    rng_pos = np.random.default_rng()
    rng_noise = np.random.default_rng()

    records = []

    print(f"Positions={N_POSITIONS}, K={K}, Delta q={DQ:.1e}")
    print(f"Tx power={SIM.tx_power_dbm:.3f} dBm, thermal noise={10*np.log10(SIM.noise_power/1e-3):.3f} dBm")

    for _ in tqdm(range(N_POSITIONS), desc="Validating ROI", unit="pos"):
        R_true = rng_pos.uniform(R_MIN, R_MAX)
        theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
        tau_true = R_true / C

        _, q_true = range_theta_to_pq(R_true, theta_true)
        q_true = float(q_true)

        theta_local, p_local = get_local_p_grid(theta_true)
        q_local = get_local_q_grid(theta_local)

        time_axis = build_time_axis(R_true, theta_true, antenna_x)
        clean_rx, _ = received_signal(time_axis, R_true, theta_true, antenna_x=antenna_x)

        noisy_rx, _ = add_awgn(clean_rx, rng=rng_noise)
        mf_rx = matched_filter(noisy_rx)

        prt = parabolic_radon_transform(mf_rx, antenna_x, p_local, q_local)
        tau_hat, p_hat, q_hat = find_peak(prt, time_axis, p_local, q_local, tau_true)

        R_hat = float(pq_to_range(p_hat, q_hat))
        theta_hat = float(p_to_theta_deg(p_hat))

        R_error = abs(R_hat - R_true)
        Kq = abs(q_hat - q_true) / DQ

        W_R = roi_radius_R(R_hat, theta_hat)
        margin = W_R - R_error
        covered = margin >= 0

        records.append({
            "R_true": R_true,
            "theta_true": theta_true,
            "R_hat": R_hat,
            "theta_hat": theta_hat,
            "q_true": q_true,
            "q_hat": q_hat,
            "Kq": Kq,
            "R_error": R_error,
            "W_R": W_R,
            "margin": margin,
            "covered": covered,
            "theta_error": abs(theta_hat - theta_true),
            "tau_error": abs(tau_hat - tau_true)
        })

    covered = np.array([x["covered"] for x in records])
    margins = np.array([x["margin"] for x in records])
    R_true = np.array([x["R_true"] for x in records])
    theta_true = np.array([x["theta_true"] for x in records])
    R_error = np.array([x["R_error"] for x in records])
    W_R = np.array([x["W_R"] for x in records])
    Kq = np.array([x["Kq"] for x in records])

    coverage = 100 * np.mean(covered)
    failures = np.sum(~covered)

    print("\n===== ROI VALIDATION =====")
    print(f"Coverage = {coverage:.3f}%")
    print(f"Failures = {failures}/{N_POSITIONS}")
    print(f"Max Kq = {np.max(Kq):.3f}")
    print(f"Mean ROI radius = {np.mean(W_R):.3f} m")
    print(f"Max ROI radius = {np.max(W_R):.3f} m")
    print(f"Worst margin = {np.min(margins):.3f} m")

    if failures > 0:
        idx = np.argmin(margins)
        print("\nWorst failure:")
        print(f"R_true={R_true[idx]:.3f} m, theta_true={theta_true[idx]:.3f} deg")
        print(f"R_error={R_error[idx]:.3f} m, ROI radius={W_R[idx]:.3f} m")
        print(f"margin={margins[idx]:.3f} m, Kq={Kq[idx]:.3f}")

    plt.figure(figsize=(9, 6))
    plt.scatter(R_true[covered], theta_true[covered], s=15, alpha=0.5, label="Covered")
    if failures > 0:
        plt.scatter(R_true[~covered], theta_true[~covered], s=45, marker="x", label="Failure")
    plt.xlabel("True range R (m)")
    plt.ylabel("True angle theta (deg)")
    plt.title(f"ROI Coverage, K={K}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(9, 5))
    plt.scatter(R_true, margins, s=12, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("True range R (m)")
    plt.ylabel("ROI margin W_R - |R_hat - R| (m)")
    plt.title("ROI Safety Margin versus Range")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(9, 5))
    plt.scatter(theta_true, margins, s=12, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("True angle theta (deg)")
    plt.ylabel("ROI margin W_R - |R_hat - R| (m)")
    plt.title("ROI Safety Margin versus Angle")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(7, 6))
    plt.scatter(W_R, R_error, s=12, alpha=0.5)
    lim = max(np.max(W_R), np.max(R_error))
    plt.plot([0, lim], [0, lim], "--")
    plt.xlabel("ROI radius W_R (m)")
    plt.ylabel("Actual absolute range error (m)")
    plt.title("ROI Radius versus Actual Range Error")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()