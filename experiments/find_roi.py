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

N_POSITIONS, N_MC = 200, 5

R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0

P_STEP_DEG, N_LOCAL_P, N_LOCAL_TAU = PRT.theta_step_deg, 4, 4
DQ = PRT.dq

P_GRID_MIN_DEG, P_GRID_MAX_DEG = -65.0, 65.0
Q_SEARCH_R_MIN, Q_SEARCH_R_MAX, Q_SEARCH_THETA_MAX = 40.0, 500.0, 65.0

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


def main():
    antenna_x = ARRAY.positions(SYSTEM)
    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)

    position_results = []

    print(f"Positions={N_POSITIONS}, MC/position={N_MC}, total={N_POSITIONS*N_MC}, Delta q={DQ:.1e}")
    print(f"Tx power={SIM.tx_power_dbm:.3f} dBm, thermal noise={10*np.log10(SIM.noise_power/1e-3):.3f} dBm")

    progress = tqdm(range(N_POSITIONS), desc="Positions", unit="pos")

    for pos in progress:
        R_true = rng_pos.uniform(R_MIN, R_MAX)
        theta_true = rng_pos.uniform(THETA_MIN, THETA_MAX)
        tau_true = R_true / C

        _, q_true = range_theta_to_pq(R_true, theta_true)
        q_true = float(q_true)

        theta_local, p_local = get_local_p_grid(theta_true)
        q_local = get_local_q_grid(theta_local)

        time_axis = build_time_axis(R_true, theta_true, antenna_x)
        clean_rx, _ = received_signal(time_axis, R_true, theta_true, antenna_x=antenna_x)

        Kq_list, Rerr_list, theta_err_list, tau_err_list = [], [], [], []

        for mc in range(N_MC):
            noisy_rx, _ = add_awgn(clean_rx, rng=rng_noise)
            mf_rx = matched_filter(noisy_rx)
            prt = parabolic_radon_transform(mf_rx, antenna_x, p_local, q_local)
            tau_hat, p_hat, q_hat = find_peak(prt, time_axis, p_local, q_local, tau_true)

            R_hat = float(pq_to_range(p_hat, q_hat))
            theta_hat = float(p_to_theta_deg(p_hat))

            Kq_list.append(abs(q_hat - q_true) / DQ)
            Rerr_list.append(abs(R_hat - R_true))
            theta_err_list.append(abs(theta_hat - theta_true))
            tau_err_list.append(abs(tau_hat - tau_true))

        Kq_list = np.array(Kq_list)
        Rerr_list = np.array(Rerr_list)

        result = {
            "R_true": R_true,
            "theta_true": theta_true,
            "Kq_mean": np.mean(Kq_list),
            "Kq95": np.percentile(Kq_list, 95),
            "Kq99": np.percentile(Kq_list, 99),
            "Kq_max": np.max(Kq_list),
            "Rerr99": np.percentile(Rerr_list, 99),
            "theta_err_max": np.max(theta_err_list),
            "tau_err_max": np.max(tau_err_list)
        }

        position_results.append(result)

        progress.set_postfix(R=f"{R_true:.1f}", theta=f"{theta_true:.1f}", Kq99=f"{result['Kq99']:.2f}")

    progress.close()

    R_pos = np.array([x["R_true"] for x in position_results])
    theta_pos = np.array([x["theta_true"] for x in position_results])
    Kq99_pos = np.array([x["Kq99"] for x in position_results])
    Rerr99_pos = np.array([x["Rerr99"] for x in position_results])

    print("\n===== POSITION-LEVEL Kq99 =====")
    print(f"Mean={np.mean(Kq99_pos):.3f}")
    print(f"P90={np.percentile(Kq99_pos,90):.3f}")
    print(f"P95={np.percentile(Kq99_pos,95):.3f}")
    print(f"P99={np.percentile(Kq99_pos,99):.3f}")
    print(f"Max={np.max(Kq99_pos):.3f}")

    print("\nCorresponding Delta q_eff:")
    print(f"P95={np.percentile(Kq99_pos,95)*DQ:.3e}")
    print(f"P99={np.percentile(Kq99_pos,99)*DQ:.3e}")

    plt.figure(figsize=(9, 5))
    plt.scatter(R_pos, Kq99_pos, s=30, alpha=0.7)
    plt.xlabel("True range R (m)")
    plt.ylabel(r"$K_{q,99}$")
    plt.title(r"$K_{q,99}$ versus user range")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(9, 5))
    plt.scatter(theta_pos, Kq99_pos, s=30, alpha=0.7)
    plt.xlabel("True angle theta (deg)")
    plt.ylabel(r"$K_{q,99}$")
    plt.title(r"$K_{q,99}$ versus user angle")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(R_pos, theta_pos, c=Kq99_pos, s=55)
    plt.colorbar(sc, label=r"$K_{q,99}$")
    plt.xlabel("True range R (m)")
    plt.ylabel("True angle theta (deg)")
    plt.title(r"Random user positions colored by $K_{q,99}$")
    plt.tight_layout()

    plt.figure(figsize=(9, 5))
    plt.scatter(R_pos, Rerr99_pos, s=30, alpha=0.7)
    plt.xlabel("True range R (m)")
    plt.ylabel("P99 absolute range error (m)")
    plt.title("P99 range error versus user range")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()