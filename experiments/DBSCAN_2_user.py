from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import DBSCAN

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import pq_to_range, p_to_theta_deg


# ============================================================
# Experiment settings
# ============================================================

USERS = [
    {"range_m": 200.00, "theta_deg": 17.08},
    {"range_m": 200.03, "theta_deg": 20.68},
]

CLOCK_OFFSET = 10e-9

THETA_MARGIN_BINS = 2
TAU_MARGIN_SAMPLES = 2

PRT_R_MIN, PRT_R_MAX = 40.0, 500.0
PRT_THETA_LIMIT = 60.0

DBSCAN_THRESHOLD = 0.5
DBSCAN_EPS = 1.8
DBSCAN_MIN_SAMPLES = 3


# ============================================================
# Signal and search grids
# ============================================================

def build_time_axis(users, antenna_x):
    all_delays = [
        propagation_delays(u["range_m"], u["theta_deg"], CLOCK_OFFSET, antenna_x)
        for u in users
    ]
    delays = np.concatenate(all_delays)
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def build_local_p_grid(users):
    theta_values = np.array([u["theta_deg"] for u in users])
    margin = THETA_MARGIN_BINS * PRT.theta_step_deg
    theta_min = theta_values.min() - margin
    theta_max = theta_values.max() + margin

    theta_grid = np.arange(
        theta_min,
        theta_max + 0.5 * PRT.theta_step_deg,
        PRT.theta_step_deg,
    )
    p_grid = -np.sin(np.deg2rad(theta_grid)) / C
    return theta_grid, p_grid


def build_global_q_grid():
    q_min = np.cos(np.deg2rad(PRT_THETA_LIMIT))**2 / (2.0 * PRT_R_MAX * C)
    q_max = 1.0 / (2.0 * PRT_R_MIN * C)
    return np.arange(q_min - 2 * PRT.dq, q_max + 2 * PRT.dq, PRT.dq)


def get_local_tau_indices(users, time_axis):
    tau_values = np.array([
        u["range_m"] / C + CLOCK_OFFSET for u in users
    ])
    margin = TAU_MARGIN_SAMPLES * PRT.tau_step
    tau_min = tau_values.min() - margin
    tau_max = tau_values.max() + margin
    return np.where((time_axis >= tau_min) & (time_axis <= tau_max))[0]


def generate_two_user_signal(users, time_axis, antenna_x):
    total_rx = np.zeros((len(antenna_x), len(time_axis)), dtype=np.complex128)

    for user in users:
        rx, _ = received_signal(
            time_axis,
            user["range_m"],
            user["theta_deg"],
            clock_offset=CLOCK_OFFSET,
            antenna_x=antenna_x,
        )
        total_rx += rx

    return total_rx


# ============================================================
# DBSCAN peak detection
# ============================================================

def valid_pq_mask(p_grid, q_grid):
    P, Q = np.meshgrid(p_grid, q_grid, indexing="ij")
    ranges = pq_to_range(P, Q)
    return np.isfinite(ranges) & (ranges >= PRT_R_MIN) & (ranges <= PRT_R_MAX)


def detect_prt_clusters(prt, time_axis, tau_indices, p_grid, q_grid):
    valid = valid_pq_mask(p_grid, q_grid)
    score = np.where(valid[None, :, :], np.abs(prt[tau_indices]), -np.inf)

    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        return []

    threshold = DBSCAN_THRESHOLD * finite_score.max()
    candidate_mask = np.isfinite(score) & (score >= threshold)
    points = np.argwhere(candidate_mask)

    if len(points) < DBSCAN_MIN_SAMPLES:
        return []

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(points.astype(float))

    estimates = []

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_points = points[labels == label]
        cluster_values = score[tuple(cluster_points.T)]
        peak_point = cluster_points[np.argmax(cluster_values)]

        tau_local_idx, p_idx, q_idx = map(int, peak_point)
        tau_idx = int(tau_indices[tau_local_idx])

        tau_hat = float(time_axis[tau_idx])
        p_hat = float(p_grid[p_idx])
        q_hat = float(q_grid[q_idx])
        range_hat = float(pq_to_range(p_hat, q_hat))
        theta_hat = float(p_to_theta_deg(p_hat))

        theta_rad = np.deg2rad(theta_hat)
        x_hat = float(range_hat * np.sin(theta_rad))
        y_hat = float(range_hat * np.cos(theta_rad))

        estimates.append({
            "range_m": range_hat,
            "theta_deg": theta_hat,
            "tau_s": tau_hat,
            "x_m": x_hat,
            "y_m": y_hat,
            "p_s_per_m": p_hat,
            "q_s_per_m2": q_hat,
            "peak": float(cluster_values.max()),
            "cluster_size": int(len(cluster_points)),
        })

    estimates.sort(key=lambda item: item["peak"], reverse=True)
    return estimates


# ============================================================
# Complete experiment
# ============================================================

def estimate_users(users=USERS):
    antenna_x = ARRAY.positions(SYSTEM)
    time_axis = build_time_axis(users, antenna_x)
    theta_grid, p_grid = build_local_p_grid(users)
    q_grid = build_global_q_grid()
    tau_indices = get_local_tau_indices(users, time_axis)

    clean_rx = generate_two_user_signal(users, time_axis, antenna_x)
    noisy_rx, _ = add_awgn(clean_rx, rng=np.random.default_rng(SIM.rng_seed))
    mf_rx = matched_filter(noisy_rx)

    prt = parabolic_radon_transform(
        mf_rx,
        antenna_x,
        p_grid,
        q_grid,
    )

    estimates = detect_prt_clusters(
        prt,
        time_axis,
        tau_indices,
        p_grid,
        q_grid,
    )

    return len(estimates), estimates, theta_grid, time_axis[tau_indices]


def main():
    num_users, estimates, theta_grid, tau_grid = estimate_users()

    print("=" * 72)
    print("TRUE USERS")
    print("=" * 72)

    for i, user in enumerate(USERS, start=1):
        tau_true = user["range_m"] / C + CLOCK_OFFSET
        theta = np.deg2rad(user["theta_deg"])
        x_true = user["range_m"] * np.sin(theta)
        y_true = user["range_m"] * np.cos(theta)

        print(
            f"User {i}: R={user['range_m']:.6f} m, "
            f"theta={user['theta_deg']:.6f} deg, "
            f"tau={tau_true * 1e9:.6f} ns, "
            f"x={x_true:.6f} m, y={y_true:.6f} m"
        )

    print("\n" + "=" * 72)
    print("LOCAL PRT SEARCH")
    print("=" * 72)
    print(
        f"theta: {theta_grid[0]:.3f} to {theta_grid[-1]:.3f} deg, "
        f"Np={len(theta_grid)}"
    )
    print(
        f"tau: {tau_grid[0] * 1e9:.6f} to "
        f"{tau_grid[-1] * 1e9:.6f} ns, Ntau={len(tau_grid)}"
    )

    print("\n" + "=" * 72)
    print(f"DBSCAN DETECTED USERS: {num_users}")
    print("=" * 72)

    if num_users == 0:
        print("No valid DBSCAN cluster was detected.")
        return

    for i, estimate in enumerate(estimates, start=1):
        print(
            f"User {i}: R={estimate['range_m']:.6f} m, "
            f"theta={estimate['theta_deg']:.6f} deg, "
            f"tau={estimate['tau_s'] * 1e9:.6f} ns, "
            f"x={estimate['x_m']:.6f} m, "
            f"y={estimate['y_m']:.6f} m, "
            f"peak={estimate['peak']:.6e}, "
            f"cluster_size={estimate['cluster_size']}"
        )


if __name__ == "__main__":
    main()