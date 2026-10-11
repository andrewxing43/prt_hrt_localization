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
# Settings
# ============================================================

USERS = [
    {"range_m": 100.00, "theta_deg": 41.58},
    {"range_m": 200.00, "theta_deg": 41.58},
]

BASE_CLOCK_OFFSET = 10e-9
THETA_MARGIN_BINS, TAU_MARGIN_SAMPLES = 2, 2
PRT_R_MIN, PRT_R_MAX, PRT_THETA_LIMIT = 40.0, 500.0, 60.0

DBSCAN_THRESHOLD = 0.05
DBSCAN_EPS = 1.01
DBSCAN_MIN_SAMPLES = 3

COMMON_TAU = max(u["range_m"] for u in USERS) / C + BASE_CLOCK_OFFSET

GLOBAL_THETA_GRID = np.arange(
    -PRT_THETA_LIMIT, PRT_THETA_LIMIT + 0.5 * PRT.theta_step_deg,
    PRT.theta_step_deg
)


# ============================================================
# Timing
# ============================================================

def user_clock_offset(user):
    return COMMON_TAU - user["range_m"] / C


def user_tau(user):
    return user["range_m"] / C + user_clock_offset(user)


# ============================================================
# Signal and grids
# ============================================================

def build_time_axis(users, antenna_x):
    delays = np.concatenate([
        propagation_delays(
            u["range_m"], u["theta_deg"], user_clock_offset(u), antenna_x
        )
        for u in users
    ])

    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def build_local_p_grid(users):
    theta = np.array([u["theta_deg"] for u in users])
    margin = THETA_MARGIN_BINS * PRT.theta_step_deg
    lo, hi = theta.min() - margin, theta.max() + margin

    theta_grid = GLOBAL_THETA_GRID[
        (GLOBAL_THETA_GRID >= lo) & (GLOBAL_THETA_GRID <= hi)
    ]
    if theta_grid.size == 0:
        raise ValueError("Local theta grid is empty.")

    return theta_grid, -np.sin(np.deg2rad(theta_grid)) / C


def build_global_q_grid():
    q_min = np.cos(np.deg2rad(PRT_THETA_LIMIT))**2 / (2.0 * PRT_R_MAX * C)
    q_max = 1.0 / (2.0 * PRT_R_MIN * C)
    return np.arange(q_min - 2 * PRT.dq, q_max + 2 * PRT.dq, PRT.dq)


def get_local_tau_indices(time_axis):
    margin = TAU_MARGIN_SAMPLES * PRT.tau_step
    indices = np.where(
        (time_axis >= COMMON_TAU - margin) &
        (time_axis <= COMMON_TAU + margin)
    )[0]

    if indices.size == 0:
        raise ValueError("Local tau grid is empty.")

    return indices


def generate_multi_user_signal(users, time_axis, antenna_x):
    total_rx = np.zeros((len(antenna_x), len(time_axis)), dtype=np.complex128)

    for user in users:
        rx, _ = received_signal(
            time_axis, user["range_m"], user["theta_deg"],
            clock_offset=user_clock_offset(user), antenna_x=antenna_x
        )
        total_rx += rx

    return total_rx


# ============================================================
# DBSCAN
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
    points = np.argwhere(np.isfinite(score) & (score >= threshold))

    if len(points) < DBSCAN_MIN_SAMPLES:
        return []

    labels = DBSCAN(
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
    ).fit_predict(points.astype(float))

    estimates = []

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster = points[labels == label]
        values = score[tuple(cluster.T)]
        peak = cluster[np.argmax(values)]

        tau_local_idx, p_idx, q_idx = map(int, peak)
        tau_idx = int(tau_indices[tau_local_idx])

        tau_hat = float(time_axis[tau_idx])
        p_hat, q_hat = float(p_grid[p_idx]), float(q_grid[q_idx])
        range_hat = float(pq_to_range(p_hat, q_hat))
        theta_hat = float(p_to_theta_deg(p_hat))

        theta_rad = np.deg2rad(theta_hat)
        x_hat = float(range_hat * np.sin(theta_rad))
        y_hat = float(range_hat * np.cos(theta_rad))

        estimates.append({
            "range_m": range_hat,
            "theta_deg": theta_hat,
            "tau_s": tau_hat,
            "clock_offset_s": tau_hat - range_hat / C,
            "x_m": x_hat,
            "y_m": y_hat,
            "p_s_per_m": p_hat,
            "q_s_per_m2": q_hat,
            "peak": float(values.max()),
            "cluster_size": int(len(cluster)),
        })

    return sorted(estimates, key=lambda x: x["peak"], reverse=True)


# ============================================================
# Complete experiment
# ============================================================

def estimate_users(users=USERS):
    antenna_x = ARRAY.positions(SYSTEM)
    time_axis = build_time_axis(users, antenna_x)
    theta_grid, p_grid = build_local_p_grid(users)
    q_grid = build_global_q_grid()
    tau_indices = get_local_tau_indices(time_axis)

    clean_rx = generate_multi_user_signal(users, time_axis, antenna_x)
    noisy_rx, _ = add_awgn(clean_rx, rng=np.random.default_rng(SIM.rng_seed))
    mf_rx = matched_filter(noisy_rx)

    prt = parabolic_radon_transform(mf_rx, antenna_x, p_grid, q_grid)
    estimates = detect_prt_clusters(
        prt, time_axis, tau_indices, p_grid, q_grid
    )

    return len(estimates), estimates, theta_grid, time_axis[tau_indices], q_grid


# ============================================================
# Output
# ============================================================

def main():
    num_users, estimates, theta_grid, tau_grid, q_grid = estimate_users()

    print("=" * 76)
    print(f"COMMON TAU: {COMMON_TAU * 1e9:.6f} ns")
    print("=" * 76)
    print("TRUE USERS")
    print("=" * 76)

    for i, user in enumerate(USERS, 1):
        R, theta = user["range_m"], user["theta_deg"]
        tau, offset = user_tau(user), user_clock_offset(user)
        theta_rad = np.deg2rad(theta)
        x, y = R * np.sin(theta_rad), R * np.cos(theta_rad)

        nearest = float(theta_grid[np.argmin(np.abs(theta_grid - theta))])
        grid_offset = theta - nearest
        status = "ON-GRID" if np.isclose(grid_offset, 0.0, atol=1e-12) else "OFF-GRID"

        print(
            f"User {i}: R={R:.6f} m, theta={theta:.6f} deg, "
            f"tau={tau * 1e9:.6f} ns, clock={offset * 1e9:.6f} ns"
        )
        print(
            f"        x={x:.6f} m, y={y:.6f} m, "
            f"nearest theta={nearest:.6f} deg, "
            f"offset={grid_offset:+.6f} deg, {status}"
        )

    print("\n" + "=" * 76)
    print("LOCAL PRT SEARCH")
    print("=" * 76)
    print(
        f"theta={theta_grid[0]:.3f} to {theta_grid[-1]:.3f} deg, "
        f"Np={len(theta_grid)}"
    )
    print(
        f"tau={tau_grid[0] * 1e9:.6f} to {tau_grid[-1] * 1e9:.6f} ns, "
        f"Ntau={len(tau_grid)}"
    )
    print(
        f"q={q_grid[0]:.6e} to {q_grid[-1]:.6e} s/m², "
        f"Nq={len(q_grid)}"
    )

    print("\n" + "=" * 76)
    print(f"DBSCAN DETECTED USERS: {num_users}")
    print("=" * 76)

    if num_users == 0:
        print("No valid DBSCAN cluster was detected.")
        return

    for i, est in enumerate(estimates, 1):
        print(
            f"User {i}: R={est['range_m']:.6f} m, "
            f"theta={est['theta_deg']:.6f} deg, "
            f"tau={est['tau_s'] * 1e9:.6f} ns"
        )
        print(
            f"        clock={est['clock_offset_s'] * 1e9:.6f} ns, "
            f"x={est['x_m']:.6f} m, y={est['y_m']:.6f} m"
        )
        print(
            f"        peak={est['peak']:.6e}, "
            f"cluster_size={est['cluster_size']}"
        )


if __name__ == "__main__":
    main()