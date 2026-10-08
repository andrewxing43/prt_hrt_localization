from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import DBSCAN

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import pq_to_range, p_to_theta_deg


R_TRUE, THETA_TRUE, CLOCK_OFFSET = 200.37, 17.37, 10e-9
THETA_STEP, THETA_HALF_BINS, TAU_HALF_SAMPLES = PRT.theta_step_deg, 4, 3
R_MIN, R_MAX, DQ = 40.0, 500.0, PRT.dq
DBSCAN_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 0.5, 1.8, 3


def build_time_axis(R, theta, tau_true, antenna_x):
    delays = propagation_delays(R, theta, CLOCK_OFFSET, antenna_x)
    margin = 8.0 * SYSTEM.gaussian_sigma
    t_min, t_max = delays.min() - margin, delays.max() + margin

    # Force tau_true halfway between two time samples -> off-grid.
    n_before = int(np.ceil((tau_true - t_min) / SYSTEM.dt - 0.5))
    t_start = tau_true - (n_before + 0.5) * SYSTEM.dt
    n = int(np.ceil((t_max - t_start) / SYSTEM.dt)) + 1
    return t_start + np.arange(n) * SYSTEM.dt


def build_grids(theta_true):
    # theta_true is halfway between theta-grid points -> off-grid.
    offsets = np.arange(-THETA_HALF_BINS, THETA_HALF_BINS) + 0.5
    theta_grid = theta_true + offsets * THETA_STEP
    p_grid = -np.sin(np.deg2rad(theta_grid)) / C

    q_min = np.cos(np.deg2rad(60.0))**2 / (2.0 * R_MAX * C)
    q_max = 1.0 / (2.0 * R_MIN * C)
    q_grid = np.arange(q_min, q_max + 0.5 * DQ, DQ)

    return theta_grid, p_grid, q_grid


def local_prt_cube(prt, time_axis, p_grid, q_grid, tau_true):
    tau_idx = np.where(np.abs(time_axis - tau_true) <= TAU_HALF_SAMPLES * SYSTEM.dt)[0]

    P, Q = np.meshgrid(p_grid, q_grid, indexing="ij")
    R = pq_to_range(P, Q)
    valid = np.isfinite(R) & (R >= R_MIN) & (R <= R_MAX)

    score = np.where(valid[None, :, :], np.abs(prt[tau_idx]), -np.inf)
    return score, tau_idx


def estimate_argmax(score, tau_idx, time_axis, p_grid, q_grid):
    i = np.unravel_index(np.argmax(score), score.shape)
    tau, p, q = time_axis[tau_idx[i[0]]], p_grid[i[1]], q_grid[i[2]]
    return i, float(pq_to_range(p, q)), float(p_to_theta_deg(p)), float(tau), float(p), float(q)


def estimate_dbscan(score, tau_idx, time_axis, p_grid, q_grid):
    threshold = DBSCAN_THRESHOLD * np.max(score)
    candidate = np.isfinite(score) & (score >= threshold)
    points = np.argwhere(candidate)

    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(points.astype(float))
    clusters = [k for k in np.unique(labels) if k != -1]

    if not clusters:
        raise RuntimeError("DBSCAN found no valid cluster.")

    # Choose the cluster containing the strongest PRT response.
    best_label = max(clusters, key=lambda k: np.max(score[tuple(points[labels == k].T)]))
    cluster_points = points[labels == best_label]

    # Peak position of that cluster.
    values = score[tuple(cluster_points.T)]
    i = tuple(cluster_points[np.argmax(values)])

    tau, p, q = time_axis[tau_idx[i[0]]], p_grid[i[1]], q_grid[i[2]]
    return i, float(pq_to_range(p, q)), float(p_to_theta_deg(p)), float(tau), float(p), float(q), len(clusters), len(points)


def main():
    antenna_x = ARRAY.positions(SYSTEM)
    tau_true = R_TRUE / C + CLOCK_OFFSET
    time_axis = build_time_axis(R_TRUE, THETA_TRUE, tau_true, antenna_x)
    theta_grid, p_grid, q_grid = build_grids(THETA_TRUE)

    rx, _ = received_signal(time_axis, R_TRUE, THETA_TRUE, CLOCK_OFFSET, antenna_x=antenna_x)
    rx, _ = add_awgn(rx, rng=np.random.default_rng(SIM.rng_seed))
    mf_rx = matched_filter(rx)

    prt = parabolic_radon_transform(mf_rx, antenna_x, p_grid, q_grid)
    score, tau_idx = local_prt_cube(prt, time_axis, p_grid, q_grid, tau_true)

    argmax_result = estimate_argmax(score, tau_idx, time_axis, p_grid, q_grid)
    dbscan_result = estimate_dbscan(score, tau_idx, time_axis, p_grid, q_grid)

    snr_db = SIM.reference_snr_db + 20.0 * np.log10(SIM.reference_range / R_TRUE)

    print(f"True   : R={R_TRUE:.6f} m, theta={THETA_TRUE:.6f} deg, tau={tau_true*1e9:.6f} ns")
    print(f"Argmax : R={argmax_result[1]:.6f} m, theta={argmax_result[2]:.6f} deg, tau={argmax_result[3]*1e9:.6f} ns")
    print(f"DBSCAN : R={dbscan_result[1]:.6f} m, theta={dbscan_result[2]:.6f} deg, tau={dbscan_result[3]*1e9:.6f} ns")
    print(f"SNR≈{snr_db:.2f} dB, Tx={SIM.tx_power_dbm:.3f} dBm")
    print(f"Clusters={dbscan_result[6]}, candidates={dbscan_result[7]}")
    print(f"Same grid peak: {argmax_result[0] == dbscan_result[0]}")


if __name__ == "__main__":
    main()