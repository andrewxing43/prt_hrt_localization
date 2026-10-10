from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 1.01, 2
THRESHOLD_SWEEP = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30,
                   0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.01]

PEAK_MIN_HEIGHT = 1e-6
PEAK_MIN_PROMINENCE = 1e-7
PEAK_MIN_DISTANCE = 3
EXPECTED_Q_WINDOW = 20
MAX_PRINTED_PEAKS = 15

COMMON_TAU = max(u["range_m"] for u in USERS) / C + BASE_CLOCK_OFFSET
GLOBAL_THETA_GRID = np.arange(-PRT_THETA_LIMIT,
                              PRT_THETA_LIMIT + 0.5 * PRT.theta_step_deg,
                              PRT.theta_step_deg)


# ============================================================
# Signal and grids
# ============================================================

def user_clock_offset(user):
    return COMMON_TAU - user["range_m"] / C


def theoretical_q(user):
    theta = np.deg2rad(user["theta_deg"])
    return np.cos(theta)**2 / (2.0 * user["range_m"] * C)


def build_time_axis(users, antenna_x):
    delays = np.concatenate([
        propagation_delays(u["range_m"], u["theta_deg"],
                           user_clock_offset(u), antenna_x)
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
    q_min = np.cos(np.deg2rad(PRT_THETA_LIMIT))**2 / (2 * PRT_R_MAX * C)
    q_max = 1.0 / (2 * PRT_R_MIN * C)
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


def generate_signal(users, time_axis, antenna_x):
    signal = np.zeros((len(antenna_x), len(time_axis)), dtype=np.complex128)
    for user in users:
        rx, _ = received_signal(
            time_axis, user["range_m"], user["theta_deg"],
            clock_offset=user_clock_offset(user), antenna_x=antenna_x
        )
        signal += rx
    return signal


def valid_pq_mask(p_grid, q_grid):
    P, Q = np.meshgrid(p_grid, q_grid, indexing="ij")
    ranges = pq_to_range(P, Q)
    return np.isfinite(ranges) & (ranges >= PRT_R_MIN) & (ranges <= PRT_R_MAX)


# ============================================================
# Diagnostics
# ============================================================

def make_score(prt, tau_indices, p_grid, q_grid):
    valid = valid_pq_mask(p_grid, q_grid)
    score = np.where(valid[None, :, :], np.abs(prt[tau_indices]), -np.inf)
    if not np.any(np.isfinite(score)):
        raise RuntimeError("No valid PRT cells.")
    return score


def q_peak_estimate(score, qi, time_axis, tau_indices, p_grid, q_grid):
    ti, pi = np.unravel_index(np.argmax(score[:, :, qi]), score[:, :, qi].shape)
    tau = float(time_axis[tau_indices[ti]])
    p, q = float(p_grid[pi]), float(q_grid[qi])
    return {
        "q_idx": int(qi), "p_idx": int(pi), "tau_idx": int(ti),
        "range_m": float(pq_to_range(p, q)),
        "theta_deg": float(p_to_theta_deg(p)),
        "tau_s": tau, "peak": float(score[ti, pi, qi])
    }


def inspect_local_peaks(score, profile, time_axis, tau_indices, p_grid, q_grid):
    peaks, props = find_peaks(
        profile, height=PEAK_MIN_HEIGHT, prominence=PEAK_MIN_PROMINENCE,
        distance=PEAK_MIN_DISTANCE
    )
    order = np.argsort(profile[peaks])[::-1][:MAX_PRINTED_PEAKS]

    print("\n" + "=" * 88)
    print("LOCAL MAXIMA OF q PROFILE")
    print("=" * 88)
    print(f"Candidates={len(peaks)}, displaying strongest {min(len(order), MAX_PRINTED_PEAKS)}")

    if len(peaks) == 0:
        print("No local q-profile peak satisfies the diagnostic settings.")
        return

    for rank, j in enumerate(order, 1):
        qi = int(peaks[j])
        est = q_peak_estimate(score, qi, time_axis, tau_indices, p_grid, q_grid)
        print(
            f"{rank:2d}: q_idx={qi:4d}, R={est['range_m']:10.4f} m, "
            f"theta={est['theta_deg']:8.4f} deg, "
            f"tau={est['tau_s'] * 1e9:11.6f} ns, "
            f"peak={profile[qi]:.6e}, ratio={profile[qi]/profile.max():.5f}, "
            f"prominence={props['prominences'][j]:.6e}"
        )


def inspect_expected_regions(score, profile, time_axis, tau_indices, p_grid, q_grid):
    observed = []

    print("\n" + "=" * 88)
    print("EXPECTED USER REGIONS")
    print("=" * 88)

    for i, user in enumerate(USERS, 1):
        q_true = theoretical_q(user)
        q_expected = int(np.argmin(np.abs(q_grid - q_true)))
        lo = max(0, q_expected - EXPECTED_Q_WINDOW)
        hi = min(len(q_grid) - 1, q_expected + EXPECTED_Q_WINDOW)
        qi = lo + int(np.argmax(profile[lo:hi + 1]))
        est = q_peak_estimate(score, qi, time_axis, tau_indices, p_grid, q_grid)
        observed.append(est)

        print(
            f"User {i}: true R={user['range_m']:.4f} m, q={q_true:.6e}, "
            f"nearest q_idx={q_expected}, local maximum q_idx={qi}"
        )
        print(
            f"        observed R={est['range_m']:.4f} m, "
            f"theta={est['theta_deg']:.4f} deg, "
            f"peak={profile[qi]:.6e}, ratio={profile[qi]/profile.max():.6f}"
        )

    if len(observed) != 2 or observed[0]["q_idx"] == observed[1]["q_idx"]:
        print("\nThe two expected regions lead to the same observed maximum.")
        print("The PRT q-profile does not currently contain two separable target peaks.")
        return observed

    q1, q2 = sorted([observed[0]["q_idx"], observed[1]["q_idx"]])
    valley_segment = profile[q1:q2 + 1]
    valley_offset = int(np.argmin(valley_segment))
    valley_idx = q1 + valley_offset
    valley, strongest = float(profile[valley_idx]), float(profile.max())
    weak_peak = min(profile[q1], profile[q2])
    valley_ratio, weak_ratio = valley / strongest, weak_peak / strongest

    print("\n" + "-" * 88)
    print(f"Peak separation       : {q2 - q1} q-bins")
    print(f"Inter-peak valley     : q_idx={valley_idx}, value={valley:.6e}")
    print(f"Valley/global ratio   : {valley_ratio:.6f}")
    print(f"Weak/global peak ratio: {weak_ratio:.6f}")

    if valley_ratio < weak_ratio:
        suggested = 0.5 * (valley_ratio + weak_ratio)
        print(f"Possible threshold interval: {valley_ratio:.6f} < threshold < {weak_ratio:.6f}")
        print(f"Suggested first test       : DBSCAN_THRESHOLD = {suggested:.6f}")
    else:
        print("No relative threshold can both retain the weak peak and remove the q bridge.")
        print("Thresholded-voxel DBSCAN cannot reliably separate these two users.")

    return observed


def dbscan_clusters(score, threshold_ratio, time_axis, tau_indices, p_grid, q_grid):
    threshold = threshold_ratio * np.max(score[np.isfinite(score)])
    points = np.argwhere(np.isfinite(score) & (score >= threshold))
    if len(points) < DBSCAN_MIN_SAMPLES:
        return points, []

    labels = DBSCAN(
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
    ).fit_predict(points.astype(float))

    clusters = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        values = score[tuple(cluster.T)]
        peak = cluster[np.argmax(values)]
        ti, pi, qi = map(int, peak)
        p, q = float(p_grid[pi]), float(q_grid[qi])
        clusters.append({
            "size": len(cluster), "q_lo": int(cluster[:, 2].min()),
            "q_hi": int(cluster[:, 2].max()), "peak": float(values.max()),
            "range_m": float(pq_to_range(p, q)),
            "theta_deg": float(p_to_theta_deg(p)),
            "tau_s": float(time_axis[tau_indices[ti]])
        })

    clusters.sort(key=lambda x: x["peak"], reverse=True)
    return points, clusters


def sweep_dbscan(score, time_axis, tau_indices, p_grid, q_grid):
    print("\n" + "=" * 88)
    print(f"DBSCAN SWEEP: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")
    print("=" * 88)

    for threshold in THRESHOLD_SWEEP:
        points, clusters = dbscan_clusters(
            score, threshold, time_axis, tau_indices, p_grid, q_grid
        )
        summary = ", ".join(
            f"R={c['range_m']:.2f}m/q[{c['q_lo']}:{c['q_hi']}]/n={c['size']}"
            for c in clusters[:6]
        )
        print(
            f"threshold={threshold:6.3f}: points={len(points):5d}, "
            f"clusters={len(clusters):2d}"
            + (f" | {summary}" if summary else "")
        )


def save_profile_plot(profile, p_grid, q_grid, observed):
    p_ref = p_grid[len(p_grid) // 2]
    ranges = np.asarray(pq_to_range(p_ref, q_grid), dtype=float)
    valid = np.isfinite(ranges) & (ranges >= PRT_R_MIN) & (ranges <= PRT_R_MAX)
    order = np.argsort(ranges[valid])
    x, y = ranges[valid][order], profile[valid][order] / profile.max()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, y, lw=1.5, color="navy", label="max over local theta and tau")

    for user in USERS:
        ax.axvline(user["range_m"], color="green", ls="--", lw=1,
                   label="true range" if user is USERS[0] else None)

    for est in observed:
        ax.axvline(est["range_m"], color="red", ls=":", lw=1,
                   label="local maximum" if est is observed[0] else None)

    for threshold in [0.8, 0.5, 0.3, 0.2, 0.1, 0.05]:
        ax.axhline(threshold, color="gray", lw=0.5, alpha=0.35)

    ax.set(xlim=(PRT_R_MIN, PRT_R_MAX), xlabel="Range (m)",
           ylabel="Normalized PRT q-profile",
           title="PRT range profile and DBSCAN thresholds")
    ax.set_yscale("log")
    ax.set_ylim(max(1e-4, np.min(y[y > 0]) * 0.8), 1.2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = Path(__file__).with_name("DBSCAN_q_diagnostic.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"\nSaved plot: {output}")


# ============================================================
# Main
# ============================================================

def main():
    antenna_x = ARRAY.positions(SYSTEM)
    time_axis = build_time_axis(USERS, antenna_x)
    theta_grid, p_grid = build_local_p_grid(USERS)
    q_grid = build_global_q_grid()
    tau_indices = get_local_tau_indices(time_axis)

    clean_rx = generate_signal(USERS, time_axis, antenna_x)
    noisy_rx, noise_variance = add_awgn(
        clean_rx, rng=np.random.default_rng(SIM.rng_seed)
    )
    mf_rx = matched_filter(noisy_rx)
    prt = parabolic_radon_transform(mf_rx, antenna_x, p_grid, q_grid)
    score = make_score(prt, tau_indices, p_grid, q_grid)

    profile = np.max(score, axis=(0, 1))
    profile = np.where(np.isfinite(profile), profile, 0.0)

    print("=" * 88)
    print("PRT/DBSCAN DIAGNOSTIC")
    print("=" * 88)
    print(f"Common tau       : {COMMON_TAU * 1e9:.6f} ns")
    print(f"Noise variance   : {noise_variance:.6e}")
    print(f"Global PRT peak  : {profile.max():.6e}")
    print(f"Theta grid       : {theta_grid[0]:.3f} to {theta_grid[-1]:.3f} deg, Np={len(p_grid)}")
    print(f"Tau grid         : {time_axis[tau_indices[0]] * 1e9:.6f} to "
          f"{time_axis[tau_indices[-1]] * 1e9:.6f} ns, Ntau={len(tau_indices)}")
    print(f"q grid           : {q_grid[0]:.6e} to {q_grid[-1]:.6e}, Nq={len(q_grid)}")

    inspect_local_peaks(score, profile, time_axis, tau_indices, p_grid, q_grid)
    observed = inspect_expected_regions(
        score, profile, time_axis, tau_indices, p_grid, q_grid
    )
    sweep_dbscan(score, time_axis, tau_indices, p_grid, q_grid)
    save_profile_plot(profile, p_grid, q_grid, observed)


if __name__ == "__main__":
    main()