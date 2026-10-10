from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import C, SYSTEM, ARRAY, SIM, PRT
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import build_roi
from localization.pipeline import run_hrt, polar_to_xy


# ============================================================
# Settings
# ============================================================

USERS = [
    {"range_m": 400.0, "theta_deg": 0.58},
    {"range_m": 400.0, "theta_deg": 0.08},

]

BASE_CLOCK_OFFSET = 10e-9
COMMON_TAU = max(u["range_m"] for u in USERS) / C + BASE_CLOCK_OFFSET

PRT_R_MIN, PRT_R_MAX = 40.0, 500.0
PRT_THETA_MIN, PRT_THETA_MAX = -60.0, 60.0
THETA_MARGIN_BINS, N_LOCAL_TAU = 2, 4

SIC_STOP_THRESHOLD = 1.95e-5
SIC_DAMPING = 1.0
MAX_SIC_ITERATIONS = 6
MAX_REPEAT_CANCELLATIONS = 3
MIN_RELATIVE_ENERGY_DROP = 1e-5

DUPLICATE_RANGE_TOL = 2.0
DUPLICATE_THETA_TOL = 0.02
DUPLICATE_TAU_TOL = SYSTEM.dt

THETA_GRID = np.arange(
    PRT_THETA_MIN, PRT_THETA_MAX + 0.5 * PRT.theta_step_deg,
    PRT.theta_step_deg
)
P_GRID_GLOBAL = -np.sin(np.deg2rad(THETA_GRID)) / C

Q_MIN = np.cos(np.deg2rad(60.0))**2 / (2.0 * PRT_R_MAX * C)
Q_MAX = 1.0 / (2.0 * PRT_R_MIN * C)
Q_GRID = np.arange(Q_MIN - 2 * PRT.dq, Q_MAX + 2 * PRT.dq, PRT.dq)


# ============================================================
# Signal and grids
# ============================================================

def user_clock_offset(user):
    return COMMON_TAU - user["range_m"] / C


def build_time_axis(antenna_x):
    delays = np.concatenate([
        propagation_delays(
            u["range_m"], u["theta_deg"], user_clock_offset(u), antenna_x
        )
        for u in USERS
    ])

    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def nearest_indices(grid, value, count):
    index = np.searchsorted(grid, value)
    start = int(np.clip(index - count // 2, 0, len(grid) - count))
    return np.arange(start, start + count)


def build_local_p_grid():
    theta = np.array([u["theta_deg"] for u in USERS])
    margin = THETA_MARGIN_BINS * PRT.theta_step_deg
    lo, hi = theta.min() - margin, theta.max() + margin

    mask = (THETA_GRID >= lo) & (THETA_GRID <= hi)
    theta_grid = THETA_GRID[mask]
    p_grid = P_GRID_GLOBAL[mask]

    if theta_grid.size == 0:
        raise ValueError("Local theta grid is empty.")

    for value in theta:
        if np.any(np.isclose(theta_grid, value, rtol=0.0, atol=1e-12)):
            raise ValueError(f"theta={value} deg is on the PRT theta grid.")

    return theta_grid, p_grid


def get_local_tau_indices(time_axis):
    return nearest_indices(time_axis, COMMON_TAU, N_LOCAL_TAU)


def valid_pq_mask(p_grid):
    P, Q = np.meshgrid(p_grid, Q_GRID, indexing="ij")
    ranges = pq_to_range(P, Q)
    return np.isfinite(ranges) & (ranges >= PRT_R_MIN) & (ranges <= PRT_R_MAX)


def generate_multi_user_signal(time_axis, antenna_x):
    total_rx = np.zeros((len(antenna_x), len(time_axis)), dtype=np.complex128)

    for user in USERS:
        rx, _ = received_signal(
            time_axis, user["range_m"], user["theta_deg"],
            clock_offset=user_clock_offset(user), antenna_x=antenna_x
        )
        total_rx += rx

    return total_rx


# ============================================================
# PRT
# ============================================================

def run_local_prt(mf_rx, time_axis, antenna_x, p_grid, tau_indices):
    prt = parabolic_radon_transform(mf_rx, antenna_x, p_grid, Q_GRID)
    valid = valid_pq_mask(p_grid)
    score = np.where(valid[None, :, :], np.abs(prt[tau_indices]), -np.inf)

    index = np.unravel_index(np.argmax(score), score.shape)
    tau_local_idx, p_idx, q_idx = index
    tau_idx = int(tau_indices[tau_local_idx])

    p_hat, q_hat = float(p_grid[p_idx]), float(Q_GRID[q_idx])

    return {
        "R": float(pq_to_range(p_hat, q_hat)),
        "theta": float(p_to_theta_deg(p_hat)),
        "tau": float(time_axis[tau_idx]),
        "p": p_hat,
        "q": q_hat,
        "peak": float(score[index]),
    }


# ============================================================
# SIC
# ============================================================

def find_duplicate(users, R, theta, tau):
    for i, user in enumerate(users):
        if (
            abs(user["range_m"] - R) <= DUPLICATE_RANGE_TOL
            and abs(user["theta_deg"] - theta) <= DUPLICATE_THETA_TOL
            and abs(user["tau_s"] - tau) <= DUPLICATE_TAU_TOL
        ):
            return i

    return None


def cancel_user(residual_rx, time_axis, antenna_x, R, theta, tau):
    clock_offset = tau - R / C

    template, _ = received_signal(
        time_axis, R, theta, clock_offset=clock_offset,
        tx_power=1.0, antenna_x=antenna_x
    )

    denominator = np.vdot(template, template)
    if abs(denominator) <= 1e-30:
        raise RuntimeError("Reconstructed template has zero energy.")

    amplitude = np.vdot(template, residual_rx) / denominator
    reconstructed = amplitude * template

    energy_before = float(np.vdot(residual_rx, residual_rx).real)
    candidate = residual_rx - SIC_DAMPING * reconstructed
    energy_after = float(np.vdot(candidate, candidate).real)
    relative_drop = (energy_before - energy_after) / max(energy_before, 1e-30)

    return candidate, amplitude, energy_before, energy_after, relative_drop


def run_sic(noisy_rx, time_axis, antenna_x):
    theta_grid, p_grid = build_local_p_grid()
    tau_indices = get_local_tau_indices(time_axis)

    residual_rx = noisy_rx.copy()
    detected_users = []

    print("=" * 80)
    print(f"COMMON TAU: {COMMON_TAU * 1e9:.6f} ns")
    print(
        f"PRT theta={theta_grid[0]:.3f} to {theta_grid[-1]:.3f} deg, "
        f"Np={len(theta_grid)}, Ntau={len(tau_indices)}, Nq={len(Q_GRID)}"
    )
    print("=" * 80)

    for user in USERS:
        nearest = float(theta_grid[np.argmin(np.abs(theta_grid - user["theta_deg"]))])
        print(
            f"True theta={user['theta_deg']:.6f} deg, "
            f"nearest grid={nearest:.6f} deg, "
            f"offset={user['theta_deg'] - nearest:+.6f} deg, OFF-GRID"
        )

    for iteration in range(1, MAX_SIC_ITERATIONS + 1):
        print("\n" + "=" * 80)
        print(f"SIC ITERATION {iteration}")
        print("=" * 80)

        mf_residual = matched_filter(residual_rx)
        prt_result = run_local_prt(
            mf_residual, time_axis, antenna_x, p_grid, tau_indices
        )

        print(
            f"PRT: R={prt_result['R']:.6f} m, "
            f"theta={prt_result['theta']:.6f} deg, "
            f"tau={prt_result['tau'] * 1e9:.6f} ns"
        )
        print(
            f"PRT peak={prt_result['peak']:.6e}, "
            f"threshold={SIC_STOP_THRESHOLD:.6e}"
        )

        if prt_result["peak"] <= SIC_STOP_THRESHOLD:
            print("Stopping: residual PRT peak is below the threshold.")
            break

        roi = build_roi(
            prt_result["R"], prt_result["theta"], prt_result["tau"],
            range_bounds=(PRT_R_MIN, PRT_R_MAX),
            theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX),
            tau_bounds=(float(time_axis[0]), float(time_axis[-1]))
        )

        print(
            f"ROI: R=[{roi.R_min:.3f}, {roi.R_max:.3f}] m, "
            f"theta=[{roi.theta_min:.3f}, {roi.theta_max:.3f}] deg, "
            f"tau=[{roi.tau_min * 1e9:.6f}, {roi.tau_max * 1e9:.6f}] ns"
        )

        R_hat, theta_hat, tau_hat, hrt_peak = run_hrt(
            mf_residual, time_axis, antenna_x, roi
        )

        x_hat, y_hat = polar_to_xy(R_hat, theta_hat)
        clock_offset_hat = tau_hat - R_hat / C

        print(
            f"HRT: R={R_hat:.6f} m, theta={theta_hat:.6f} deg, "
            f"tau={tau_hat * 1e9:.6f} ns, peak={hrt_peak:.6e}"
        )

        candidate, amplitude, energy_before, energy_after, relative_drop = (
            cancel_user(
                residual_rx, time_axis, antenna_x,
                R_hat, theta_hat, tau_hat
            )
        )

        print(f"Amplitude={amplitude.real:+.6e}{amplitude.imag:+.6e}j")
        print(
            f"Energy={energy_before:.6e} -> {energy_after:.6e}, "
            f"drop={100.0 * relative_drop:.6f}%"
        )

        if energy_after >= energy_before:
            print("Stopping: cancellation did not reduce residual energy.")
            break

        if relative_drop < MIN_RELATIVE_ENERGY_DROP:
            print("Stopping: residual energy reduction is too small.")
            break

        residual_rx = candidate
        duplicate_idx = find_duplicate(
            detected_users, R_hat, theta_hat, tau_hat
        )

        if duplicate_idx is None:
            detected_users.append({
                "range_m": R_hat,
                "theta_deg": theta_hat,
                "tau_s": tau_hat,
                "clock_offset_s": clock_offset_hat,
                "x_m": x_hat,
                "y_m": y_hat,
                "amplitude": amplitude,
                "prt_peak": prt_result["peak"],
                "hrt_peak": hrt_peak,
                "cancellations": 1,
            })
        else:
            user = detected_users[duplicate_idx]
            user["amplitude"] += amplitude
            user["cancellations"] += 1

            print(
                f"Residual matched detected user {duplicate_idx + 1}; "
                "user count not increased."
            )

            if user["cancellations"] >= MAX_REPEAT_CANCELLATIONS:
                print("Stopping: maximum repeated cancellations reached.")
                break

    return detected_users, residual_rx


# ============================================================
# Main
# ============================================================

def main():
    antenna_x = ARRAY.positions(SYSTEM)
    time_axis = build_time_axis(antenna_x)

    clean_rx = generate_multi_user_signal(time_axis, antenna_x)
    noisy_rx, noise_variance = add_awgn(
        clean_rx, rng=np.random.default_rng(SIM.rng_seed)
    )

    detected_users, residual_rx = run_sic(
        noisy_rx, time_axis, antenna_x
    )

    initial_energy = float(np.vdot(noisy_rx, noisy_rx).real)
    residual_energy = float(np.vdot(residual_rx, residual_rx).real)

    print("\n" + "=" * 80)
    print("TRUE USERS")
    print("=" * 80)

    for i, user in enumerate(USERS, 1):
        R, theta = user["range_m"], user["theta_deg"]
        tau, clock = COMMON_TAU, user_clock_offset(user)
        x, y = polar_to_xy(R, theta)

        print(
            f"User {i}: R={R:.6f} m, theta={theta:.6f} deg, "
            f"tau={tau * 1e9:.6f} ns"
        )
        print(
            f"        clock={clock * 1e9:.6f} ns, "
            f"x={x:.6f} m, y={y:.6f} m"
        )

    print("\n" + "=" * 80)
    print(f"SIC DETECTED USERS: {len(detected_users)}")
    print("=" * 80)

    for i, user in enumerate(detected_users, 1):
        print(
            f"User {i}: R={user['range_m']:.6f} m, "
            f"theta={user['theta_deg']:.6f} deg, "
            f"tau={user['tau_s'] * 1e9:.6f} ns"
        )
        print(
            f"        clock={user['clock_offset_s'] * 1e9:.6f} ns, "
            f"x={user['x_m']:.6f} m, y={user['y_m']:.6f} m"
        )
        print(
            f"        PRT peak={user['prt_peak']:.6e}, "
            f"HRT peak={user['hrt_peak']:.6e}, "
            f"cancellations={user['cancellations']}"
        )

    print(f"\nNoise variance: {noise_variance:.6e}")
    print(
        f"Total energy: {initial_energy:.6e} -> "
        f"{residual_energy:.6e}"
    )


if __name__ == "__main__":
    main()