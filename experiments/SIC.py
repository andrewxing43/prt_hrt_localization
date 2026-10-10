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

R_TRUE, THETA_TRUE, CLOCK_OFFSET_TRUE = 450.0, 30.08, 10e-9
TAU_TRUE = R_TRUE / C + CLOCK_OFFSET_TRUE

PRT_R_MIN, PRT_R_MAX = 40.0, 500.0
PRT_THETA_MIN, PRT_THETA_MAX = -60.0, 60.0
N_LOCAL_P, N_LOCAL_TAU = 4, 4

SIC_STOP_THRESHOLD = 1e-5
SIC_DAMPING = 1.0
MAX_SIC_ITERATIONS = 5
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
# Signal and PRT grids
# ============================================================

def build_time_axis(antenna_x):
    delays = propagation_delays(
        R_TRUE, THETA_TRUE, CLOCK_OFFSET_TRUE, antenna_x
    )
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def nearest_indices(grid, value, count):
    index = np.searchsorted(grid, value)
    start = int(np.clip(index - count // 2, 0, len(grid) - count))
    return np.arange(start, start + count)


def build_local_p_grid():
    indices = nearest_indices(THETA_GRID, THETA_TRUE, N_LOCAL_P)
    theta_grid = THETA_GRID[indices]

    if np.any(np.isclose(theta_grid, THETA_TRUE, rtol=0.0, atol=1e-12)):
        raise ValueError(
            f"THETA_TRUE={THETA_TRUE} deg is on the PRT theta grid."
        )

    return theta_grid, P_GRID_GLOBAL[indices]


def get_local_tau_indices(time_axis):
    return nearest_indices(time_axis, TAU_TRUE, N_LOCAL_TAU)


def valid_pq_mask(p_grid):
    P, Q = np.meshgrid(p_grid, Q_GRID, indexing="ij")
    ranges = pq_to_range(P, Q)
    return np.isfinite(ranges) & (ranges >= PRT_R_MIN) & (ranges <= PRT_R_MAX)


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

    nearest_theta = float(
        theta_grid[np.argmin(np.abs(theta_grid - THETA_TRUE))]
    )

    print(
        f"True theta={THETA_TRUE:.6f} deg, "
        f"nearest PRT theta={nearest_theta:.6f} deg, "
        f"offset={THETA_TRUE - nearest_theta:+.6f} deg, OFF-GRID"
    )

    for iteration in range(1, MAX_SIC_ITERATIONS + 1):
        print("\n" + "=" * 76)
        print(f"SIC ITERATION {iteration}")
        print("=" * 76)

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

    return detected_users, residual_rx, theta_grid, time_axis[tau_indices]


# ============================================================
# Main
# ============================================================

def main():
    antenna_x = ARRAY.positions(SYSTEM)
    time_axis = build_time_axis(antenna_x)

    clean_rx, _ = received_signal(
        time_axis, R_TRUE, THETA_TRUE,
        clock_offset=CLOCK_OFFSET_TRUE, antenna_x=antenna_x
    )

    noisy_rx, noise_variance = add_awgn(
        clean_rx, rng=np.random.default_rng(SIM.rng_seed)
    )

    detected_users, residual_rx, theta_grid, tau_grid = run_sic(
        noisy_rx, time_axis, antenna_x
    )

    x_true, y_true = polar_to_xy(R_TRUE, THETA_TRUE)
    initial_energy = float(np.vdot(noisy_rx, noisy_rx).real)
    residual_energy = float(np.vdot(residual_rx, residual_rx).real)

    print("\n" + "=" * 76)
    print("TRUE USER")
    print("=" * 76)
    print(
        f"R={R_TRUE:.6f} m, theta={THETA_TRUE:.6f} deg, "
        f"tau={TAU_TRUE * 1e9:.6f} ns"
    )
    print(
        f"clock={CLOCK_OFFSET_TRUE * 1e9:.6f} ns, "
        f"x={x_true:.6f} m, y={y_true:.6f} m"
    )
    print(
        f"theta search={theta_grid[0]:.3f} to "
        f"{theta_grid[-1]:.3f} deg, Np={len(theta_grid)}"
    )
    print(
        f"tau search={tau_grid[0] * 1e9:.6f} to "
        f"{tau_grid[-1] * 1e9:.6f} ns, Ntau={len(tau_grid)}"
    )
    print(f"Noise variance={noise_variance:.6e}")

    print("\n" + "=" * 76)
    print(f"SIC DETECTED USERS: {len(detected_users)}")
    print("=" * 76)

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

    print(
        f"\nTotal energy: {initial_energy:.6e} -> "
        f"{residual_energy:.6e}"
    )


if __name__ == "__main__":
    main()