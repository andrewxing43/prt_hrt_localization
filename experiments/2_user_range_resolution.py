
"""Monte Carlo evaluation of two-user range resolution.

Both users have identical angle and TOA, but their ranges differ by
DELTA_R. The experiment directly uses the complete localization
algorithm in multi_user_final.py, including PRT, ROI, HRT, joint Powell
refinement, and SIC.

Results are displayed in the terminal without saving files.
"""

from pathlib import Path
from contextlib import redirect_stdout
from scipy.optimize import linear_sum_assignment
from tqdm import trange
import numpy as np
import sys
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SYSTEM, ARRAY
from signal_model.noise import add_awgn
import multi_user_final as sic


# ==================== Settings ====================

N_MC = 500
DELTA_R =200
MC_SEED = 2026

R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
TOA_MIN_NS, TOA_MAX_NS = 1500.0, 1502.0

VERBOSE_SIC = False

# Optional accuracy criterion for correctly resolving both users.
CHECK_LOCALIZATION = True
MAX_RANGE_ERROR_M = 40.0
MAX_ANGLE_ERROR_DEG = 2
MAX_TOA_ERROR_PS = 100.0


# ==================== User Generation ====================

def generate_user_pair(rng):
    R1 = float(rng.uniform(R_MIN, R_MAX - DELTA_R))
    R2 = R1 + DELTA_R
    theta = float(rng.uniform(THETA_MIN, THETA_MAX))
    tau = float(rng.uniform(TOA_MIN_NS, TOA_MAX_NS))

    return [
        {"range_m": R1, "theta_deg": theta, "tau_ns": tau},
        {"range_m": R2, "theta_deg": theta, "tau_ns": tau}
    ]


# ==================== Single Trial ====================

def run_trial(users, antenna_x, noise_seed):
    sic.USERS = users

    t = sic.build_time_axis(antenna_x)
    clean_rx = sic.generate_signal(t, antenna_x)
    noisy_rx, _ = add_awgn(
        clean_rx, rng=np.random.default_rng(noise_seed)
    )

    if VERBOSE_SIC:
        detected, residual = sic.run_sic(noisy_rx, t, antenna_x)
    else:
        with redirect_stdout(io.StringIO()):
            detected, residual = sic.run_sic(noisy_rx, t, antenna_x)

    return detected, residual


# ==================== Success Evaluation ====================

def evaluate_detection(users, detected):
    if len(detected) != 2:
        return False

    if not CHECK_LOCALIZATION:
        return True

    truth = np.array([
        [u["range_m"], u["theta_deg"], u["tau_ns"]]
        for u in users
    ])

    estimates = np.array([
        [u["range_m"], u["theta_deg"], u["tau_s"] * 1e9]
        for u in detected
    ])

    errors = np.abs(truth[:, None, :] - estimates[None, :, :])
    tolerances = np.array([
        MAX_RANGE_ERROR_M, MAX_ANGLE_ERROR_DEG, MAX_TOA_ERROR_PS * 1e-3
    ])

    normalized = errors / tolerances
    cost = np.max(normalized, axis=2)

    rows, cols = linear_sum_assignment(cost)
    return bool(np.all(cost[rows, cols] <= 1.0))


# ==================== Monte Carlo ====================

def monte_carlo():
    if not (0 < DELTA_R < R_MAX - R_MIN):
        raise ValueError("DELTA_R must be between 0 and R_MAX-R_MIN.")

    rng = np.random.default_rng(MC_SEED)
    antenna_x = ARRAY.positions(SYSTEM)

    success_count = 0
    detection_counts = {}
    records = []

    print("=" * 65)
    print("TWO-USER RANGE RESOLUTION MONTE CARLO")
    print("=" * 65)
    print(f"Monte Carlo trials : {N_MC}")
    print(f"Range separation   : {DELTA_R:.3f} m")
    print(f"Range interval     : [{R_MIN}, {R_MAX}] m")
    print(f"Angle interval     : [{THETA_MIN}, {THETA_MAX}] deg")
    print("Angle condition    : theta1 = theta2")
    print("Arrival condition  : tau1 = tau2")
    print(f"HRT threshold      : {sic.HRT_STOP_THRESHOLD:.2e}")
    print(f"Max SIC iterations : {sic.MAX_SIC_ITERATIONS}")
    print(f"Localization check : {CHECK_LOCALIZATION}")
    print("=" * 65)

    progress = trange(N_MC, desc="Monte Carlo", unit="trial")

    for i in progress:
        users = generate_user_pair(rng)
        detected, residual = run_trial(
            users, antenna_x, MC_SEED + i + 1
        )

        n_detected = len(detected)
        success = evaluate_detection(users, detected)

        success_count += int(success)
        detection_counts[n_detected] = detection_counts.get(n_detected, 0) + 1

        records.append({
            "trial": i + 1,
            "users": users,
            "detected": detected,
            "detected_count": n_detected,
            "success": success
        })

        progress.set_postfix(
            detected=n_detected,
            success=success_count,
            P_detect=f"{success_count / (i + 1):.2%}"
        )

    # ==================== Statistics ====================

    p_detect = success_count / N_MC

    z = 1.96
    denominator = 1 + z**2 / N_MC
    center = (p_detect + z**2 / (2 * N_MC)) / denominator
    half_width = z * np.sqrt(
        p_detect * (1 - p_detect) / N_MC
        + z**2 / (4 * N_MC**2)
    ) / denominator

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"Range separation    : {DELTA_R:.3f} m")
    print(f"Total trials        : {N_MC}")
    print(f"Successful trials   : {success_count}")
    print(f"Failed trials       : {N_MC - success_count}")
    print(f"Detection probability: {p_detect:.4%}")
    print(
        f"95% confidence interval: "
        f"[{center - half_width:.4%}, {center + half_width:.4%}]"
    )

    print("\nDetected-user count distribution:")
    for n in sorted(detection_counts):
        count = detection_counts[n]
        print(f"  {n} users: {count:4d} / {N_MC} ({count / N_MC:.2%})")

    print("=" * 65)

    return records, p_detect


if __name__ == "__main__":
    monte_carlo()
