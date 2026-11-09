
"""Monte Carlo: two-user range resolution with Delta R = 20 m."""

from pathlib import Path
import sys
import csv
import contextlib
import io
import numpy as np
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.noise import add_awgn
import multi_user_SIC as sic

# ==================== Settings ====================
N_MC = 200
DELTA_R = 50.0

R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
BASE_CLOCK_OFFSET = 10e-9

VERBOSE_SIC = False
OUTPUT_CSV = Path(__file__).with_name(
    "mc_two_user_range_resolution.csv"
)


def generate_user_pair(rng):
    # Ensure both users remain within [50, 400] m
    R1 = rng.uniform(R_MIN, R_MAX - DELTA_R)
    R2 = R1 + DELTA_R
    theta = rng.uniform(THETA_MIN, THETA_MAX)

    return [
        {"range_m": R1, "theta_deg": theta},
        {"range_m": R2, "theta_deg": theta}
    ]


def run_trial(users, antenna_x, rng_noise):

    # Update the simulation scenario only
    sic.USERS = users
    sic.COMMON_TAU = (
        max(u["range_m"] for u in users) / C
        + BASE_CLOCK_OFFSET
    )

    # Same arrival time, different clock offsets
    time_axis = sic.build_time_axis(antenna_x)

    # Generate two-user signal
    clean_rx = sic.generate_multi_user_signal(
        time_axis, antenna_x
    )

    # Add AWGN
    noisy_rx, noise_variance = add_awgn(
        clean_rx, rng=rng_noise
    )

    # Original PRT -> ROI -> HRT -> Powell -> SIC
    if VERBOSE_SIC:
        detected, residual = sic.run_sic(
            noisy_rx, time_axis, antenna_x, noise_variance
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            detected, residual = sic.run_sic(
                noisy_rx, time_axis, antenna_x, noise_variance
            )

    return detected, residual


def main():
    if not (0 < DELTA_R < R_MAX - R_MIN):
        raise ValueError("DELTA_R must be between 0 and R_MAX-R_MIN.")

    rng_pos = np.random.default_rng(SIM.rng_seed)
    rng_noise = np.random.default_rng(SIM.rng_seed + 1)
    antenna_x = ARRAY.positions(SYSTEM)

    records = []
    detection_counts = {}
    success_count = 0

    print("=" * 65)
    print("Two-User Range Resolution Monte Carlo")
    print("=" * 65)
    print(f"Monte Carlo trials : {N_MC}")
    print(f"Range separation   : {DELTA_R:.3f} m")
    print(f"Range interval     : [{R_MIN}, {R_MAX}] m")
    print(f"Angle interval     : [{THETA_MIN}, {THETA_MAX}] deg")
    print("Angle condition    : theta1 = theta2")
    print("Arrival condition  : tau1 = tau2")
    print(f"Gain mode          : {sic.GAIN_MODE}")
    print(f"Max SIC iterations : {sic.MAX_SIC_ITERATIONS}")
    print("=" * 65)

    for i in trange(N_MC, desc="Monte Carlo"):
        users = generate_user_pair(rng_pos)
        detected, residual = run_trial(
            users, antenna_x, rng_noise
        )

        n_detected = len(detected)
        success = (n_detected == 2)
        success_count += int(success)

        detection_counts[n_detected] = (
            detection_counts.get(n_detected, 0) + 1
        )

        record = {
            "trial": i + 1,
            "R1_true": users[0]["range_m"],
            "R2_true": users[1]["range_m"],
            "theta_true": users[0]["theta_deg"],
            "delta_R": DELTA_R,
            "tau_true_ns": sic.COMMON_TAU * 1e9,
            "detected_count": n_detected,
            "success": int(success),
            "R1_est": np.nan,
            "theta1_est": np.nan,
            "R2_est": np.nan,
            "theta2_est": np.nan
        }

        if n_detected >= 1:
            record["R1_est"] = detected[0]["range_m"]
            record["theta1_est"] = detected[0]["theta_deg"]

        if n_detected >= 2:
            record["R2_est"] = detected[1]["range_m"]
            record["theta2_est"] = detected[1]["theta_deg"]

        records.append(record)

        current_rate = success_count / (i + 1)

        print(
            f"Trial {i+1:3d} | "
            f"R=({users[0]['range_m']:7.3f}, "
            f"{users[1]['range_m']:7.3f}) m | "
            f"theta={users[0]['theta_deg']:+8.4f} deg | "
            f"Detected={n_detected} | "
            f"Success={success} | "
            f"Rate={current_rate:.2%}"
        )

    # ==================== Statistics ====================
    p_success = success_count / N_MC

    # Wilson 95% confidence interval
    z = 1.96
    denominator = 1 + z**2 / N_MC
    center = (
        p_success + z**2 / (2 * N_MC)
    ) / denominator
    half_width = (
        z * np.sqrt(
            p_success * (1 - p_success) / N_MC
            + z**2 / (4 * N_MC**2)
        ) / denominator
    )

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"Range separation   : {DELTA_R:.3f} m")
    print(f"Total trials       : {N_MC}")
    print(f"Successful trials  : {success_count}")
    print(f"Failed trials      : {N_MC - success_count}")
    print(f"Detection probability: {p_success:.4%}")
    print(
        f"95% confidence interval: "
        f"[{center-half_width:.4%}, "
        f"{center+half_width:.4%}]"
    )

    print("\nDetected-user count distribution:")
    for k in sorted(detection_counts):
        count = detection_counts[k]
        print(
            f"  {k} users: {count:4d} trials "
            f"({count/N_MC:.2%})"
        )

    # ==================== Save CSV ====================
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nResults saved to: {OUTPUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
