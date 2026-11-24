"""
Compare localization errors across one-, two-, and three-user scenes.

Run with python experiments/multi_user_test.py in an environment with CUDA and CuPy.
Each block runs A, B, C, AB, AC, BC, and ABC with the same array, time axis,
user parameters, and receiver noise. Localization reuses multi_user_final.py.
paired_user_results.csv stores per-scene estimates.
paired_user_comparison.csv stores paired user-level errors.
"""

from pathlib import Path
from itertools import combinations
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import multi_user_final as core


# Configuration

# Each block runs seven scenes.
# Use two blocks for a quick environment check.
N_BLOCKS = 40

POSITION_SEED = 2026
NOISE_SEED = 20261110

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "paired_user_results.csv"
COMPARISON_CSV = OUTPUT_DIR / "paired_user_comparison.csv"

# Show the paired comparison plots.
SHOW_PLOTS = True


# Helpers

def scene_name(indices):
    return "".join("ABC"[i] for i in indices)


def match_estimates(truth, detected):
    """
    Return 2D errors and matched estimates in true-user order.
    Pairing requires equal true and estimated user counts.
    """
    if len(truth) != len(detected):
        return None, None

    true_xy = core.positions_from_users(truth)
    estimated_xy = core.positions_from_users(
        detected, estimated=True
    )

    distances = np.linalg.norm(
        true_xy[:, None, :] - estimated_xy[None, :, :],
        axis=2,
    )

    rows, cols = linear_sum_assignment(distances)

    errors = np.empty(len(truth), dtype=float)
    matched = [None] * len(truth)

    for i, j in zip(rows, cols):
        errors[i] = distances[i, j]
        matched[i] = detected[j]

    return errors, matched


def metrics(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return {
            "n": 0,
            "ale": np.nan,
            "rmse": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "n": values.size,
        "ale": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def print_metrics(label, values):
    s = metrics(values)
    print(
        f"{label:<16}"
        f"{s['n']:>9d}"
        f"{s['ale']:>13.6f}"
        f"{s['rmse']:>13.6f}"
        f"{s['p95']:>13.6f}"
        f"{s['max']:>13.6f}"
    )


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Individual scene

def run_scene(
    block_id,
    trial_id,
    indices,
    base_users,
    components,
    noise,
    t,
    x,
    noise_seed,
):
    truth = [dict(base_users[i]) for i in indices]

    # The core reads search hints for this scene from USERS.
    core.USERS = truth

    clean_rx = np.zeros_like(noise)
    for i in indices:
        clean_rx += components[i]

    # All scenes in the block use the same noise matrix.
    noisy_rx = clean_rx + noise

    with core.redirect_stdout(core.io.StringIO()):
        detected, residual = core.run_sic(noisy_rx, t, x)

    n_true = len(truth)
    n_hat = len(detected)

    status = (
        "SUCCESS" if n_hat == n_true
        else "OVER" if n_hat > n_true
        else "MISS"
    )

    errors, matched = match_estimates(truth, detected)

    record = {
        "trial": trial_id,
        "block_id": block_id,
        "scene": scene_name(indices),
        "indices": tuple(indices),
        "noise_seed": noise_seed,
        "true_n": n_true,
        "detected_n": n_hat,
        "status": status,
        "truth": truth,
        "estimated": detected,
        "errors": (
            errors if errors is not None else np.array([])
        ),
        "initial_energy": core.energy(noisy_rx),
        "residual_energy": core.energy(residual),
    }

    rows = []

    for local_i, global_i in enumerate(indices):
        u = truth[local_i]
        true_x, true_y = core.polar_to_xy(
            u["range_m"], u["theta_deg"]
        )

        row = {
            "block_id": block_id,
            "trial_id": trial_id,
            "scene": scene_name(indices),
            "user_count": n_true,
            "base_user_id": "ABC"[global_i],
            "noise_seed": noise_seed,
            "detected_n": n_hat,
            "status": status,
            "included_in_error_statistics": int(
                status == "SUCCESS"
            ),
            "true_R_m": u["range_m"],
            "true_theta_deg": u["theta_deg"],
            "true_tau_ns": u["tau_ns"],
            "true_x_m": true_x,
            "true_y_m": true_y,
        }

        if matched is not None:
            estimate = matched[local_i]

            row.update({
                "estimated_R_m": estimate["range_m"],
                "estimated_theta_deg": estimate["theta_deg"],
                "estimated_tau_ns": estimate["tau_s"] * 1e9,
                "estimated_x_m": estimate["x_m"],
                "estimated_y_m": estimate["y_m"],
                "error_2d_m": float(errors[local_i]),
            })

        rows.append(row)

    return record, rows


# Strictly paired statistics

def build_paired_rows(block_records, base_users, block_id):
    """
    Include only blocks with correct counts in all seven scenes.
    Each base user contributes one single-user, two two-user, and one
    three-user error. Keep both two-user errors when computing RMSE.
    """
    if not all(
        r["status"] == "SUCCESS"
        for r in block_records
    ):
        return []

    errors_by_user = {
        i: {1: [], 2: [], 3: []}
        for i in range(3)
    }

    for record in block_records:
        for local_i, global_i in enumerate(record["indices"]):
            errors_by_user[global_i][record["true_n"]].append(
                float(record["errors"][local_i])
            )

    rows = []

    for i, u in enumerate(base_users):
        e1 = errors_by_user[i][1]
        e2 = errors_by_user[i][2]
        e3 = errors_by_user[i][3]

        assert len(e1) == 1
        assert len(e2) == 2
        assert len(e3) == 1

        rows.append({
            "block_id": block_id,
            "base_user_id": "ABC"[i],
            "true_R_m": u["range_m"],
            "true_theta_deg": u["theta_deg"],
            "true_tau_ns": u["tau_ns"],
            "single_error_m": e1[0],
            "two_user_error_1_m": e2[0],
            "two_user_error_2_m": e2[1],
            "two_user_mean_error_m": float(np.mean(e2)),
            "three_user_error_m": e3[0],
            "delta_two_minus_single_m": (
                float(np.mean(e2)) - e1[0]
            ),
            "delta_three_minus_single_m": e3[0] - e1[0],
        })

    return rows


def print_paired_summary(paired_rows, total_blocks):
    valid_blocks = len(paired_rows) // 3

    print("\nStrictly paired statistics (correct counts in all seven scenes)")
    print(f"Valid paired blocks: {valid_blocks}/{total_blocks}")
    print(f"Base users at shared positions: {len(paired_rows)}")

    if not paired_rows:
        print("No complete paired blocks.")
        return

    single = [r["single_error_m"] for r in paired_rows]

    double = [
        r[key]
        for r in paired_rows
        for key in (
            "two_user_error_1_m",
            "two_user_error_2_m",
        )
    ]

    triple = [
        r["three_user_error_m"]
        for r in paired_rows
    ]

    print(
        f"{'Group':<16}"
        f"{'Samples':>9}"
        f"{'ALE(m)':>13}"
        f"{'RMSE(m)':>13}"
        f"{'P95(m)':>13}"
        f"{'Max(m)':>13}"
    )

    print_metrics("1 user", single)
    print_metrics("2 users", double)
    print_metrics("3 users", triple)

    mean_r = np.mean([
        r["true_R_m"] for r in paired_rows
    ])

    delta_two = np.mean([
        r["delta_two_minus_single_m"]
        for r in paired_rows
    ])

    delta_three = np.mean([
        r["delta_three_minus_single_m"]
        for r in paired_rows
    ])

    print(
        "\nAll three groups use the same true position distribution. "
        f"The mean range is {mean_r:.6f} m in each group."
    )
    print("Each base user appears twice in the two-user group, with equal weight.")
    print(
        f"Mean error difference (two users minus one): "
        f"{delta_two:+.6f} m"
    )
    print(
        f"Mean error difference (three users minus one): "
        f"{delta_three:+.6f} m"
    )
    print("Positive differences indicate larger multi-user errors.")


# Main

def main():
    if N_BLOCKS <= 0:
        raise ValueError("N_BLOCKS must be a positive integer.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use independent random streams for positions and noise.
    position_rng = np.random.default_rng(POSITION_SEED)
    x = core.ARRAY.positions(core.SYSTEM)

    scenes = (
        [(i,) for i in range(3)]
        + list(combinations(range(3), 2))
        + [(0, 1, 2)]
    )

    records = []
    result_rows = []
    paired_rows = []

    previous_users = core.USERS
    trial_id = 0

    try:
        with tqdm(
            total=N_BLOCKS * len(scenes),
            desc="Paired comparison",
            unit="scene",
        ) as progress:
            for block_id in range(1, N_BLOCKS + 1):
                # Enforce the core experiment's pairwise separation constraints.
                base_users = core.generate_users(
                    position_rng, 3
                )

                # Build one time axis from all three base users and reuse it in all scenes.
                core.USERS = base_users
                t = core.build_time_axis(x)

                # Generate each clean user signal once and reuse it across scenes.
                components = [
                    core.reconstruct(
                        t,
                        x,
                        u["range_m"],
                        u["theta_deg"],
                        core.user_tau(u),
                    )
                    for u in base_users
                ]

                noise_seed = NOISE_SEED + block_id

                # Add noise to a zero signal to isolate receiver noise.
                noise, _ = core.add_awgn(
                    np.zeros_like(components[0]),
                    rng=np.random.default_rng(noise_seed),
                )

                block_records = []

                for indices in scenes:
                    trial_id += 1

                    record, rows = run_scene(
                        block_id=block_id,
                        trial_id=trial_id,
                        indices=indices,
                        base_users=base_users,
                        components=components,
                        noise=noise,
                        t=t,
                        x=x,
                        noise_seed=noise_seed,
                    )

                    records.append(record)
                    block_records.append(record)
                    result_rows.extend(rows)

                    progress.update(1)

                paired_rows.extend(
                    build_paired_rows(
                        block_records,
                        base_users,
                        block_id,
                    )
                )

                # Save after each block to retain completed results.
                write_csv(
                    RESULTS_CSV,
                    result_rows,
                    RESULT_FIELDS,
                )

                write_csv(
                    COMPARISON_CSV,
                    paired_rows,
                    COMPARISON_FIELDS,
                )

    finally:
        core.USERS = previous_users

    print("\nUser-count detection by group")
    for n in (1, 2, 3):
        group = [
            r for r in records if r["true_n"] == n
        ]
        success = sum(
            r["status"] == "SUCCESS" for r in group
        )
        miss = sum(r["status"] == "MISS" for r in group)
        over = sum(r["status"] == "OVER" for r in group)

        print(
            f"{n} user(s): scenes={len(group)}, "
            f"correct count={success}, undercounts={miss}, overcounts={over}"
        )

    print("\nPer-group statistics (scenes with correct user counts)")
    stats = core.print_summary(records)

    print_paired_summary(paired_rows, N_BLOCKS)

    print(f"\nScene results: {RESULTS_CSV}")
    print(f"Strictly paired results: {COMPARISON_CSV}")

    if SHOW_PLOTS:
        # Plot strictly paired samples so all groups share the same position distribution.
        valid_block_ids = {
            r["block_id"] for r in paired_rows
        }

        paired_records = [
            r for r in records
            if r["block_id"] in valid_block_ids
        ]

        if paired_records:
            paired_stats = {
                n: core.summarize([
                    r for r in paired_records
                    if r["true_n"] == n
                ])
                for n in (1, 2, 3)
            }
            paired_stats["all"] = core.summarize(
                paired_records
            )

            core.plot_results(
                paired_stats, paired_records
            )
        else:
            print("No valid strictly paired samples to plot.")

    return records, paired_rows, stats


RESULT_FIELDS = [
    "block_id",
    "trial_id",
    "scene",
    "user_count",
    "base_user_id",
    "noise_seed",
    "detected_n",
    "status",
    "included_in_error_statistics",
    "true_R_m",
    "true_theta_deg",
    "true_tau_ns",
    "true_x_m",
    "true_y_m",
    "estimated_R_m",
    "estimated_theta_deg",
    "estimated_tau_ns",
    "estimated_x_m",
    "estimated_y_m",
    "error_2d_m",
]

COMPARISON_FIELDS = [
    "block_id",
    "base_user_id",
    "true_R_m",
    "true_theta_deg",
    "true_tau_ns",
    "single_error_m",
    "two_user_error_1_m",
    "two_user_error_2_m",
    "two_user_mean_error_m",
    "three_user_error_m",
    "delta_two_minus_single_m",
    "delta_three_minus_single_m",
]


if __name__ == "__main__":
    main()