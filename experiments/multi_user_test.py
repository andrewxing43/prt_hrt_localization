"""
固定用户位置和噪声，比较单用户、双用户、三用户定位误差。

保存位置：
    experiments/paired_user_comparison.py

运行：
    python experiments/paired_user_comparison.py

依赖：
    原项目及其依赖，包括可用的 CUDA/CuPy。

每个配对块：
    生成用户 A、B、C。
    单用户场景：A、B、C
    双用户场景：AB、AC、BC
    三用户场景：ABC

同一配对块中的七个场景：
    1. 用户参数保持不变。
    2. 使用相同阵列和时间轴。
    3. 使用完全相同的接收机噪声。
    4. 复用 multi_user_final.py 中的原始定位算法。

输出：
    paired_user_results.csv
        每个场景、每个真实用户的估计结果。
    paired_user_comparison.csv
        严格配对的用户级误差比较。

注意：
    原算法使用真实角度、tau 作为辅助搜索提示。
    本脚本保留此设定。
"""

from pathlib import Path
from itertools import combinations
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import multi_user_final as core


# ==================== 配置 ====================

# 每个配对块运行七个场景。
# 建议先设置为 2，确认环境可正常运行后再增加。
N_BLOCKS = 40

POSITION_SEED = 2026
NOISE_SEED = 20261110

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "paired_user_results.csv"
COMPARISON_CSV = OUTPUT_DIR / "paired_user_comparison.csv"

# 原算法默认会由底层函数打印信息，这里保留其静默行为。
SHOW_PLOTS = True


# ==================== 工具函数 ====================

def scene_name(indices):
    return "".join("ABC"[i] for i in indices)


def match_estimates(truth, detected):
    """
    返回：
        errors：按真实用户顺序排列的二维误差；
        matched：按真实用户顺序排列的估计结果。

    若人数不一致，则不强制配对。
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


# ==================== 单个场景 ====================

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

    # 原算法通过全局 USERS 读取本场景的真实参数提示。
    core.USERS = truth

    clean_rx = np.zeros_like(noise)
    for i in indices:
        clean_rx += components[i]

    # 同一个 block 中，所有场景使用完全相同的噪声矩阵。
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


# ==================== 严格配对统计 ====================

def build_paired_rows(block_records, base_users, block_id):
    """
    仅当本 block 的七个场景全部检出人数正确时纳入。

    每个基础用户：
        单用户：一个误差；
        双用户：两个误差，分别对应两个不同干扰用户；
        三用户：一个误差。

    保留两个双用户误差，避免先平均误差再计算 RMSE。
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

    print("\n严格配对统计：仅保留七个场景均检出人数正确的块")
    print(f"有效配对块：{valid_blocks}/{total_blocks}")
    print(f"相同位置的基础用户数：{len(paired_rows)}")

    if not paired_rows:
        print("没有完整有效的配对块。")
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
        "\n严格配对统计中，三组的真实位置分布完全一致，"
        f"平均距离均为 {mean_r:.6f} m。"
    )
    print("双用户组每个基础用户出现两次，权重相同。")
    print(
        f"平均误差差值，双用户减单用户："
        f"{delta_two:+.6f} m"
    )
    print(
        f"平均误差差值，三用户减单用户："
        f"{delta_three:+.6f} m"
    )
    print("差值为正表示多用户误差更大。")


# ==================== 主程序 ====================

def main():
    if N_BLOCKS <= 0:
        raise ValueError("N_BLOCKS 必须为正整数。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 使用相互独立的随机流生成位置与噪声。
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
                # 三个基础用户满足原算法的两两分离条件。
                base_users = core.generate_users(
                    position_rng, 3
                )

                # 时间轴由三个基础用户共同确定。
                # 七个场景均使用该时间轴，避免采样窗口不同。
                core.USERS = base_users
                t = core.build_time_axis(x)

                # 每个基础用户只生成一次无噪声信号，
                # 各场景直接复用，确保信号参数完全一致。
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

                # 对零信号加噪，得到接收机噪声本身。
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

                # 每完成一个配对块就保存，保留已完成结果。
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

    print("\n各组人数检出情况")
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
            f"{n} 用户：场景数={len(group)}，"
            f"人数正确={success}，少检={miss}，多检={over}"
        )

    print("\n原统计口径：分别保留各组人数正确的场景")
    stats = core.print_summary(records)

    print_paired_summary(paired_rows, N_BLOCKS)

    print(f"\n场景结果：{RESULTS_CSV}")
    print(f"严格配对结果：{COMPARISON_CSV}")

    if SHOW_PLOTS:
        # 图片使用严格配对样本，保证位置分布一致。
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
            print("无有效严格配对样本，跳过绘图。")

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