
"""Monte Carlo multi-user localization with HRT-initialized joint Powell refinement and SIC.

Run from within the original project; displays statistics and figures without saving data.
"""

from pathlib import Path
from contextlib import redirect_stdout
from scipy.optimize import minimize, linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import io
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import build_roi
from localization.pipeline import (
    run_hrt, polar_to_xy, P_GRID_GLOBAL, DP, Q_GRID,
    PRT_R_MIN, PRT_R_MAX, PRT_THETA_MIN, PRT_THETA_MAX,
    N_LOCAL_P, N_LOCAL_TAU
)

# ==================== Configuration ====================

N_MC_BY_USER_COUNT = {1: 600, 2: 300, 3: 200}
MC_SEED = 2026
R_MIN, R_MAX = 50.0, 400.0
TH_MIN, TH_MAX = -60.0, 60.0

TOA_MIN_NS, TOA_MAX_NS = 1500.0, 1502.0
MIN_RANGE_SEP_M = 100.0
MIN_ANGLE_SEP_DEG = 1.0
MIN_TOA_SEP_PS = 125.0
MAX_GENERATION_ATTEMPTS = 10000

HRT_STOP_THRESHOLD = 6e-6
MAX_SIC_ITERATIONS = 3

ANTENNA_STRIDE = 2
TIME_STRIDE = 2
REFINE_R = 50.0
REFINE_THETA = 0.25
REFINE_TAU_NS = 0.035

MAX_JOINT_SWEEPS = 3
POWELL_MAXITER = 100
POWELL_FTOL = 1e-10

USERS = []

# ==================== Signal ====================

def energy(x):
    return float(np.vdot(x, x).real)

def user_tau(u):
    return u["tau_ns"] * 1e-9

def clock_offset(u):
    return user_tau(u) - u["range_m"] / C

def reconstruct(t, x, R, theta, tau):
    rx, _ = received_signal(
        t, float(R), float(theta),
        clock_offset=float(tau) - float(R) / C,
        antenna_x=x
    )
    return rx

def build_time_axis(x):
    delays = np.concatenate([
        propagation_delays(
            u["range_m"], u["theta_deg"], clock_offset(u), x
        ) for u in USERS
    ])
    margin = 8 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt

def generate_signal(t, x):
    y = np.zeros((len(x), len(t)), dtype=np.complex128)
    for u in USERS:
        y += reconstruct(t, x, u["range_m"], u["theta_deg"], user_tau(u))
    return y

# ==================== Assisted PRT ====================

def nearest_indices(grid, value, count):
    idx = np.searchsorted(grid, value)
    start = int(np.clip(
        idx - count // 2, 0, max(len(grid) - count, 0)
    ))
    return np.arange(start, min(start + count, len(grid)))

def build_p_grid():
    hints = [-np.sin(np.deg2rad(u["theta_deg"])) / C for u in USERS]
    indices = np.unique(np.concatenate([
        nearest_indices(P_GRID_GLOBAL, p, N_LOCAL_P)
        for p in hints
    ]))
    return P_GRID_GLOBAL[indices]

def build_tau_indices(t):
    hints = np.array([user_tau(u) for u in USERS])
    lo = nearest_indices(t, hints.min(), N_LOCAL_TAU)
    hi = nearest_indices(t, hints.max(), N_LOCAL_TAU)
    return np.arange(lo[0], hi[-1] + 1)

def run_prt(mf, t, x, p_grid, tau_indices):
    prt = parabolic_radon_transform(mf, x, p_grid, Q_GRID)
    P, Q = np.meshgrid(p_grid, Q_GRID, indexing="ij")
    ranges = pq_to_range(P, Q)

    valid = (
        np.isfinite(ranges)
        & (ranges >= PRT_R_MIN)
        & (ranges <= PRT_R_MAX)
    )

    score = np.where(
        valid[None, :, :], np.abs(prt[tau_indices]), -np.inf
    )

    ti, pi, qi = np.unravel_index(np.argmax(score), score.shape)
    p, q = p_grid[pi], Q_GRID[qi]

    return {
        "R": float(pq_to_range(p, q)),
        "theta": float(p_to_theta_deg(p)),
        "tau": float(t[tau_indices[ti]]),
        "peak": float(score[ti, pi, qi])
    }

# ==================== Subsampling ====================

def subsample(y, t, x):
    return (
        y[::ANTENNA_STRIDE, ::TIME_STRIDE],
        t[::TIME_STRIDE],
        x[::ANTENNA_STRIDE]
    )


# ==================== Joint Powell ====================

def parameter_bounds(s):
    R, theta, tau_ns = s["R"], s["theta"], s["tau"] * 1e9
    return [
        (max(PRT_R_MIN, R - REFINE_R),
         min(PRT_R_MAX, R + REFINE_R)),
        (max(PRT_THETA_MIN, theta - REFINE_THETA),
         min(PRT_THETA_MAX, theta + REFINE_THETA)),
        (tau_ns - REFINE_TAU_NS, tau_ns + REFINE_TAU_NS)
    ]

def template_vector(t, x, z):
    return reconstruct(t, x, z[0], z[1], z[2] * 1e-9)

def joint_refit(y, vectors, bounds, t, x):
    ys, ts, xs = subsample(y, t, x)
    vectors = [np.array(v, dtype=float) for v in vectors]
    templates = [template_vector(ts, xs, v) for v in vectors]
    scale = max(energy(ys), 1e-30)

    def fit_energy(signal, components):
        total = np.zeros_like(signal)
        for s in components:
            total += s
        return energy(signal - total)

    for _ in range(MAX_JOINT_SWEEPS):
        old = fit_energy(ys, templates)

        for j in range(len(vectors)):
            def objective(z):
                trial = templates.copy()
                trial[j] = template_vector(ts, xs, z)
                return fit_energy(ys, trial) / scale

            result = minimize(
                objective, vectors[j],
                method="Powell", bounds=bounds[j],
                options={
                    "maxiter": POWELL_MAXITER,
                    "ftol": POWELL_FTOL,
                    "xtol": 1e-5
                }
            )

            if (
                np.all(np.isfinite(result.x))
                and np.isfinite(result.fun)
                and result.fun < objective(vectors[j])
            ):
                vectors[j] = result.x
                templates[j] = template_vector(ts, xs, result.x)

        new = fit_energy(ys, templates)
        if (old - new) / scale < 1e-7:
            break

    total = np.zeros_like(y)
    for v in vectors:
        total += template_vector(t, x, v)

    residual = y - total
    return vectors, residual, energy(residual)

# ==================== SIC ====================

def run_sic(y, t, x):
    p_grid = build_p_grid()
    tau_indices = build_tau_indices(t)
    residual = y.copy()
    vectors, bounds, detected = [], [], []

    for _ in range(MAX_SIC_ITERATIONS):
        previous_energy = energy(residual)

        mf = matched_filter(residual)
        prt = run_prt(mf, t, x, p_grid, tau_indices)

        roi = build_roi(
            prt["R"], prt["theta"], prt["tau"], dp=DP,
            range_bounds=(PRT_R_MIN, PRT_R_MAX),
            theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX)
        )

        R, theta, tau, peak = run_hrt(mf, t, x, roi)

        if peak <= HRT_STOP_THRESHOLD:
            break

        hrt = {"R": R, "theta": theta, "tau": tau}
        z = np.array([R, theta, tau * 1e9])
        trial_vectors = [v.copy() for v in vectors] + [z]
        trial_bounds = bounds + [parameter_bounds(hrt)]
        refined, trial_residual, E = joint_refit(y, trial_vectors, trial_bounds, t, x)
        if not np.isfinite(E) or E >= previous_energy:
            break
        vectors, bounds, residual = refined, trial_bounds, trial_residual
        detected = []

        for v in vectors:
            R_hat, theta_hat = float(v[0]), float(v[1])
            tau_hat = float(v[2]) * 1e-9
            px, py = polar_to_xy(R_hat, theta_hat)

            detected.append({
                "range_m": R_hat,
                "theta_deg": theta_hat,
                "tau_s": tau_hat,
                "clock_offset_s": tau_hat - R_hat / C,
                "x_m": float(px),
                "y_m": float(py)
            })

    return detected, residual

# ==================== Monte Carlo Users ====================

def separated(a, b):
    """At least one coordinate meets the adopted experimental resolution."""
    return (
        abs(a["range_m"] - b["range_m"]) >= MIN_RANGE_SEP_M
        or abs(a["theta_deg"] - b["theta_deg"]) >= MIN_ANGLE_SEP_DEG
        or abs(a["tau_ns"] - b["tau_ns"]) >= MIN_TOA_SEP_PS * 1e-3
    )


def generate_users(rng, n):
    if n not in (1, 2, 3):
        raise ValueError("User count must be 1, 2, or 3.")
    # Reject the whole scene to retain the conditional independent distribution.
    for _ in range(MAX_GENERATION_ATTEMPTS):
        users = []
        for _ in range(n):
            R = float(rng.uniform(R_MIN, R_MAX))
            theta = float(rng.uniform(TH_MIN, TH_MAX))
            tau = float(rng.uniform(TOA_MIN_NS, TOA_MAX_NS))
            users.append({
                "range_m": R, "theta_deg": theta, "tau_ns": tau,
                "tx_delay_ns": tau - R / C * 1e9,
            })
        if all(separated(users[i], users[j])
               for i in range(n) for j in range(i)):
            return users
    raise RuntimeError("Cannot generate a scene satisfying pairwise separation.")


# ==================== Matching and Statistics ====================

def positions_from_users(users, estimated=False):
    return np.asarray([
        [u["x_m"], u["y_m"]] if estimated
        else polar_to_xy(u["range_m"], u["theta_deg"])
        for u in users
    ], dtype=float).reshape(-1, 2)


def localization_errors(truth, detected):
    """Errors in truth order, from a minimum-total-distance assignment."""
    if len(truth) != len(detected) or not truth:
        return np.array([])
    a = positions_from_users(truth)
    b = positions_from_users(detected, estimated=True)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    errors = np.empty(len(truth))
    errors[rows] = distances[rows, cols]
    return errors


def summarize(group):
    errors = np.asarray([
        e for r in group if r["status"] == "SUCCESS" for e in r["errors"]
    ], dtype=float)
    total = len(group)
    return {
        "trials": total,
        "matched_users": int(errors.size),
        "ale_m": float(np.mean(errors)) if errors.size else np.nan,
        "rmse_m": float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan,
        "p95_m": float(np.percentile(errors, 95)) if errors.size else np.nan,
        "max_m": float(np.max(errors)) if errors.size else np.nan,
    }


def print_summary(records):
    stats = {n: summarize([r for r in records if r["true_n"] == n])
             for n in (1, 2, 3)}
    stats["all"] = summarize(records)
    print("\n" + "=" * 112)
    print(f"{'True N':>7} {'Trials':>7} "
          f"{'Matched':>8} {'ALE(m)':>11} {'RMSE(m)':>11} {'P95(m)':>11} {'Max(m)':>11}")
    for key in (1, 2, 3, "all"):
        s = stats[key]
        label = "ALL" if key == "all" else str(key)
        print(f"{label:>7} {s['trials']:>7} "
              f"{s['matched_users']:>8} {s['ale_m']:>11.6f} "
              f"{s['rmse_m']:>11.6f} {s['p95_m']:>11.6f} {s['max_m']:>11.6f}")
    print("=" * 112)
    return stats


# ==================== Visualization ====================

def draw_sector(ax):
    angles = np.deg2rad(np.linspace(TH_MIN, TH_MAX, 400))
    for radius in (R_MIN, R_MAX):
        ax.plot(radius * np.sin(angles), radius * np.cos(angles), "k--", lw=1, alpha=0.5)
    for theta in (TH_MIN, TH_MAX):
        ax.plot(np.array([R_MIN, R_MAX]) * np.sin(np.deg2rad(theta)),
                np.array([R_MIN, R_MAX]) * np.cos(np.deg2rad(theta)),
                "k--", lw=1, alpha=0.5)


def plot_results(stats, records):
    ns = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [stats[n]["ale_m"] for n in ns], "o-", label="ALE")
    ax.plot(ns, [stats[n]["rmse_m"] for n in ns], "s-", label="RMSE")
    ax.set(xlabel="Number of true users", ylabel="Error (m)", xticks=ns,
           title="Localization Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in (1, 2, 3, "all"):
        errors = np.sort([e for r in records
                          if r["status"] == "SUCCESS" and (key == "all" or r["true_n"] == key)
                          for e in r["errors"]])
        if errors.size:
            ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors),
                    label="All" if key == "all" else f"{key} user(s)")
    ax.set(xlabel="2-D Localization Error (m)", ylabel="CDF",
           title="Localization Error CDF")
    if ax.lines:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid error samples", transform=ax.transAxes, ha="center")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    good = [r for r in records if r["status"] == "SUCCESS"]
    errors = np.asarray([e for r in good for e in r["errors"]])
    if errors.size:
        positions = np.vstack([positions_from_users(r["truth"]) for r in good])
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=errors,
                        s=48, cmap="viridis", edgecolors="none", label="User positions")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        fig.colorbar(sc, cax=cax).set_label("Absolute Localization Error (m)")
    draw_sector(ax)
    xmax = R_MAX * np.sin(np.deg2rad(max(abs(TH_MIN), abs(TH_MAX))))
    ax.set(xlim=(-xmax - 25, xmax + 25), ylim=(0, R_MAX + 25),
           xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    s = stats["all"]
    ax.text(0.02, 0.98,
            f"ALE={s['ale_m']:.3f} m\nRMSE={s['rmse_m']:.3f} m",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    if errors.size:
        ax.legend(loc="lower center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No valid error samples",
                transform=ax.transAxes, ha="center")
    fig.tight_layout()
    plt.show()


# ==================== Main ====================

def monte_carlo():
    global USERS
    assert PRT_R_MIN <= R_MIN < R_MAX <= PRT_R_MAX
    assert PRT_THETA_MIN <= TH_MIN < TH_MAX <= PRT_THETA_MAX
    assert TOA_MIN_NS < TOA_MAX_NS
    assert TOA_MIN_NS > R_MAX / C * 1e9
    assert set(N_MC_BY_USER_COUNT) == {1, 2, 3}
    assert all(isinstance(count, int) and count > 0
               for count in N_MC_BY_USER_COUNT.values())
    rng = np.random.default_rng(MC_SEED)
    schedule = np.repeat([1, 2, 3], [N_MC_BY_USER_COUNT[n] for n in (1, 2, 3)])
    rng.shuffle(schedule)
    x = ARRAY.positions(SYSTEM)
    records = []
    progress = tqdm(schedule, desc="Monte Carlo", unit="trial")
    for i, n in enumerate(progress):
        USERS = generate_users(rng, int(n))
        t = build_time_axis(x)
        clean_rx = generate_signal(t, x)
        noise_seed = MC_SEED + i + 1
        noisy_rx, _ = add_awgn(clean_rx, rng=np.random.default_rng(noise_seed))
        with redirect_stdout(io.StringIO()):
            detected, residual = run_sic(noisy_rx, t, x)
        n_true, n_hat = len(USERS), len(detected)
        status = "SUCCESS" if n_hat == n_true else "OVER" if n_hat > n_true else "MISS"
        # Exclude the entire scene from errors whenever the count is incorrect.
        errors = localization_errors(USERS, detected) if status == "SUCCESS" else np.array([])
        record = {
            "trial": i + 1, "noise_seed": noise_seed,
            "true_n": n_true, "detected_n": n_hat, "status": status,
            "truth": USERS, "estimated": detected, "errors": errors,
            "initial_energy": energy(noisy_rx), "residual_energy": energy(residual),
        }
        records.append(record)
    stats = print_summary(records)
    plot_results(stats, records)
    return records, stats


if __name__ == "__main__":
    monte_carlo()
