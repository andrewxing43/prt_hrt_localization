
"""Monte Carlo: two-user temporal resolution, PRT -> ROI -> HRT -> Powell -> SIC."""

from pathlib import Path
from contextlib import redirect_stdout
import sys
import io
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

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

# ==================== Settings ====================

N_MC = 200
DELTA_TAU_PS = 15.625*8
MC_SEED = 2026

HRT_STOP_THRESHOLD = 6e-6
MAX_SIC_ITERATIONS = 3

SEARCH_R_STEP = 4.0
SEARCH_THETA_OFFSETS = [-0.15, 0.0, 0.15]
SEARCH_TAU_OFFSETS_NS = [-0.015, 0.0, 0.015]

ANTENNA_STRIDE = 2
TIME_STRIDE = 2
N_CANDIDATES = 2

SEP_R = 8.0
SEP_THETA = 0.01
SEP_TAU_PS = 1.0

REFINE_R = 15.0
REFINE_THETA = 0.25
REFINE_TAU_NS = 0.035

MAX_JOINT_SWEEPS = 3
POWELL_MAXITER = 100
POWELL_FTOL = 1e-10

USERS = []

# ==================== Signal Model ====================

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
        nearest_indices(P_GRID_GLOBAL, p, N_LOCAL_P) for p in hints
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
        valid[None, :, :],
        np.abs(prt[tau_indices]),
        -np.inf
    )

    ti, pi, qi = np.unravel_index(np.argmax(score), score.shape)
    p, q = p_grid[pi], Q_GRID[qi]

    return {
        "R": float(pq_to_range(p, q)),
        "theta": float(p_to_theta_deg(p)),
        "tau": float(t[tau_indices[ti]]),
        "peak": float(score[ti, pi, qi])
    }

# ==================== Exact Model Search ====================

def subsample(y, t, x):
    return (
        y[::ANTENNA_STRIDE, ::TIME_STRIDE],
        t[::TIME_STRIDE],
        x[::ANTENNA_STRIDE]
    )

def search_candidates(residual, t, x, seeds):
    ys, ts, xs = subsample(residual, t, x)

    R_grid = np.unique(np.r_[
        np.arange(PRT_R_MIN, PRT_R_MAX, SEARCH_R_STEP),
        PRT_R_MIN, PRT_R_MAX
    ])

    theta_grid = np.unique([
        u["theta_deg"] + d
        for u in USERS
        for d in SEARCH_THETA_OFFSETS
        if PRT_THETA_MIN <= u["theta_deg"] + d <= PRT_THETA_MAX
    ])

    tau_hints = [user_tau(u) for u in USERS]
    tau_centers = [min(tau_hints), max(tau_hints)]

    tau_grid = np.unique([
        tc + d * 1e-9
        for tc in tau_centers
        for d in SEARCH_TAU_OFFSETS_NS
    ])

    candidates = []

    def add_candidate(R, theta, tau):
        template = reconstruct(ts, xs, R, theta, tau)
        score = energy(ys) - energy(ys - template)
        candidates.append({
            "R": float(R),
            "theta": float(theta),
            "tau": float(tau),
            "score": float(score)
        })

    for theta in theta_grid:
        for tau in tau_grid:
            for R in R_grid:
                add_candidate(R, theta, tau)

    for s in seeds:
        if (
            PRT_R_MIN <= s["R"] <= PRT_R_MAX
            and PRT_THETA_MIN <= s["theta"] <= PRT_THETA_MAX
        ):
            add_candidate(s["R"], s["theta"], s["tau"])

    candidates.sort(key=lambda z: z["score"], reverse=True)

    selected = []
    for c in candidates:
        duplicate = any(
            abs(c["R"] - v["R"]) < SEP_R
            and abs(c["theta"] - v["theta"]) < SEP_THETA
            and abs(c["tau"] - v["tau"]) < SEP_TAU_PS * 1e-12
            for v in selected
        )
        if not duplicate:
            selected.append(c)
        if len(selected) >= N_CANDIDATES:
            break

    return selected

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
                method="Powell",
                bounds=bounds[j],
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

    for iteration in range(1, MAX_SIC_ITERATIONS + 1):
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

        candidates = search_candidates(
            residual, t, x,
            seeds=[
                prt,
                {"R": R, "theta": theta, "tau": tau}
            ]
        )

        best = None
        best_energy = previous_energy

        for c in candidates:
            z = np.array([c["R"], c["theta"], c["tau"] * 1e9])
            trial_vectors = [v.copy() for v in vectors] + [z]
            trial_bounds = bounds + [parameter_bounds(c)]

            refined, trial_residual, E = joint_refit(
                y, trial_vectors, trial_bounds, t, x
            )

            if E < best_energy:
                best_energy = E
                best = (refined, trial_bounds, trial_residual)

        if best is None:
            break

        vectors, bounds, residual = best
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

# ==================== Monte Carlo ====================

def monte_carlo():
    global USERS

    rng = np.random.default_rng(MC_SEED)
    x = ARRAY.positions(SYSTEM)
    success = 0
    counts = np.zeros(MAX_SIC_ITERATIONS + 1, dtype=int)

    progress = tqdm(range(N_MC), desc="Monte Carlo", unit="trial")

    for i in progress:
        R = rng.uniform(PRT_R_MIN, PRT_R_MAX)
        theta = rng.uniform(PRT_THETA_MIN, PRT_THETA_MAX)
        tau_ns = rng.uniform(299.9, 300.1)

        USERS = [
            {"range_m": R, "theta_deg": theta, "tau_ns": tau_ns},
            {"range_m": R, "theta_deg": theta,
             "tau_ns": tau_ns + DELTA_TAU_PS * 1e-3}
        ]

        t = build_time_axis(x)
        clean_rx = generate_signal(t, x)

        noisy_rx, _ = add_awgn(
            clean_rx,
            rng=np.random.default_rng(MC_SEED + i + 1)
        )

        with redirect_stdout(io.StringIO()):
            detected, residual = run_sic(noisy_rx, t, x)

        n = len(detected)
        counts[n] += 1
        success += (n == 2)

        progress.set_postfix(
            success=success,
            P_detect=f"{success / (i + 1):.2%}"
        )

    probability = success / N_MC

    print("\n" + "=" * 50)
    print("MONTE CARLO TEMPORAL RESOLUTION")
    print("=" * 50)
    print(f"Trials             : {N_MC}")
    print(f"TOA separation     : {DELTA_TAU_PS:.3f} ps")
    print(f"Successful trials  : {success}")
    print(f"Detection probability: {probability:.2%}")
    print("-" * 50)

    for n, count in enumerate(counts):
        print(f"Detected {n} users   : {count:3d} / {N_MC}")

    print("=" * 50)

if __name__ == "__main__":
    monte_carlo()
