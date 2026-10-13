
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
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

# ============================================================
# Settings
# ============================================================

USERS = [
    {"range_m": 300.1, "theta_deg": 59.7},
    {"range_m": 150.1, "theta_deg": 59.7},
    {"range_m": 60.2, "theta_deg": 59.7},
]

BASE_CLOCK_OFFSET = 10e-9
COMMON_TAU = max(u["range_m"] for u in USERS) / C + BASE_CLOCK_OFFSET

HRT_STOP_THRESHOLD = 6e-6
MAX_SIC_ITERATIONS = 3

GAIN_MODE = "fixed"  # "complex" or "fixed"

SEARCH_R_MIN = PRT_R_MIN
SEARCH_R_MAX = PRT_R_MAX
SEARCH_R_STEP = 4.0
SEARCH_THETA_OFFSETS = [-0.15, 0.0, 0.15]
SEARCH_TAU_OFFSETS_NS = [-0.015, 0.0, 0.015]

OPT_ANTENNA_STRIDE = 2
OPT_TIME_STRIDE = 2
N_SEARCH_CANDIDATES = 2
CANDIDATE_SEPARATION_M = 8.0

REFINE_R_RADIUS = 15.0
REFINE_THETA_RADIUS = 0.25
REFINE_TAU_RADIUS_NS = 0.035

MAX_JOINT_SWEEPS = 3
OPT_MAXITER = 100
OPT_FTOL = 1e-10

# ============================================================
# Signal model
# ============================================================

def energy(x):
    return float(np.vdot(x, x).real)


def user_clock_offset(user):
    return COMMON_TAU - user["range_m"] / C


def build_time_axis(antenna_x):
    delays = np.concatenate([
        propagation_delays(
            u["range_m"], u["theta_deg"],
            user_clock_offset(u), antenna_x
        ) for u in USERS
    ])
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def generate_multi_user_signal(time_axis, antenna_x):
    total = np.zeros((len(antenna_x), len(time_axis)), dtype=np.complex128)
    for user in USERS:
        rx, _ = received_signal(
            time_axis, user["range_m"], user["theta_deg"],
            clock_offset=user_clock_offset(user), antenna_x=antenna_x
        )
        total += rx
    return total


def reconstruct_user(time_axis, antenna_x, R, theta, tau):
    reconstructed, _ = received_signal(
        time_axis, float(R), float(theta),
        clock_offset=float(tau) - float(R) / C,
        antenna_x=antenna_x
    )
    return reconstructed


# ============================================================
# Assisted PRT
# ============================================================

def nearest_indices(grid, value, count):
    index = np.searchsorted(grid, value)
    start = int(np.clip(index - count // 2, 0, max(len(grid) - count, 0)))
    return np.arange(start, min(start + count, len(grid)))


def build_local_p_grid():
    p_hints = [-np.sin(np.deg2rad(u["theta_deg"])) / C for u in USERS]
    indices = np.unique(np.concatenate([
        nearest_indices(P_GRID_GLOBAL, p, N_LOCAL_P) for p in p_hints
    ]))
    p_grid = P_GRID_GLOBAL[indices]
    return p_to_theta_deg(p_grid), p_grid


def get_local_tau_indices(time_axis):
    return nearest_indices(time_axis, COMMON_TAU, N_LOCAL_TAU)


def valid_pq_mask(p_grid):
    P, Q = np.meshgrid(p_grid, Q_GRID, indexing="ij")
    ranges = pq_to_range(P, Q)
    return np.isfinite(ranges) & (ranges >= PRT_R_MIN) & (ranges <= PRT_R_MAX)


def run_local_prt(mf_rx, time_axis, antenna_x, p_grid, tau_indices):
    prt = parabolic_radon_transform(mf_rx, antenna_x, p_grid, Q_GRID)
    valid = valid_pq_mask(p_grid)
    score = np.where(valid[None, :, :], np.abs(prt[tau_indices]), -np.inf)
    index = np.unravel_index(np.argmax(score), score.shape)
    ti, pi, qi = index
    p, q = float(p_grid[pi]), float(Q_GRID[qi])
    return {
        "R": float(pq_to_range(p, q)),
        "theta": float(p_to_theta_deg(p)),
        "tau": float(time_axis[int(tau_indices[ti])]),
        "peak": float(score[index])
    }


# ============================================================
# Signal fitting
# ============================================================

def fit_complex_gain(signal, template):
    denominator = energy(template)
    if denominator <= 1e-30:
        return 0j, 0.0

    if GAIN_MODE == "complex":
        alpha = np.vdot(template, signal) / denominator
    elif GAIN_MODE == "fixed":
        alpha = 1.0 + 0j
    else:
        raise ValueError("Invalid GAIN_MODE")

    explained = energy(signal) - energy(signal - alpha * template)
    return alpha, float(explained)


def fit_components(signal, templates):
    if not templates:
        return np.zeros(0, dtype=complex), signal.copy(), energy(signal)

    T = np.column_stack([t.reshape(-1) for t in templates])
    y = signal.reshape(-1)

    if GAIN_MODE == "complex":
        alpha = np.linalg.lstsq(T, y, rcond=1e-9)[0]
    elif GAIN_MODE == "fixed":
        alpha = np.ones(len(templates), dtype=complex)
    else:
        raise ValueError("Invalid GAIN_MODE")

    residual = (y - T @ alpha).reshape(signal.shape)
    return alpha, residual, energy(residual)


# ============================================================
# Full-range exact-model search
# ============================================================

def make_search_grid():
    grid = np.arange(SEARCH_R_MIN, SEARCH_R_MAX + 1e-9, SEARCH_R_STEP)
    return np.unique(np.clip(
        np.r_[grid, SEARCH_R_MIN, SEARCH_R_MAX],
        SEARCH_R_MIN, SEARCH_R_MAX
    ))


def make_subsampled_data(signal, time_axis, antenna_x):
    return (
        signal[::OPT_ANTENNA_STRIDE, ::OPT_TIME_STRIDE],
        time_axis[::OPT_TIME_STRIDE],
        antenna_x[::OPT_ANTENNA_STRIDE]
    )


def exact_search_candidates(
    residual, time_axis, antenna_x,
    theta_centers, tau_center, extra_seeds=None
):
    y_sub, t_sub, x_sub = make_subsampled_data(
        residual, time_axis, antenna_x
    )
    range_grid = make_search_grid()

    theta_grid = np.unique([
        float(center + offset)
        for center in theta_centers
        for offset in SEARCH_THETA_OFFSETS
        if PRT_THETA_MIN <= center + offset <= PRT_THETA_MAX
    ])
    tau_grid = [
        tau_center + offset * 1e-9 for offset in SEARCH_TAU_OFFSETS_NS
    ]

    candidates = []
    print(
        f"Exact scan: {len(range_grid)} ranges, "
        f"{len(theta_grid)} angles, {len(tau_grid)} delays"
    )

    for theta in theta_grid:
        for tau in tau_grid:
            for R in range_grid:
                template = reconstruct_user(t_sub, x_sub, R, theta, tau)
                _, explained = fit_complex_gain(y_sub, template)
                candidates.append({
                    "R": float(R), "theta": float(theta),
                    "tau": float(tau), "score": float(explained)
                })

    if extra_seeds:
        for seed in extra_seeds:
            R, theta, tau = (
                float(seed["R"]), float(seed["theta"]), float(seed["tau"])
            )
            if not (
                SEARCH_R_MIN <= R <= SEARCH_R_MAX
                and PRT_THETA_MIN <= theta <= PRT_THETA_MAX
            ):
                continue

            template = reconstruct_user(t_sub, x_sub, R, theta, tau)
            _, explained = fit_complex_gain(y_sub, template)
            candidates.append({
                "R": R, "theta": theta,
                "tau": tau, "score": float(explained)
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for candidate in candidates:
        if any(
            abs(candidate["R"] - other["R"]) < CANDIDATE_SEPARATION_M
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= N_SEARCH_CANDIDATES:
            break

    print("Top exact-model search candidates:")
    for i, c in enumerate(selected, 1):
        print(
            f"  {i}: R={c['R']:.3f} m, theta={c['theta']:.4f} deg, "
            f"tau={c['tau']*1e9:.6f} ns, explained={c['score']:.6e}"
        )
    return selected


# ============================================================
# Joint parameter optimization
# ============================================================

def parameter_bounds(seed):
    R, theta, tau_ns = (
        float(seed["R"]),
        float(seed["theta"]),
        float(seed["tau"]) * 1e9
    )
    return [
        (max(SEARCH_R_MIN, R - REFINE_R_RADIUS),
         min(SEARCH_R_MAX, R + REFINE_R_RADIUS)),
        (max(PRT_THETA_MIN, theta - REFINE_THETA_RADIUS),
         min(PRT_THETA_MAX, theta + REFINE_THETA_RADIUS)),
        (tau_ns - REFINE_TAU_RADIUS_NS, tau_ns + REFINE_TAU_RADIUS_NS)
    ]


def template_from_vector(time_axis, antenna_x, v):
    return reconstruct_user(
        time_axis, antenna_x,
        float(v[0]), float(v[1]), float(v[2]) * 1e-9
    )


def joint_parameter_refit(
    original_signal, initial_vectors, bounds, time_axis, antenna_x
):
    y_sub, t_sub, x_sub = make_subsampled_data(
        original_signal, time_axis, antenna_x
    )
    vectors = [np.asarray(v, dtype=float).copy() for v in initial_vectors]
    templates = [template_from_vector(t_sub, x_sub, v) for v in vectors]
    scale = max(energy(y_sub), 1e-30)

    for sweep in range(MAX_JOINT_SWEEPS):
        old_energy = fit_components(y_sub, templates)[2]

        for j in range(len(vectors)):
            def objective(z):
                trial = templates.copy()
                trial[j] = template_from_vector(t_sub, x_sub, z)
                return fit_components(y_sub, trial)[2] / scale

            result = minimize(
                objective, vectors[j], method="Powell",
                bounds=bounds[j],
                options={
                    "maxiter": OPT_MAXITER,
                    "ftol": OPT_FTOL, "xtol": 1e-5
                }
            )

            if (
                np.all(np.isfinite(result.x))
                and np.isfinite(result.fun)
                and result.fun < objective(vectors[j])
            ):
                vectors[j] = result.x
                templates[j] = template_from_vector(t_sub, x_sub, vectors[j])

        new_energy = fit_components(y_sub, templates)[2]
        print(
            f"  Joint sweep {sweep+1}: "
            f"{old_energy:.6e} -> {new_energy:.6e}"
        )
        if (old_energy - new_energy) / scale < 1e-7:
            break

    full_templates = [
        template_from_vector(time_axis, antenna_x, v) for v in vectors
    ]
    alpha, residual, final_energy = fit_components(
        original_signal, full_templates
    )
    return vectors, alpha, residual, final_energy, full_templates


# ============================================================
# SIC: HRT threshold + energy reduction only
# ============================================================

def run_sic(
    noisy_rx, time_axis, antenna_x,
    noise_variance=SIM.noise_power
):
    theta_grid, p_grid = build_local_p_grid()
    tau_indices = get_local_tau_indices(time_axis)

    residual = noisy_rx.copy()
    accepted, vectors, bounds = [], [], []
    theta_hints = [u["theta_deg"] for u in USERS]

    print("=" * 80)
    print("ENHANCED SIC | HRT threshold + energy reduction")
    print(f"COMMON TAU: {COMMON_TAU*1e9:.6f} ns")
    print(f"GAIN MODE: {GAIN_MODE}")
    print(f"HRT threshold: {HRT_STOP_THRESHOLD:.6e}")
    print(f"Exact search: {SEARCH_R_MIN:.1f} to {SEARCH_R_MAX:.1f} m")
    print(
        f"PRT: Np={len(p_grid)}, Ntau={len(tau_indices)}, "
        f"Nq={len(Q_GRID)}"
    )
    print("=" * 80)

    for iteration in range(1, MAX_SIC_ITERATIONS + 1):
        print("\n" + "=" * 80)
        print(f"SIC ITERATION {iteration}")
        print("=" * 80)

        previous_energy = energy(residual)

        # Step 1: PRT
        mf_residual = matched_filter(residual)
        prt = run_local_prt(
            mf_residual, time_axis, antenna_x, p_grid, tau_indices
        )
        print(
            f"PRT: R={prt['R']:.3f} m, theta={prt['theta']:.4f} deg, "
            f"tau={prt['tau']*1e9:.6f} ns, peak={prt['peak']:.6e}"
        )

        # Step 2: ROI
        roi = build_roi(
            prt["R"], prt["theta"], prt["tau"], dp=DP,
            range_bounds=(PRT_R_MIN, PRT_R_MAX),
            theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX)
        )
        print(
            f"ROI: R=[{roi.R_min:.3f}, {roi.R_max:.3f}] m, "
            f"theta=[{roi.theta_min:.3f}, {roi.theta_max:.3f}] deg"
        )

        # Step 3: HRT
        R_hrt, theta_hrt, tau_hrt, hrt_peak = run_hrt(
            mf_residual, time_axis, antenna_x, roi
        )
        print(
            f"HRT: R={R_hrt:.3f} m, theta={theta_hrt:.4f} deg, "
            f"tau={tau_hrt*1e9:.6f} ns, peak={hrt_peak:.6e}"
        )

        # Decision 1: HRT hard threshold
        if hrt_peak <= HRT_STOP_THRESHOLD:
            print(
                f"Stop: HRT peak {hrt_peak:.6e} <= "
                f"{HRT_STOP_THRESHOLD:.6e}"
            )
            break

        # Step 4: Full-range search
        candidates = exact_search_candidates(
            residual, time_axis, antenna_x,
            theta_centers=theta_hints,
            tau_center=COMMON_TAU,
            extra_seeds=[
                {"R": prt["R"], "theta": prt["theta"], "tau": prt["tau"]},
                {"R": R_hrt, "theta": theta_hrt, "tau": tau_hrt}
            ]
        )

        if not candidates:
            print("Stop: no valid candidates.")
            break

        # Step 5: Joint fitting and candidate selection
        best_trial = None
        best_energy = previous_energy

        for i, candidate in enumerate(candidates, 1):
            new_vector = np.array([
                candidate["R"], candidate["theta"],
                candidate["tau"] * 1e9
            ])

            trial_vectors = [v.copy() for v in vectors] + [new_vector]
            trial_bounds = list(bounds) + [parameter_bounds(candidate)]

            refined, alpha, trial_residual, trial_energy, templates = (
                joint_parameter_refit(
                    noisy_rx, trial_vectors, trial_bounds,
                    time_axis, antenna_x
                )
            )

            improvement = previous_energy - trial_energy

            print(
                f"Candidate {i}: R={refined[-1][0]:.3f} m, "
                f"theta={refined[-1][1]:.4f} deg, "
                f"tau={refined[-1][2]:.6f} ns"
            )
            print(
                f"  Energy: {previous_energy:.6e} -> {trial_energy:.6e}, "
                f"drop={100*improvement/max(previous_energy,1e-30):.4f}%"
            )

            # Decision 2 and duplicate detection removed.
            # Only require a strictly positive energy reduction.
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_trial = {
                    "vectors": refined,
                    "bounds": trial_bounds,
                    "alpha": alpha,
                    "residual": trial_residual,
                    "energy": trial_energy,
                    "prt_peak": prt["peak"],
                    "hrt_peak": hrt_peak,
                    "iteration": iteration
                }

        # Decision: SIC energy must decrease
        if best_trial is None:
            print("Stop: no candidate reduces residual energy.")
            break

        # Step 6: Commit joint SIC
        vectors = best_trial["vectors"]
        bounds = best_trial["bounds"]
        residual = best_trial["residual"]
        alpha = best_trial["alpha"]

        accepted.append({
            "prt_peak": best_trial["prt_peak"],
            "hrt_peak": best_trial["hrt_peak"],
            "iteration": iteration
        })

        for j, v in enumerate(vectors):
            R, theta, tau = (
                float(v[0]), float(v[1]), float(v[2]) * 1e-9
            )
            x, y = polar_to_xy(R, theta)
            accepted[j].update({
                "range_m": R,
                "theta_deg": theta,
                "tau_s": tau,
                "clock_offset_s": tau - R / C,
                "x_m": x,
                "y_m": y,
                "alpha": complex(alpha[j])
            })

        relative_drop = (
            previous_energy - energy(residual)
        ) / max(previous_energy, 1e-30)

        print(f"\nAccepted user {len(accepted)}.")
        print(f"SIC energy reduction: {100*relative_drop:.4f}%")
        print(f"Residual energy: {energy(residual):.6e}")

        for j, u in enumerate(accepted, 1):
            print(
                f"  User {j}: R={u['range_m']:.4f} m, "
                f"theta={u['theta_deg']:.4f} deg, "
                f"tau={u['tau_s']*1e9:.6f} ns"
            )

    return accepted, residual


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
    detected, residual = run_sic(
        noisy_rx, time_axis, antenna_x, noise_variance
    )

    print("\n" + "=" * 80)
    print("TRUE USERS")
    print("=" * 80)

    for i, u in enumerate(USERS, 1):
        R, theta = u["range_m"], u["theta_deg"]
        x, y = polar_to_xy(R, theta)
        print(
            f"User {i}: R={R:.6f} m, theta={theta:.6f} deg, "
            f"tau={COMMON_TAU*1e9:.6f} ns"
        )
        print(
            f"        clock={user_clock_offset(u)*1e9:.6f} ns, "
            f"x={x:.6f} m, y={y:.6f} m"
        )

    print("\n" + "=" * 80)
    print(f"SIC DETECTED USERS: {len(detected)}")
    print("=" * 80)

    for i, u in enumerate(detected, 1):
        print(
            f"User {i}: R={u['range_m']:.6f} m, "
            f"theta={u['theta_deg']:.6f} deg, "
            f"tau={u['tau_s']*1e9:.6f} ns"
        )
        print(
            f"        clock={u['clock_offset_s']*1e9:.6f} ns, "
            f"x={u['x_m']:.6f} m, y={u['y_m']:.6f} m"
        )
        print(
            f"        alpha={u['alpha']}, "
            f"PRT peak={u['prt_peak']:.6e}, "
            f"HRT peak={u['hrt_peak']:.6e}, "
            f"iteration={u['iteration']}"
        )

    print(f"\nNoise variance: {noise_variance:.6e}")
    print(
        f"Total energy: {energy(noisy_rx):.6e} "
        f"-> {energy(residual):.6e}"
    )


if __name__ == "__main__":
    main()
