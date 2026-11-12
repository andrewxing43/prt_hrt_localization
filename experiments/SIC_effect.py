
"""
SIC necessity demonstration:
    User 1: R = 70 m,  theta = 0 deg
    User 2: R = 300 m, theta = 0 deg

Steps:
1. Generate two-user superposed signal.
2. Add AWGN.
3. Compute PRT distance response before SIC.
4. Estimate the first (stronger) user from PRT.
5. Refine the first user's parameters using the exact signal model.
6. Reconstruct and subtract the estimated first user.
7. Compute PRT distance response after SIC.
8. Plot before/after PRT peaks.

Run from the existing project:
    python experiments/sic_prt_demo.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import (
    propagation_delays,
    received_signal,
)
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import (
    pq_to_range,
    p_to_theta_deg,
)
from localization.pipeline import (
    P_GRID_GLOBAL,
    Q_GRID,
    PRT_R_MIN,
    PRT_R_MAX,
    N_LOCAL_P,
    N_LOCAL_TAU,
)


# ============================================================
# 1. Experiment settings
# ============================================================

R1 =60.0
R2 = 100.0

THETA1 = 0.0
THETA2 = 0.0

BASE_CLOCK_OFFSET = 10e-9

# Same center arrival time for both users
COMMON_TAU = max(R1, R2) / C + BASE_CLOCK_OFFSET

ADD_NOISE = True
NOISE_SEED = SIM.rng_seed

# SIC gain:
# "fixed" follows multi_user_SIC.py default.
GAIN_MODE = "fixed"

# Refine the first user's parameters around its PRT estimate.
REFINE_R_RADIUS = 25.0
REFINE_THETA_RADIUS = 0.5
REFINE_TAU_RADIUS_NS = 0.05

# Plot settings
FIGURE_SIZE = (12, 10)
SAVE_FIGURE = True
OUTPUT_PATH = ROOT / "sic_prt_demo.png"

EPS = 1e-30


# ============================================================
# 2. Helper functions
# ============================================================

def energy(x):
    return float(np.vdot(x, x).real)


def nearest_indices(grid, value, count):
    idx = np.searchsorted(grid, value)
    start = int(np.clip(
        idx - count // 2,
        0,
        max(len(grid) - count, 0),
    ))
    return np.arange(start, min(start + count, len(grid)))


def clock_offset_for_user(R):
    return COMMON_TAU - R / C


def build_time_axis(antenna_x):
    delays = []

    for R, theta in [(R1, THETA1), (R2, THETA2)]:
        d = propagation_delays(
            R,
            theta,
            clock_offset_for_user(R),
            antenna_x,
        )
        delays.append(d)

    delays = np.concatenate(delays)

    margin = 8.0 * SYSTEM.gaussian_sigma

    n0 = int(np.floor(
        (delays.min() - margin) / SYSTEM.dt
    ))
    n1 = int(np.ceil(
        (delays.max() + margin) / SYSTEM.dt
    ))

    return np.arange(n0, n1 + 1) * SYSTEM.dt


def reconstruct_user(time_axis, antenna_x, R, theta, tau):
    rx, _ = received_signal(
        time_axis,
        float(R),
        float(theta),
        clock_offset=float(tau) - float(R) / C,
        antenna_x=antenna_x,
    )
    return rx


def generate_two_user_signal(time_axis, antenna_x):
    rx1 = reconstruct_user(
        time_axis, antenna_x,
        R1, THETA1, COMMON_TAU,
    )

    rx2 = reconstruct_user(
        time_axis, antenna_x,
        R2, THETA2, COMMON_TAU,
    )

    return rx1, rx2, rx1 + rx2


# ============================================================
# 3. PRT search grid
# ============================================================

def build_local_p_grid():
    p_hint = -np.sin(np.deg2rad(0.0)) / C

    indices = nearest_indices(
        P_GRID_GLOBAL,
        p_hint,
        N_LOCAL_P,
    )

    return P_GRID_GLOBAL[indices]


def build_local_tau_indices(time_axis):
    return nearest_indices(
        time_axis,
        COMMON_TAU,
        N_LOCAL_TAU,
    )


def valid_pq_mask(p_grid):
    P, Q = np.meshgrid(
        p_grid,
        Q_GRID,
        indexing="ij",
    )

    ranges = pq_to_range(P, Q)

    return (
        np.isfinite(ranges)
        & (ranges >= PRT_R_MIN)
        & (ranges <= PRT_R_MAX)
    )


# ============================================================
# 4. Compute PRT distance response
# ============================================================

def calculate_prt_response(
    rx,
    time_axis,
    antenna_x,
    p_grid,
    tau_indices,
):
    # Matched filter
    mf_rx = matched_filter(rx)

    # Original GPU PRT
    prt = parabolic_radon_transform(
        mf_rx,
        antenna_x,
        p_grid,
        Q_GRID,
    )

    # Only evaluate the local arrival-time window
    magnitude = np.abs(prt[tau_indices])

    # Restrict physical range
    valid = valid_pq_mask(p_grid)

    score = np.where(
        valid[None, :, :],
        magnitude,
        -np.inf,
    )

    # Global PRT peak
    ti, pi, qi = np.unravel_index(
        np.argmax(score),
        score.shape,
    )

    p_hat = float(p_grid[pi])
    q_hat = float(Q_GRID[qi])

    peak = {
        "R": float(pq_to_range(p_hat, q_hat)),
        "theta": float(p_to_theta_deg(p_hat)),
        "tau": float(time_axis[tau_indices[ti]]),
        "value": float(score[ti, pi, qi]),
    }

    # For visualization, maximize over local tau and p
    # for each q-grid point.
    q_response = np.max(score, axis=(0, 1))

    # Map q to R for each p; the angular search is narrowly
    # centered on 0 degrees. Use the p closest to zero for
    # a common distance axis.
    p_reference = float(
        p_grid[np.argmin(np.abs(p_grid))]
    )

    ranges = np.asarray(
        pq_to_range(p_reference, Q_GRID)
    )

    mask = (
        np.isfinite(ranges)
        & (ranges >= PRT_R_MIN)
        & (ranges <= PRT_R_MAX)
        & np.isfinite(q_response)
    )

    ranges = ranges[mask]
    response = q_response[mask]

    order = np.argsort(ranges)

    return {
        "ranges": ranges[order],
        "response": response[order],
        "peak": peak,
    }


# ============================================================
# 5. First-user parameter refinement
# ============================================================

def estimate_first_user(
    rx,
    time_axis,
    antenna_x,
    prt_peak,
):
    """
    Estimate the first user using:
      PRT coarse initialization + exact-model refinement.

    No true user position is passed to this function.

    Optimization variables:
        R [m], theta [deg], delta_tau [ns]
    """

    R0 = prt_peak["R"]
    theta0 = prt_peak["theta"]
    tau0 = prt_peak["tau"]

    r_lower = max(
        PRT_R_MIN,
        R0 - REFINE_R_RADIUS,
    )
    r_upper = min(
        PRT_R_MAX,
        R0 + REFINE_R_RADIUS,
    )

    theta_lower = theta0 - REFINE_THETA_RADIUS
    theta_upper = theta0 + REFINE_THETA_RADIUS

    bounds = [
        (r_lower, r_upper),
        (theta_lower, theta_upper),
        (-REFINE_TAU_RADIUS_NS, REFINE_TAU_RADIUS_NS),
    ]

    def objective(z):
        R, theta, delta_tau_ns = z

        tau = tau0 + delta_tau_ns * 1e-9

        template = reconstruct_user(
            time_axis,
            antenna_x,
            R,
            theta,
            tau,
        )

        if GAIN_MODE == "complex":
            denom = energy(template)

            alpha = (
                np.vdot(template, rx) / denom
                if denom > EPS else 0j
            )
        else:
            alpha = 1.0 + 0j

        residual = rx - alpha * template

        return energy(residual)

    x0 = np.array([
        np.clip(R0, r_lower, r_upper),
        theta0,
        0.0,
    ])

    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 200,
            "ftol": 1e-10,
            "xtol": 1e-6,
        },
    )

    R_hat, theta_hat, delta_tau_ns = result.x
    tau_hat = tau0 + delta_tau_ns * 1e-9

    template = reconstruct_user(
        time_axis,
        antenna_x,
        R_hat,
        theta_hat,
        tau_hat,
    )

    if GAIN_MODE == "complex":
        alpha = np.vdot(template, rx) / max(
            energy(template), EPS
        )
    else:
        alpha = 1.0 + 0j

    return {
        "R": float(R_hat),
        "theta": float(theta_hat),
        "tau": float(tau_hat),
        "alpha": complex(alpha),
        "template": alpha * template,
        "success": bool(result.success),
    }


# ============================================================
# 6. Peak analysis
# ============================================================

def print_local_peaks(name, result, top_n=8):
    x = result["ranges"]
    y = result["response"]

    peaks, _ = find_peaks(y)

    if len(peaks) == 0:
        peaks = np.array([np.argmax(y)])

    indices = peaks[np.argsort(y[peaks])[::-1]]

    print(f"\n{name}")
    print("-" * 50)

    for idx in indices[:top_n]:
        print(
            f"R = {x[idx]:8.3f} m, "
            f"PRT magnitude = {y[idx]:.6e}"
        )


def peak_near(result, target, radius=15.0):
    x = result["ranges"]
    y = result["response"]

    mask = np.abs(x - target) <= radius

    if not np.any(mask):
        return np.nan, np.nan

    inds = np.flatnonzero(mask)
    idx = inds[np.argmax(y[inds])]

    return float(x[idx]), float(y[idx])


# ============================================================
# 7. Plot before / after SIC
# ============================================================

def plot_results(
    before,
    after,
    estimated_user,
    energy_before,
    energy_after,
):
    rb = before["ranges"]
    yb = before["response"]

    ra = after["ranges"]
    ya = after["response"]

    # Same scale for all plots; never normalize each
    # curve independently.
    scale = max(float(np.max(yb)), EPS)

    yb = yb / scale
    ya = ya / scale

    fig, axes = plt.subplots(
        3, 1,
        figsize=FIGURE_SIZE,
    )

    # --------------------------------------------------------
    # Panel 1: PRT before SIC
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        rb, yb,
        color="tab:blue",
        linewidth=2,
        label="Two-user superposition",
    )

    ax.axvline(
        R1,
        color="tab:red",
        linestyle="--",
        label=f"True User 1: {R1:g} m",
    )

    ax.axvline(
        R2,
        color="tab:green",
        linestyle="--",
        label=f"True User 2: {R2:g} m",
    )

    ax.axvline(
        before["peak"]["R"],
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"PRT peak: "
            f"{before['peak']['R']:.2f} m"
        ),
    )

    ax.set_title(
        "(a) Before SIC: PRT of Two-User Superposition"
    )
    ax.set_ylabel("Normalized PRT magnitude")
    ax.set_xlim(45, 400)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # Panel 2: PRT after SIC
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        ra, ya,
        color="tab:orange",
        linewidth=2,
        label="Residual after subtracting User 1",
    )

    ax.axvline(
        R1,
        color="tab:red",
        linestyle="--",
        label=f"True User 1: {R1:g} m",
    )

    ax.axvline(
        R2,
        color="tab:green",
        linestyle="--",
        label=f"True User 2: {R2:g} m",
    )

    ax.axvline(
        after["peak"]["R"],
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Residual PRT peak: "
            f"{after['peak']['R']:.2f} m"
        ),
    )

    ax.set_title(
        "(b) After SIC: PRT of the Residual Signal"
    )
    ax.set_ylabel("Normalized PRT magnitude")
    ax.set_xlim(45, 400)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # Panel 3: Direct comparison
    # --------------------------------------------------------

    ax = axes[2]

    ax.plot(
        rb, yb,
        color="tab:blue",
        linewidth=2,
        label="Before SIC",
    )

    ax.plot(
        ra, ya,
        color="tab:orange",
        linewidth=2,
        label="After SIC",
    )

    ax.axvline(
        R1,
        color="tab:red",
        linestyle="--",
        alpha=0.7,
    )

    ax.axvline(
        R2,
        color="tab:green",
        linestyle="--",
        alpha=0.7,
    )

    ax.set_title(
        "(c) PRT Magnitude Comparison: Before vs After SIC"
    )
    ax.set_xlabel("Range R (m)")
    ax.set_ylabel("Normalized PRT magnitude")
    ax.set_xlim(45, 400)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Why SIC Is Needed: Two Co-Angular Users\n"
        f"User 1 = ({R1:g} m, {THETA1:g} deg), "
        f"User 2 = ({R2:g} m, {THETA2:g} deg)\n"
        f"Estimated first user: "
        f"({estimated_user['R']:.2f} m, "
        f"{estimated_user['theta']:.3f} deg)",
        fontsize=13,
    )

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(
            OUTPUT_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"\nFigure saved to: {OUTPUT_PATH}")

    plt.show()


# ============================================================
# 8. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1: Generate two-user signal
    # --------------------------------------------------------

    antenna_x = ARRAY.positions(SYSTEM)
    time_axis = build_time_axis(antenna_x)

    rx1, rx2, clean_rx = generate_two_user_signal(
        time_axis,
        antenna_x,
    )

    if ADD_NOISE:
        noisy_rx, noise_variance = add_awgn(
            clean_rx,
            rng=np.random.default_rng(NOISE_SEED),
        )
    else:
        noisy_rx = clean_rx.copy()
        noise_variance = 0.0

    print("=" * 60)
    print("SIC necessity experiment")
    print("=" * 60)

    print(f"User 1: R={R1} m, theta={THETA1} deg")
    print(f"User 2: R={R2} m, theta={THETA2} deg")

    print(f"Noise variance: {noise_variance:.6e}")
    print(f"User 1 energy: {energy(rx1):.6e}")
    print(f"User 2 energy: {energy(rx2):.6e}")

    print(
        "User 1 / User 2 energy ratio: "
        f"{10 * np.log10(energy(rx1) / energy(rx2)):.2f} dB"
    )

    # --------------------------------------------------------
    # Step 2: PRT before SIC
    # --------------------------------------------------------

    p_grid = build_local_p_grid()

    tau_indices = build_local_tau_indices(
        time_axis
    )

    before = calculate_prt_response(
        noisy_rx,
        time_axis,
        antenna_x,
        p_grid,
        tau_indices,
    )

    print("\n========== BEFORE SIC ==========")

    print(
        f"Strongest PRT peak: "
        f"R={before['peak']['R']:.3f} m, "
        f"theta={before['peak']['theta']:.4f} deg"
    )

    print_local_peaks(
        "PRT peaks before SIC",
        before,
    )

    # --------------------------------------------------------
    # Step 3: Estimate first user
    # --------------------------------------------------------

    first_user = estimate_first_user(
        noisy_rx,
        time_axis,
        antenna_x,
        before["peak"],
    )

    print("\n========== FIRST USER ==========")

    print(
        f"Estimated R: "
        f"{first_user['R']:.6f} m"
    )

    print(
        f"Estimated theta: "
        f"{first_user['theta']:.6f} deg"
    )

    print(
        f"Estimated tau: "
        f"{first_user['tau'] * 1e9:.6f} ns"
    )

    print(
        f"Optimization success: "
        f"{first_user['success']}"
    )

    # --------------------------------------------------------
    # Step 4: SIC subtraction
    # --------------------------------------------------------

    residual = (
        noisy_rx - first_user["template"]
    )

    energy_before = energy(noisy_rx)
    energy_after = energy(residual)

    print("\n========== SIC ==========")

    print(
        f"Energy before SIC: "
        f"{energy_before:.6e}"
    )

    print(
        f"Energy after SIC: "
        f"{energy_after:.6e}"
    )

    print(
        f"Energy reduction: "
        f"{100 * (1 - energy_after / energy_before):.3f}%"
    )

    # --------------------------------------------------------
    # Step 5: PRT after SIC
    # --------------------------------------------------------

    after = calculate_prt_response(
        residual,
        time_axis,
        antenna_x,
        p_grid,
        tau_indices,
    )

    print("\n========== AFTER SIC ==========")

    print(
        f"Strongest residual PRT peak: "
        f"R={after['peak']['R']:.3f} m, "
        f"theta={after['peak']['theta']:.4f} deg"
    )

    print_local_peaks(
        "PRT peaks after SIC",
        after,
    )

    # --------------------------------------------------------
    # Step 6: Compare both user responses
    # --------------------------------------------------------

    _, peak1_before = peak_near(
        before, R1
    )
    _, peak2_before = peak_near(
        before, R2
    )

    _, peak1_after = peak_near(
        after, R1
    )
    _, peak2_after = peak_near(
        after, R2
    )

    print("\n========== PEAK COMPARISON ==========")

    print(
        f"{R1:g} m before SIC : {peak1_before:.6e}"
    )
    print(
        f"{R1:g} m after SIC  : {peak1_after:.6e}"
    )

    print(
        f"{R2:g} m before SIC: {peak2_before:.6e}"
    )
    print(
        f"{R2:g} m after SIC : {peak2_after:.6e}"
    )

    ratio_before = (
        peak2_before / max(peak1_before, EPS)
    )
    ratio_after = (
        peak2_after / max(peak1_after, EPS)
    )

    print(
        f"{R2:g}m/{R1:g}m ratio before: "
        f"{ratio_before:.6f}"
    )

    print(
        f"{R2:g}m/{R1:g}m ratio after : "
        f"{ratio_after:.6f}"
    )

    if np.isfinite(ratio_before) and np.isfinite(ratio_after):
        if ratio_before > 0 and ratio_after > 0:
            print(
                f"Relative prominence improvement ({R2:g}m/{R1:g}m): "
                f"{20 * np.log10(ratio_after / ratio_before):.2f} dB"
            )

    # --------------------------------------------------------
    # Step 7: Visualization
    # --------------------------------------------------------

    plot_results(
        before,
        after,
        first_user,
        energy_before,
        energy_after,
    )


if __name__ == "__main__":
    main()
