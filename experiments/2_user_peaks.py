
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY
from signal_model.propagation import (
    propagation_delays,
    received_signal,
)
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from localization.parameter_mapping import (
    pq_to_range,
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
# Experiment settings
# ============================================================

R1 = 50.2
R2 = 395.2

THETA_DEG = 50.1

BASE_CLOCK_OFFSET = 10e-9

# Both users have exactly the same center arrival time.
COMMON_TAU = max(R1, R2) / C + BASE_CLOCK_OFFSET

# No noise by default, to isolate coherent superposition.
ADD_NOISE = False
NOISE_SEED = 42

# Plot configuration
NORMALIZE_CURVES = True
FIGURE_SIZE = (12, 7)


# ============================================================
# Helpers
# ============================================================

def nearest_indices(grid, value, count):
    index = np.searchsorted(grid, value)

    start = int(np.clip(
        index - count // 2,
        0,
        len(grid) - count
    ))

    return np.arange(start, start + count)


def build_time_axis(antenna_x):
    delays_all = []

    for R in [R1, R2]:
        clock = COMMON_TAU - R / C

        delays = propagation_delays(
            R,
            THETA_DEG,
            clock,
            antenna_x
        )

        delays_all.append(delays)

    delays_all = np.concatenate(delays_all)

    margin = 8.0 * SYSTEM.gaussian_sigma

    n0 = int(np.floor(
        (delays_all.min() - margin) / SYSTEM.dt
    ))

    n1 = int(np.ceil(
        (delays_all.max() + margin) / SYSTEM.dt
    ))

    return np.arange(n0, n1 + 1) * SYSTEM.dt


def generate_user_signal(
    time_axis,
    antenna_x,
    R,
):
    clock_offset = COMMON_TAU - R / C

    rx, _ = received_signal(
        time_axis,
        R,
        THETA_DEG,
        clock_offset=clock_offset,
        antenna_x=antenna_x,
    )

    return rx


def build_local_p_grid():
    # Preserve the project's assisted PRT angular search.
    p_hint = -np.sin(np.deg2rad(THETA_DEG)) / C

    indices = nearest_indices(
        P_GRID_GLOBAL,
        p_hint,
        N_LOCAL_P
    )

    return P_GRID_GLOBAL[indices]


def build_local_tau_indices(time_axis):
    # Preserve the project's assisted tau search.
    return nearest_indices(
        time_axis,
        COMMON_TAU,
        N_LOCAL_TAU
    )


# ============================================================
# Extract PRT distance response
# ============================================================

def calculate_prt_distance_response(
    rx,
    time_axis,
    antenna_x,
    p_grid,
    tau_indices,
):
    # Step 1: Matched filter
    mf_rx = matched_filter(rx)

    # Step 2: Original GPU PRT
    prt = parabolic_radon_transform(
        mf_rx,
        antenna_x,
        p_grid,
        Q_GRID,
    )

    # Step 3:
    # Use the p-bin closest to the true common angle.
    #
    # This isolates the range response at a fixed p.
    # All three signals use exactly the same p and tau grid.
    #
    # It is intentionally a 1-D PRT range slice,
    # rather than a full 3-D global-peak search.

    p_true = -np.sin(np.deg2rad(THETA_DEG)) / C

    p_idx = int(np.argmin(
        np.abs(p_grid - p_true)
    ))

    p_selected = float(p_grid[p_idx])

    # Maximize magnitude over the same local tau bins
    # independently for every q.
    response_q = np.max(
        np.abs(prt[tau_indices, p_idx, :]),
        axis=0
    )

    # Step 4: Convert q to physical range R.
    ranges = pq_to_range(
        p_selected,
        Q_GRID
    )

    # Step 5: Keep valid ranges only.
    valid = (
        np.isfinite(ranges)
        & (ranges >= PRT_R_MIN)
        & (ranges <= PRT_R_MAX)
    )

    ranges = np.asarray(ranges[valid])
    response = np.asarray(response_q[valid])

    # Sort by increasing distance.
    order = np.argsort(ranges)

    ranges = ranges[order]
    response = response[order]

    peak_idx = int(np.argmax(response))

    return {
        "ranges": ranges,
        "response": response,
        "peak_R": float(ranges[peak_idx]),
        "peak_value": float(response[peak_idx]),
        "p_selected": p_selected,
        "theta_grid_deg": float(
            np.rad2deg(
                np.arcsin(
                    np.clip(-p_selected * C, -1, 1)
                )
            )
        ),
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    antenna_x = ARRAY.positions(SYSTEM)

    time_axis = build_time_axis(antenna_x)

    p_grid = build_local_p_grid()

    tau_indices = build_local_tau_indices(
        time_axis
    )

    print("=" * 75)
    print("PRT COHERENT SUPERPOSITION EXPERIMENT")
    print("=" * 75)

    print(f"User 1: R={R1:.3f} m")
    print(f"User 2: R={R2:.3f} m")
    print(f"Common theta: {THETA_DEG:.6f} deg")
    print(f"Common tau: {COMMON_TAU*1e9:.6f} ns")
    print(f"Distance midpoint: {(R1+R2)/2:.3f} m")
    print(f"PRT p-grid size: {len(p_grid)}")
    print(f"PRT tau-grid size: {len(tau_indices)}")
    print(f"PRT q-grid size: {len(Q_GRID)}")
    print("=" * 75)

    # Generate two user signals independently.
    rx1 = generate_user_signal(
        time_axis,
        antenna_x,
        R1
    )

    rx2 = generate_user_signal(
        time_axis,
        antenna_x,
        R2
    )

    # Coherent superposition (not magnitude addition).
    rx_sum = rx1 + rx2

    if ADD_NOISE:
        from signal_model.noise import add_awgn

        rng = np.random.default_rng(NOISE_SEED)

        rx_sum, _ = add_awgn(
            rx_sum,
            rng=rng
        )

    print("\nComputing PRT for user 1...")

    result1 = calculate_prt_distance_response(
        rx1,
        time_axis,
        antenna_x,
        p_grid,
        tau_indices,
    )

    print("Computing PRT for user 2...")

    result2 = calculate_prt_distance_response(
        rx2,
        time_axis,
        antenna_x,
        p_grid,
        tau_indices,
    )

    print("Computing PRT for coherent sum...")

    result_sum = calculate_prt_distance_response(
        rx_sum,
        time_axis,
        antenna_x,
        p_grid,
        tau_indices,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PRT PEAK RESULTS")
    print("=" * 75)

    print(
        f"User 1 alone: "
        f"R_peak={result1['peak_R']:.6f} m, "
        f"peak={result1['peak_value']:.6e}"
    )

    print(
        f"User 2 alone: "
        f"R_peak={result2['peak_R']:.6f} m, "
        f"peak={result2['peak_value']:.6e}"
    )

    print(
        f"Coherent sum: "
        f"R_peak={result_sum['peak_R']:.6f} m, "
        f"peak={result_sum['peak_value']:.6e}"
    )

    print(
        f"Selected p corresponds to theta="
        f"{result_sum['theta_grid_deg']:.6f} deg"
    )

    R_peak = result_sum["peak_R"]

    is_between = (
        min(R1, R2) < R_peak < max(R1, R2)
    )

    print("\n" + "=" * 75)

    if is_between:
        print(
            "RESULT: The coherent-sum PRT maximum "
            "is between the two true ranges."
        )
    else:
        print(
            "RESULT: The coherent-sum PRT maximum "
            "is NOT between the two true ranges."
        )

    print(
        f"Peak distance from midpoint: "
        f"{abs(R_peak-(R1+R2)/2):.6f} m"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ranges = result1["ranges"]

    y1 = result1["response"].copy()
    y2 = result2["response"].copy()
    ysum = result_sum["response"].copy()

    # Normalization is optional.
    # Use one common scaling factor, so relative
    # peak heights are preserved.
    if NORMALIZE_CURVES:
        scale = max(
            np.max(y1),
            np.max(y2),
            np.max(ysum),
            1e-30
        )

        y1 = y1 / scale
        y2 = y2 / scale
        ysum = ysum / scale

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE
    )

    ax.plot(
        ranges,
        y1,
        color="tab:blue",
        linewidth=1.8,
        linestyle="--",
        label=f"User 1 alone (R={R1:.1f} m)"
    )

    ax.plot(
        ranges,
        y2,
        color="tab:green",
        linewidth=1.8,
        linestyle="--",
        label=f"User 2 alone (R={R2:.1f} m)"
    )

    ax.plot(
        ranges,
        ysum,
        color="tab:red",
        linewidth=2.5,
        label="Coherent sum: User 1 + User 2"
    )

    # True positions
    ax.axvline(
        R1,
        color="tab:blue",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8
    )

    ax.axvline(
        R2,
        color="tab:green",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8
    )

    # Midpoint
    midpoint = (R1 + R2) / 2.0

    ax.axvline(
        midpoint,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Midpoint ({midpoint:.1f} m)"
    )

    # Actual combined peak
    peak_idx = int(np.argmax(ysum))

    ax.scatter(
        [ranges[peak_idx]],
        [ysum[peak_idx]],
        color="black",
        s=65,
        zorder=6
    )

    ax.axvline(
        R_peak,
        color="tab:red",
        linestyle="-.",
        linewidth=1.4,
        label=f"Combined peak ({R_peak:.2f} m)"
    )

    ax.annotate(
        f"Combined peak\nR = {R_peak:.2f} m",
        xy=(ranges[peak_idx], ysum[peak_idx]),
        xytext=(0, 30),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        arrowprops={
            "arrowstyle": "->",
            "color": "black"
        },
    )

    ax.set_title(
        "PRT Range Response: Two Coherent Users\n"
        f"Common theta = {THETA_DEG:.2f} deg, "
        f"Common tau = {COMMON_TAU*1e9:.4f} ns"
    )

    ax.set_xlabel("Estimated Range R (m)")

    if NORMALIZE_CURVES:
        ax.set_ylabel(
            "PRT Magnitude (Common-Scale Normalized)"
        )
    else:
        ax.set_ylabel("PRT Magnitude")

    ax.set_xlim(PRT_R_MIN, PRT_R_MAX)

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="best",
        fontsize=9
    )

    fig.tight_layout()

    # Display only. No savefig is called.
    plt.show()


if __name__ == "__main__":
    main()
