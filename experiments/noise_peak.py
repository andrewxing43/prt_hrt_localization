
from pathlib import Path
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays
from signal_model.noise import add_awgn
from localization.pipeline import localize

# ============================================================
# Settings
# ============================================================

RANGES_M = [50, 100, 150, 200, 300, 400]
ANGLES_DEG = [0, 20, 40, 50, 60]

N_REALIZATIONS = 30
CLOCK_OFFSET = 10e-9
SEED = SIM.rng_seed + 100

PRT_THRESHOLD = 1e-5
OUTPUT_DIR = ROOT / "results" / "noise_peak_sweep"

# ============================================================
# Time axis
# ============================================================

def build_time_axis(R, theta, antenna_x):
    delays = propagation_delays(
        R, theta,
        clock_offset=CLOCK_OFFSET,
        antenna_x=antenna_x
    )
    margin = 8.0 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


# ============================================================
# Statistics
# ============================================================

def summarize(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main experiment
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    antenna_x = ARRAY.positions(SYSTEM)
    rng = np.random.default_rng(SEED)

    rows = []
    summary = []

    print("=" * 80)
    print("PURE NOISE PRT / HRT PEAK EXPERIMENT")
    print("=" * 80)
    print(f"TX power setting : {SIM.tx_power:.6f} W (no TX signal generated)")
    print(f"Noise power      : {SIM.noise_power:.6e} W")
    print(f"Realizations     : {N_REALIZATIONS}")
    print(f"PRT threshold    : {PRT_THRESHOLD:.6e}")
    print("=" * 80)

    for R in RANGES_M:
        for theta in ANGLES_DEG:
            time_axis = build_time_axis(R, theta, antenna_x)
            tau_hint = R / C + CLOCK_OFFSET

            zero_rx = np.zeros(
                (len(antenna_x), len(time_axis)),
                dtype=np.complex128
            )

            prt_peaks = []
            hrt_peaks = []

            for trial in range(N_REALIZATIONS):
                noise_rx, _ = add_awgn(zero_rx, rng=rng)

                result = localize(
                    noise_rx, time_axis, antenna_x,
                    theta, tau_hint
                )

                prt_peak = float(result.prt_peak)
                hrt_peak = float(result.hrt_peak)

                prt_peaks.append(prt_peak)
                hrt_peaks.append(hrt_peak)

                rows.append({
                    "R_hint_m": R,
                    "theta_hint_deg": theta,
                    "trial": trial + 1,
                    "PRT_peak": prt_peak,
                    "HRT_peak": hrt_peak,
                    "PRT_above_threshold": int(prt_peak > PRT_THRESHOLD),
                    "PRT_R_m": result.prt_R,
                    "PRT_theta_deg": result.prt_theta,
                    "HRT_R_m": result.R,
                    "HRT_theta_deg": result.theta,
                })

            p = summarize(prt_peaks)
            h = summarize(hrt_peaks)

            summary.append({
                "R_hint_m": R,
                "theta_hint_deg": theta,
                "PRT_mean": p["mean"],
                "PRT_std": p["std"],
                "PRT_p95": p["p95"],
                "PRT_p99": p["p99"],
                "PRT_max": p["max"],
                "HRT_mean": h["mean"],
                "HRT_std": h["std"],
                "HRT_p95": h["p95"],
                "HRT_p99": h["p99"],
                "HRT_max": h["max"],
                "PRT_false_alarm_rate": float(
                    np.mean(np.array(prt_peaks) > PRT_THRESHOLD)
                ),
            })

            print(
                f"R={R:5.0f} m, theta={theta:5.0f} deg | "
                f"PRT mean={p['mean']:.6e}, "
                f"p99={p['p99']:.6e} | "
                f"HRT mean={h['mean']:.6e}, "
                f"p99={h['p99']:.6e}"
            )

    save_csv(OUTPUT_DIR / "noise_peaks_raw.csv", rows)
    save_csv(OUTPUT_DIR / "noise_peaks_summary.csv", summary)

    # ========================================================
    # Global statistics
    # ========================================================

    all_prt = np.array([r["PRT_peak"] for r in rows])
    all_hrt = np.array([r["HRT_peak"] for r in rows])

    print("\n" + "=" * 80)
    print("GLOBAL NOISE PEAK STATISTICS")
    print("=" * 80)

    for name, values in [("PRT", all_prt), ("HRT", all_hrt)]:
        s = summarize(values)
        print(f"\n{name}:")
        print(f"  Mean : {s['mean']:.6e}")
        print(f"  Std  : {s['std']:.6e}")
        print(f"  P95  : {s['p95']:.6e}")
        print(f"  P99  : {s['p99']:.6e}")
        print(f"  Max  : {s['max']:.6e}")

    false_alarm = float(np.mean(all_prt > PRT_THRESHOLD))
    print(f"\nPRT false alarm rate @ {PRT_THRESHOLD:.2e}: {false_alarm:.4%}")

    print("\nEmpirical thresholds from pooled noise-only trials:")
    for target_pfa in [0.05, 0.01, 0.001]:
        q = 100 * (1 - target_pfa)
        print(
            f"Target Pfa={target_pfa:.3%}: "
            f"PRT={np.percentile(all_prt, q):.6e}, "
            f"HRT={np.percentile(all_hrt, q):.6e}"
        )

    # ========================================================
    # Plots
    # ========================================================

    for prefix in ["PRT", "HRT"]:
        plt.figure(figsize=(8, 5))
        for theta in ANGLES_DEG:
            subset = [
                s for s in summary
                if s["theta_hint_deg"] == theta
            ]
            plt.plot(
                [s["R_hint_m"] for s in subset],
                [s[f"{prefix}_mean"] for s in subset],
                "o-", label=f"theta={theta} deg"
            )

        plt.xlabel("Reference range (m)")
        plt.ylabel(f"{prefix} noise peak")
        plt.title(f"Pure noise {prefix} peak versus reference range")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"noise_{prefix.lower()}_vs_range.png",
            dpi=180
        )
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(all_prt, bins=40, alpha=0.6, label="PRT")
    plt.hist(all_hrt, bins=40, alpha=0.6, label="HRT")
    plt.axvline(
        PRT_THRESHOLD, color="red", linestyle="--",
        label=f"PRT threshold = {PRT_THRESHOLD:.1e}"
    )
    plt.xlabel("Peak amplitude")
    plt.ylabel("Count")
    plt.title("Pure noise peak distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_peak_histogram.png", dpi=180)
    plt.close()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
#1e-6