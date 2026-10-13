"""Single-user position sweep for PRT/HRT peak calibration.
Place in experiments/scan_prt_hrt_peaks.py and run from project environment.
Uses the identical signal/noise and truth-assisted localization pipeline as single_user.py.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import numpy as np
import matplotlib.pyplot as plt
from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn
from localization.pipeline import localize, PRT_R_MIN, PRT_R_MAX

# Edit these parameters to explore different positions.
RANGES_M = [50, 100, 150, 200, 300, 400]
ANGLES_DEG = [0, 20, 40, 50, 60]
N_REALIZATIONS = 3          # Independent noise realizations per position
CLOCK_OFFSET = 10e-9
SEED = SIM.rng_seed + 1
OUTPUT_DIR = ROOT / "results" / "peak_position_sweep"


def build_time_axis(R, theta, antenna_x):
    delays = propagation_delays(R, theta, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x)
    margin = 8 * SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
    return np.arange(n0, n1 + 1) * SYSTEM.dt


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    antenna_x = ARRAY.positions(SYSTEM)
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"Tx power={SIM.tx_power:.6g} W; noise power={SIM.noise_power:.6e} W")
    print("PRT/HRT: localize(..., theta_true, tau_true), matching single_user.py")

    for R in RANGES_M:
        for theta in ANGLES_DEG:
            t = build_time_axis(R, theta, antenna_x)
            clean_rx, _ = received_signal(
                t, R, theta, clock_offset=CLOCK_OFFSET, antenna_x=antenna_x
            )
            tau = R / C + CLOCK_OFFSET
            for trial in range(N_REALIZATIONS):
                noisy_rx, _ = add_awgn(clean_rx, rng=rng)
                result = localize(noisy_rx, t, antenna_x, theta, tau)
                x_true = R * np.sin(np.deg2rad(theta))
                y_true = R * np.cos(np.deg2rad(theta))
                error = float(np.hypot(result.x - x_true, result.y - y_true))
                rows.append({
                    "R_true_m": R, "theta_true_deg": theta, "trial": trial + 1,
                    "SNR_ref_db": SIM.reference_snr_db + 20 * np.log10(SIM.reference_range / R),
                    "PRT_peak": result.prt_peak, "HRT_peak": result.hrt_peak,
                    "HRT_over_PRT": result.hrt_peak / result.prt_peak if result.prt_peak > 0 else np.nan,
                    "PRT_R_m": result.prt_R, "PRT_theta_deg": result.prt_theta,
                    "HRT_R_m": result.R, "HRT_theta_deg": result.theta,
                    "PRT_tau_ns": result.prt_tau * 1e9,
                    "HRT_tau_ns": result.tau * 1e9, "error_2d_m": error,
                    "HRT_at_range_bound": int(abs(result.R - PRT_R_MIN) < 1.001 or abs(result.R - PRT_R_MAX) < 1.001),
                })
                print(f"R={R:6.1f} theta={theta:6.1f} trial={trial + 1:2d} "
                      f"PRT={result.prt_peak:.6e} HRT={result.hrt_peak:.6e} "
                      f"Rhat={result.R:.1f} theta_hat={result.theta:.3f} error={error:.3f} m")

    csv_path = OUTPUT_DIR / "peaks_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for R in RANGES_M:
        for theta in ANGLES_DEG:
            subset = [row for row in rows if row["R_true_m"] == R and row["theta_true_deg"] == theta]
            p = np.array([row["PRT_peak"] for row in subset])
            h = np.array([row["HRT_peak"] for row in subset])
            summary.append({"R_m": R, "theta_deg": theta,
                            "PRT_mean": p.mean(), "PRT_std": p.std(ddof=1) if len(p) > 1 else 0,
                            "HRT_mean": h.mean(), "HRT_std": h.std(ddof=1) if len(h) > 1 else 0,
                            "error_mean_m": np.mean([row["error_2d_m"] for row in subset])})
    with (OUTPUT_DIR / "peaks_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for key, name in [("PRT_mean", "prt_peak_vs_range.png"), ("HRT_mean", "hrt_peak_vs_range.png")]:
        plt.figure(figsize=(7.6, 5))
        for theta in ANGLES_DEG:
            subset = [s for s in summary if s["theta_deg"] == theta]
            plt.plot([s["R_m"] for s in subset], [s[key] for s in subset], "o-", label=f"theta={theta} deg")
        plt.xlabel("User range (m)")
        plt.ylabel(key.replace("_", " "))
        plt.title("Single-user " + key.replace("_", " "))
        plt.grid(True, alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / name, dpi=180)
        plt.close()
    print(f"\nSaved CSV and figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
