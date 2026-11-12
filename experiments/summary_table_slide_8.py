"""Compare the three single-source experiments on the same trial positions.

Place this file in experiments/ beside the three result folders, then run:
    python compare_phase_hrt_metrics.py

The table uses 2-D localization error > FAILURE_THRESHOLD_M as a failure.
Runtime uses recorded estimator stages, excluding signal generation when possible.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
HRT_DIR = HERE / "hrt_time_domain_results_gaussian_no_mf_noisy_20261021_204834"
PHASE_DIR = HERE / "phase_matching_results_one_value_raw_noisy_gaussian__20261022_002637"
PRT_DIR = HERE / "prt_roi_hrt_no_mf_global_noisy_20261022_105234"
FAILURE_THRESHOLD_M = 20.0
OUTPUT = HERE / "phase_vs_hrt_prt_metrics.json"
CDF_OUTPUT = HERE / "range_error_cdf_comparison.png"


def read_run(folder):
    with (folder / "trials.csv").open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    summary = json.loads((folder / "summary.json").read_text())
    if not trials:
        raise ValueError(f"No trials in {folder / 'trials.csv'}")
    if len(trials) != summary["n_mc"]:
        raise ValueError(f"Trial count differs from summary in {folder}")
    return trials, summary


def column(rows, name):
    return np.array([float(row[name]) for row in rows], dtype=float)


def recorded_runtime(rows, summary, method):
    """Return median timing and an exact description of the recorded fields."""
    fields = set(rows[0])
    if method == "prt" and "pipeline_seconds" in fields:
        return float(np.median(column(rows, "pipeline_seconds"))), "pipeline_seconds"
    for key in ("estimator_seconds", "runtime_estimator_s", "estimator_runtime_s"):
        if key in fields:
            return float(np.median(column(rows, key))), key
    if method == "hrt" and {"hrt_seconds", "refinement_seconds"} <= fields:
        values = column(rows, "hrt_seconds") + column(rows, "refinement_seconds")
        return float(np.median(values)), "hrt_seconds + refinement_seconds"
    if method == "prt":
        # Do not add coarse/fine HRT fields to hrt_seconds: that would double count.
        prt_field = next((key for key in ("prt_seconds", "runtime_prt_s",
                                         "prt_search_seconds")
                          if key in fields), None)
        hrt_field = next((key for key in ("hrt_seconds", "runtime_hrt_s",
                                         "fine_hrt_seconds")
                          if key in fields), None)
        if prt_field and hrt_field:
            used = [prt_field, hrt_field]
            for alternatives in (("roi_seconds", "runtime_roi_s"),
                                 ("refinement_seconds", "runtime_refinement_s")):
                extra = next((key for key in alternatives if key in fields), None)
                if extra is not None:
                    used.append(extra)
            values = sum((column(rows, key) for key in used),
                         np.zeros(len(rows), dtype=float))
            return float(np.median(values)), " + ".join(used)
    for key in ("runtime_median_estimator_s", "median_estimator_seconds",
                "median_runtime_estimator_s"):
        if key in summary:
            return float(summary[key]), f"summary.json:{key}"
    return None, "Timing unavailable: inspect the trial CSV's stage columns"


def metrics(rows, summary, method):
    range_error = column(rows, "range_error_m")
    spatial_error = column(rows, "localization_error_m")
    runtime, runtime_definition = recorded_runtime(rows, summary, method)
    coverage_field = next((key for key in ("roi_contains_truth", "source_in_roi")
                           if key in rows[0]), None) if method != "phase" else None
    coverage = (sum(row[coverage_field].strip().lower() == "true" for row in rows)
                if coverage_field is not None else None)
    elapsed_field = next((key for key in ("elapsed_s", "elapsed_seconds")
                          if key in summary), None)

    n = len(rows)
    return {
        "n_trials": n,
        "range_rmse_m": float(np.sqrt(np.mean(range_error**2))),
        "localization_rmse_m": float(np.sqrt(np.mean(spatial_error**2))),
        "median_abs_range_error_m": float(np.median(np.abs(range_error))),
        "p95_abs_range_error_m": float(np.percentile(np.abs(range_error), 95)),
        "failure_threshold_2d_m": FAILURE_THRESHOLD_M,
        "failure_count": int(np.count_nonzero(spatial_error > FAILURE_THRESHOLD_M)),
        "failure_rate": float(np.mean(spatial_error > FAILURE_THRESHOLD_M)),
        "over_5m_count": int(np.count_nonzero(spatial_error > 5.0)),
        "over_5m_rate": float(np.mean(spatial_error > 5.0)),
        "roi_covered_count": coverage,
        "roi_coverage_rate": (coverage / n if coverage is not None else
                              summary.get("roi_coverage_rate", summary.get("roi_coverage"))
                              if method != "phase" else None),
        "median_recorded_estimator_s": runtime,
        "runtime_fields": runtime_definition,
        "mean_run_wall_s_per_trial": (float(summary[elapsed_field] / n)
                                      if elapsed_field is not None else None),
        "boundary_count": sum(
            row["range_at_boundary"].strip().lower() == "true"
            or row["angle_at_boundary"].strip().lower() == "true"
            for row in rows
        ) if {"range_at_boundary", "angle_at_boundary"} <= rows[0].keys() else None,
    }


def plot_range_error_cdf(phase_rows, hrt_rows, prt_rows):
    """Plot all three empirical CDFs of absolute range error on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for label, rows, color in (
        ("Phase matching", phase_rows, "#d55e00"),
        ("Coarse HRT → Regions -> fine HRT", hrt_rows, "#0072b2"),
        ("PRT → ROI → fine HRT", prt_rows, "#009e73"),
    ):
        errors = np.sort(np.abs(column(rows, "range_error_m")))
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.step(np.r_[0.0, errors], np.r_[0.0, cdf], where="post",
                linewidth=2, color=color, label=label)
    ax.set_xlabel("Absolute range error (m)", fontsize=20)
    ax.set_ylabel("Empirical CDF", fontsize=20)
    ax.set(xlim=(0, None), ylim=(0, 1))
    ax.set_title(f"Range error across {len(phase_rows)} matched trials", fontsize=25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=15)
    fig.savefig(CDF_OUTPUT, dpi=220)
    plt.close(fig)


def main():
    hrt_rows, hrt_summary = read_run(HRT_DIR)
    phase_rows, phase_summary = read_run(PHASE_DIR)
    prt_rows, prt_summary = read_run(PRT_DIR)
    if len({len(hrt_rows), len(phase_rows), len(prt_rows)}) != 1:
        raise ValueError("The runs have different trial counts")

    summaries = (hrt_summary, phase_summary, prt_summary)
    for key in ("seed", "f0_hz", "bandwidth_hz", "n_antennas",
                "reference_peak_snr_db", "reference_range_m", "waveform",
                "matched_filter"):
        values = [s[key] for s in summaries if key in s]
        if len(values) > 1 and any(value != values[0] for value in values[1:]):
            raise ValueError(f"Runs differ in {key}: {values}")
    for other_name, other_rows in (("phase", phase_rows), ("PRT", prt_rows)):
        for hrt, other in zip(hrt_rows, other_rows):
            if hrt["trial"] != other["trial"] or any(
                abs(float(hrt[key]) - float(other[key])) > 1e-9
                for key in ("range_true_m", "theta_true_deg")
            ):
                raise ValueError(f"Trial IDs or source positions differ: HRT vs {other_name}")

    results = {
        "failure_definition": f"2-D localization error > {FAILURE_THRESHOLD_M:g} m",
        "runtime_definition": "Median recorded estimator stages; see runtime_fields per method",
        "phase_matching": metrics(phase_rows, phase_summary, "phase"),
        "coarse_hrt_five_regions_fine_hrt": metrics(hrt_rows, hrt_summary, "hrt"),
        "prt_roi_fine_hrt": metrics(prt_rows, prt_summary, "prt"),
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    plot_range_error_cdf(phase_rows, hrt_rows, prt_rows)

    print(f"Same {len(hrt_rows)} source positions in all three runs; "
          f"failure: {results['failure_definition']}\n")
    print("| Method | Range RMSE | Median | P95 | Failure | ROI coverage | "
          "Median estimator time |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, key in (("Phase matching", "phase_matching"),
                       ("Coarse HRT → 5 regions → fine HRT",
                        "coarse_hrt_five_regions_fine_hrt"),
                       ("PRT → ROI → fine HRT", "prt_roi_fine_hrt")):
        m = results[key]
        coverage = ("N/A" if m["roi_coverage_rate"] is None else
                    (f"{m['roi_covered_count']}/{m['n_trials']} "
                     if m["roi_covered_count"] is not None else "") +
                    f"({100*m['roi_coverage_rate']:.1f}%)")
        timing = ("N/A" if m["median_recorded_estimator_s"] is None else
                  f"{m['median_recorded_estimator_s']:.3f} s")
        print(f"| {label} | {m['range_rmse_m']:.2f} m | "
              f"{m['median_abs_range_error_m']:.2f} m | "
              f"{m['p95_abs_range_error_m']:.2f} m | "
              f"{m['failure_count']}/{m['n_trials']} "
              f"({100*m['failure_rate']:.2f}%) | {coverage} | {timing} |")
    print("\nAdditional: error >5 m (2-D); mean run wall time per trial")
    for key in ("phase_matching", "coarse_hrt_five_regions_fine_hrt",
                "prt_roi_fine_hrt"):
        m = results[key]
        wall = ("N/A" if m["mean_run_wall_s_per_trial"] is None else
                f"{m['mean_run_wall_s_per_trial']:.3f} s/trial")
        print(f"  {key}: {m['over_5m_count']}/{m['n_trials']} "
              f"({100*m['over_5m_rate']:.1f}%); {wall}; "
              f"timing = {m['runtime_fields']}")
    print(f"Saved {OUTPUT} and {CDF_OUTPUT}")


if __name__ == "__main__":
    main()
