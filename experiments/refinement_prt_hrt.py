"""Blind single-source, no-MF PRT -> ROI -> HRT Monte Carlo experiment.

Place alongside SIC_2.py in experiments/ and run without command-line options:
    python experiments/prt_roi_hrt_single_mc.py

Uses SIC_2's PRT/ROI geometry and an FFT HRT, with a uniform global p grid
selected from the minimum-p-resolution calculation and a minimum-node uniform
q grid covering SIC_2's original q search interval.
PRT searches all arrival times at 4x Fourier interpolation; HRT searches
only the data-derived ROI at the same temporal resolution.
Neither transform is matched filtered. No true-angle/time hints, template fit,
or SIC follow the HRT peak.

As in single_user.py, the simulator records an arrival-centered finite window;
the estimators search that entire window without an oracle time index. The
acquisition window is shared across methods in any fair comparison.

The global PRT uses the repo's batched CuPy peak kernel because materializing
the full (time, p, q) cube is impractical. Requires a CUDA-enabled CuPy setup.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from localization.pipeline import (
    PRT_R_MIN, PRT_R_MAX, PRT_THETA_MIN, PRT_THETA_MAX,
    Q_GRID, fixed_grid, polar_to_xy,
)
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import (
    build_roi, K_Q, THETA_HALF_WIDTH_DEG, TAU_HALF_WIDTH_SAMPLES,
)
from signal_model.propagation import propagation_delays, received_signal


N_MC = 400 #400
R_MIN, R_MAX = 50.0, 400.0
THETA_MIN, THETA_MAX = -60.0, 60.0
CLOCK_OFFSET = 0 #10e-9
N_P = 497  # np_range_requirement_smoke/np_range_requirements.json, 24 GHz config
Q_GRID_MODE = "minimum"  # "original" restores SIC_2's 413-node uniform grid.
# Np=497 row of results/np_range_requirement_smoke/np_range_sweep.csv.
# These two terms must be recomputed if the array/waveform/source domain changes.
Q_CONT_ERROR_BOUND = 2.9954322135191524e-13  # s/m^2
Q_MODEL_ERROR_BOUND = 7.33454749925482e-15  # s/m^2
PRT_BATCH_SIZE = 256
PRT_TIME_OVERSAMPLE = 4
HRT_RANGE_STEP_M = 1.0
HRT_THETA_STEP_DEG = 0.1
HRT_TIME_OVERSAMPLE = 4
HRT_TILE_SIZE = 128
NEWTON_MAXITER = 25
NOISELESS = False
SHOW_PLOTS = True

OUTPUT_DIR = HERE / (
    f"prt_roi_hrt_no_mf_global_{'noiseless' if NOISELESS else 'noisy'}_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


def gpu_sync():
    """Include queued GPU operations in elapsed time if the repo uses CuPy."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except ImportError:
        pass


def contains(value, lo, hi):
    return bool(lo <= value <= hi)


def global_p_grid():
    # Np=497 is the minimum-p study's recommendation for this exact system.
    # If config changes, rerun that study and update N_P before comparing.
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    expected = (np.isclose(SYSTEM.fc, 24e9)
                and np.isclose(SYSTEM.bandwidth, 8e9)
                and np.isclose(np.ptp(x), 3.197786218666667)
                and np.isclose(SYSTEM.dt, 1.5625e-11, rtol=1e-8, atol=0))
    if not expected:
        raise ValueError("N_P=497 was designed for f0=24 GHz, B=8 GHz, "
                         "L=3.197786 m and dt=15.625 ps. Rerun your "
                         "minimum-p requirement study for this config.")
    p_end = np.sin(np.deg2rad(max(abs(PRT_THETA_MIN), abs(PRT_THETA_MAX)))) / C
    return np.linspace(-p_end, p_end, N_P)


def q_resolution_budget():
    # Existing minimum-p study: E_q_cont + dq/2 + E_q_model <= q_min/2.
    q_min = (np.cos(np.deg2rad(max(abs(THETA_MIN), abs(THETA_MAX)))) ** 2
             / (2 * C * R_MAX))
    target = q_min / 2
    max_dq = 2 * (target - Q_CONT_ERROR_BOUND - Q_MODEL_ERROR_BOUND)
    if max_dq <= 0:
        raise ValueError("The p/model errors already exceed the q error target")
    return float(target), float(max_dq)


def q_grid_and_valid_mask(p_grid):
    if Q_GRID_MODE == "original":
        q_grid = np.asarray(Q_GRID, dtype=float)
    elif Q_GRID_MODE == "minimum":
        _, max_dq = q_resolution_budget()
        q_first, q_last = float(Q_GRID[0]), float(Q_GRID[-1])
        n_q = int(np.ceil((q_last - q_first) / max_dq)) + 1
        q_grid = np.linspace(q_first, q_last, n_q)
    else:
        raise ValueError("Q_GRID_MODE must be 'minimum' or 'original'")
    r_grid = pq_to_range(p_grid[:, None], q_grid[None, :])
    valid = (np.isfinite(r_grid) & (r_grid >= PRT_R_MIN)
             & (r_grid <= PRT_R_MAX))
    return q_grid, valid


def local_q_step(q_grid, index):
    # The ROI must use the actual spacing of whichever uniform grid was selected.
    left = q_grid[index] - q_grid[max(index - 1, 0)]
    right = q_grid[min(index + 1, len(q_grid) - 1)] - q_grid[index]
    return float(max(left, right))


def global_prt_peak(noisy, time_axis, antenna_x, p_grid, q_grid, valid):
    try:
        from transforms.fast_prt_peak import run_parabolic_radon_peak
    except ImportError as exc:
        raise ImportError("The global PRT needs CUDA-enabled CuPy and "
                          "transforms.fast_prt_peak") from exc
    result = run_parabolic_radon_peak(
        noisy, antenna_x, SYSTEM.dt, float(SYSTEM.fc), p_grid, q_grid,
        cpo=True, candidate_batch=PRT_BATCH_SIZE,
        valid_spatial_mask=valid, time_factor=PRT_TIME_OVERSAMPLE,
    )
    it, ip, iq, peak_power = result
    p, q = float(p_grid[ip]), float(q_grid[iq])
    return (float(pq_to_range(p, q)), float(p_to_theta_deg(p)),
            float(time_axis[0] + it * SYSTEM.dt / PRT_TIME_OVERSAMPLE),
            float(np.sqrt(peak_power)), q,
            local_q_step(q_grid, iq))


def roi_hrt_peak(noisy, t, x, roi):
    """Exact spherical delays, 4x FFT time interpolation, equal antenna weights."""
    from transforms.fast_hrt_peak import run_hyperbolic_radon_peak
    r_grid = fixed_grid(roi.R_min, roi.R_max, HRT_RANGE_STEP_M)
    theta_grid = fixed_grid(roi.theta_min, roi.theta_max,
                            HRT_THETA_STEP_DEG)
    return run_hyperbolic_radon_peak(
        noisy, t, x, r_grid, theta_grid,
        (roi.tau_min, roi.tau_max), float(SYSTEM.fc),
        time_factor=HRT_TIME_OVERSAMPLE, tile_size=HRT_TILE_SIZE,
    )


def bounded_damped_newton(evaluate, bounds, hessian_steps, maxiter=25):
    """Trust-region Newton maximizer from hrt_newton_only.py."""
    lower, upper = np.asarray(bounds, float).T
    u = np.zeros(3)
    initial_score = evaluate(u)[0]
    scale = max(initial_score, 1e-30)
    radius = 1.0
    evaluations = 1
    accepted_steps = 0
    reason = "maximum iterations"

    for _ in range(maxiter):
        score, raw_gradient = evaluate(u)
        evaluations += 1
        gradient = raw_gradient / scale
        hessian = np.empty((3, 3))
        for axis, h in enumerate(hessian_steps):
            shift = np.zeros(3)
            shift[axis] = h
            plus = evaluate(u + shift)[1]
            minus = evaluate(u - shift)[1]
            evaluations += 2
            hessian[:, axis] = (plus - minus) / (2 * h * scale)
        hessian = (hessian + hessian.T) / 2

        negative_hessian = -hessian
        metric = np.diag(np.maximum(np.abs(np.diag(negative_hessian)), 1e-8))
        damping = 0.0
        for _ in range(12):
            try:
                np.linalg.cholesky(negative_hessian + damping * metric)
                break
            except np.linalg.LinAlgError:
                damping = 1e-3 if damping == 0 else 10 * damping
        else:
            reason = "no positive definite damped Hessian"
            break

        direction = np.linalg.solve(negative_hessian + damping * metric, gradient)
        direction *= min(1.0, radius / max(np.max(np.abs(direction)), 1e-100))
        candidate = np.clip(u + direction, lower, upper)
        displacement = candidate - u
        if np.max(np.abs(displacement)) < 1e-8:
            reason = "Newton step below scaled tolerance"
            break

        candidate_score, candidate_gradient = evaluate(candidate)
        evaluations += 1
        actual_gain = (candidate_score - score) / scale
        predicted_gain = (gradient @ displacement
                          + 0.5 * displacement @ hessian @ displacement)
        ratio = actual_gain / predicted_gain if predicted_gain > 0 else -np.inf
        roundoff = (abs(actual_gain) < 10 * np.finfo(float).eps and
                    np.linalg.norm(candidate_gradient / scale) <
                    np.linalg.norm(gradient))
        if (predicted_gain > 0 and ratio > 0.1) or roundoff:
            u = candidate
            accepted_steps += 1
            if ratio > 0.75 and np.isclose(
                np.max(np.abs(direction)), radius, rtol=0, atol=1e-8
            ):
                radius = min(16.0, 2 * radius)
            if np.max(np.abs(displacement)) < 1e-6:
                reason = "small accepted Newton step"
                break
        else:
            radius *= 0.5
            if radius < 1e-8:
                reason = "trust radius reached numerical tolerance"
                break

    final_score = evaluate(u)[0]
    evaluations += 1
    return u, final_score, evaluations, accepted_steps, reason


def refine_hrt_peak(noisy, t, x, roi, coarse):
    """Refine (R, theta, tau) by continuously aligning the *observed* traces.

    Uses the bounded damped Newton method from hrt_newton_only.py. Antenna
    weights match this pipeline's HRT (uniform). The HRT grid evaluates
    FFT-interpolated times, followed by the same continuous score here.
    """
    r0, theta0, tau0 = map(float, coarse)
    dr, dtheta = float(HRT_RANGE_STEP_M), float(HRT_THETA_STEP_DEG)
    dt = float(SYSTEM.dt)
    guard = int(np.ceil(np.max(np.abs(x)) / (C * dt))) + 4
    nfft = next_fast_len(len(t) + 2 * guard)
    spectrum = np.fft.fft(
        np.pad(noisy, ((0, 0), (guard, nfft - len(t) - guard))), axis=1
    )
    freq = np.fft.fftfreq(nfft, d=dt)
    total_freq = freq[None, :] + float(SYSTEM.fc)

    def score_and_gradient(u):
        r = r0 + u[0] * dr
        theta = np.deg2rad(theta0 + u[1] * dtheta)
        tau = tau0 + u[2] * dt
        quadratic = x * x - 2 * r * x * np.sin(theta)
        dist = np.sqrt(r * r + quadratic)
        delta_length = quadratic / (dist + r)
        ddelta_dr = -(x * x + x * np.sin(theta) * delta_length) / (
            C * dist * (dist + r)
        )
        ddelta_dtheta = (-r * x * np.cos(theta) / dist) / C
        delta = delta_length / C

        aligned = spectrum * np.exp(2j * np.pi * total_freq * delta[:, None])
        u_freq = np.mean(aligned, axis=0)
        du_dr = np.mean(aligned * (2j * np.pi * total_freq)
                        * ddelta_dr[:, None], axis=0) * dr
        du_dtheta = np.mean(aligned * (2j * np.pi * total_freq)
                            * ddelta_dtheta[:, None], axis=0) * np.deg2rad(dtheta)

        time_phase = np.exp(2j * np.pi * freq * (tau - t[0] + guard * dt))
        value = np.dot(u_freq, time_phase) / nfft
        derivative = np.array([
            np.dot(du_dr, time_phase),
            np.dot(du_dtheta, time_phase),
            np.dot(u_freq, 2j * np.pi * freq * dt * time_phase),
        ]) / nfft
        return float(abs(value) ** 2), 2 * np.real(np.conj(value) * derivative)

    initial_score = score_and_gradient(np.zeros(3))[0]
    bounds = [
        ((roi.R_min - r0) / dr, (roi.R_max - r0) / dr),
        (max(-1.0, (roi.theta_min - theta0) / dtheta),
         min(1.0, (roi.theta_max - theta0) / dtheta)),
        (max(-1.0, (roi.tau_min - tau0) / dt),
         min(1.0, (roi.tau_max - tau0) / dt)),
    ]

    best, best_score, evaluations, steps, reason = bounded_damped_newton(
        score_and_gradient, bounds,
        hessian_steps=np.array([0.01 / dr, 1e-3, 1e-3]),
        maxiter=NEWTON_MAXITER,
    )
    accepted = bool(steps > 0 and best_score > initial_score *
                    (1 - 5 * np.finfo(float).eps))
    if not accepted:
        best = np.zeros(3)
    return (r0 + best[0] * dr, theta0 + best[1] * dtheta,
            tau0 + best[2] * dt, accepted,
            reason in ("Newton step below scaled tolerance",
                       "small accepted Newton step"),
            int(evaluations), int(steps), reason)


def main():
    if N_MC < 1:
        raise ValueError("N_MC must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.asarray(ARRAY.positions(SYSTEM), dtype=float)
    p_grid = global_p_grid()
    q_grid, valid = q_grid_and_valid_mask(p_grid)
    dp = float(p_grid[1] - p_grid[0])
    q_target, q_max_step = q_resolution_budget()
    dq = float(q_grid[1] - q_grid[0])
    seed = int(SIM.rng_seed)
    rng_pos = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1)

    # Fixed hardware noise: peak SNR at the center antenna and reference range.
    # SIC_2.add_awgn instead recalculates variance for each received signal.
    reference_range = float(SIM.reference_range)
    reference_snr_db = float(SIM.reference_snr_db)
    center_index = int(np.argmin(np.abs(x)))
    reference_arrival = reference_range / C + CLOCK_OFFSET
    reference_clean, _ = received_signal(
        np.array([reference_arrival]), reference_range, 0.0,
        clock_offset=CLOCK_OFFSET, antenna_x=x,
    )
    reference_peak_power = float(abs(reference_clean[center_index, 0]) ** 2)
    variance = (0.0 if NOISELESS else
                reference_peak_power / 10.0 ** (reference_snr_db / 10.0))
    noise_scale = np.sqrt(variance / 2.0)

    rows = []
    trials_path = OUTPUT_DIR / "trials.csv"
    started = time.perf_counter()
    print(f"Global PRT: Np={len(p_grid)}, Nq={len(q_grid)} "
          f"({Q_GRID_MODE}, dq={dq:.4g} s/m²), "
          f"valid spatial cells={np.count_nonzero(valid)}; "
          f"matched filter: off; trials: {N_MC}", flush=True)
    print(f"Fixed noise variance: {variance:.6g}; reference peak SNR: "
          f"{reference_snr_db:g} dB at {reference_range:g} m", flush=True)

    for i in range(N_MC):
        r_true = float(rng_pos.uniform(R_MIN, R_MAX))
        theta_true = float(rng_pos.uniform(THETA_MIN, THETA_MAX))
        tau_true = r_true / C + CLOCK_OFFSET
        xt, yt = polar_to_xy(r_true, theta_true)

        delays = propagation_delays(
            r_true, theta_true, clock_offset=CLOCK_OFFSET, antenna_x=x,
        )
        margin = 8.0 * SYSTEM.gaussian_sigma
        n0 = int(np.floor((delays.min() - margin) / SYSTEM.dt))
        n1 = int(np.ceil((delays.max() + margin) / SYSTEM.dt))
        t = np.arange(n0, n1 + 1) * SYSTEM.dt
        clean, _ = received_signal(
            t, r_true, theta_true, clock_offset=CLOCK_OFFSET, antenna_x=x,
        )
        noisy = clean + noise_scale * (
            rng_noise.standard_normal(clean.shape)
            + 1j * rng_noise.standard_normal(clean.shape)
        ) if not NOISELESS else clean.copy()

        gpu_sync()
        pipeline_started = time.perf_counter()
        # Search every p, every valid q, and the 4x-interpolated time trace.
        r_prt, theta_prt, tau_prt, prt_peak, q_prt, dq_local = global_prt_peak(
            noisy, t, x, p_grid, q_grid, valid
        )
        gpu_sync()
        prt_seconds = time.perf_counter() - pipeline_started

        roi_started = time.perf_counter()
        roi = build_roi(
            r_prt, theta_prt, tau_prt, dq=dq_local,
            range_bounds=(PRT_R_MIN, PRT_R_MAX),
            theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX),
        )
        range_in_roi = contains(r_true, roi.R_min, roi.R_max)
        theta_in_roi = contains(theta_true, roi.theta_min, roi.theta_max)
        tau_in_roi = contains(tau_true, roi.tau_min, roi.tau_max)
        roi_seconds = time.perf_counter() - roi_started

        gpu_sync()
        hrt_started = time.perf_counter()
        r_hat, theta_hat, tau_hat, hrt_peak = roi_hrt_peak(noisy, t, x, roi)
        gpu_sync()
        hrt_seconds = time.perf_counter() - hrt_started
        hrt_grid_r, hrt_grid_theta, hrt_grid_tau = r_hat, theta_hat, tau_hat

        refinement_started = time.perf_counter()
        (r_hat, theta_hat, tau_hat, refinement_accepted,
         refinement_converged, refinement_evaluations,
         newton_steps, newton_reason) = refine_hrt_peak(
            noisy, t, x, roi, (r_hat, theta_hat, tau_hat)
        )
        refinement_seconds = time.perf_counter() - refinement_started
        total_seconds = time.perf_counter() - pipeline_started

        xh, yh = polar_to_xy(r_hat, theta_hat)
        range_error = float(r_hat - r_true)
        localization_error = float(np.hypot(xh - xt, yh - yt))
        rows.append(dict(
            trial=i, range_true_m=r_true, theta_true_deg=theta_true,
            tau_true_s=tau_true, x_true_m=xt, y_true_m=yt,
            time_samples=len(t), noise_variance=variance,
            sampled_peak_snr_db=(None if NOISELESS else float(
                10 * np.log10(np.max(abs(clean[center_index]) ** 2) / variance)
            )),
            prt_range_m=r_prt, prt_theta_deg=theta_prt,
            prt_tau_s=tau_prt, prt_peak=prt_peak,
            prt_q_s_per_m2=q_prt, prt_local_dq_s_per_m2=dq_local,
            roi_range_min_m=roi.R_min, roi_range_max_m=roi.R_max,
            roi_theta_min_deg=roi.theta_min, roi_theta_max_deg=roi.theta_max,
            roi_tau_min_s=roi.tau_min, roi_tau_max_s=roi.tau_max,
            range_in_roi=range_in_roi, theta_in_roi=theta_in_roi,
            tau_in_roi=tau_in_roi,
            source_in_roi=range_in_roi and theta_in_roi and tau_in_roi,
            hrt_grid_range_m=float(hrt_grid_r),
            hrt_grid_theta_deg=float(hrt_grid_theta),
            hrt_grid_tau_s=float(hrt_grid_tau),
            range_est_m=float(r_hat), theta_est_deg=float(theta_hat),
            tau_est_s=float(tau_hat), hrt_peak=float(hrt_peak),
            refinement_accepted=refinement_accepted,
            refinement_converged=refinement_converged,
            refinement_evaluations=refinement_evaluations,
            newton_steps=newton_steps, newton_reason=newton_reason,
            range_error_m=range_error, abs_range_error_m=abs(range_error),
            localization_error_m=localization_error,
            prt_seconds=prt_seconds, roi_seconds=roi_seconds,
            hrt_seconds=hrt_seconds, refinement_seconds=refinement_seconds,
            pipeline_seconds=total_seconds,
        ))
        # Keep completed trials if a long Monte Carlo run is interrupted.
        with trials_path.open("w" if i == 0 else "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[-1].keys())
            if i == 0:
                writer.writeheader()
            writer.writerow(rows[-1])
        if (i + 1) % 20 == 0 or i == 0:
            print(f"{i + 1}/{N_MC}; elapsed {time.perf_counter() - started:.1f} s; "
                    f"PRT {prt_seconds:.2f} s; HRT {hrt_seconds:.2f} s; "
                    f"refinement {refinement_seconds:.2f} s; "
                    f"ROI contains truth: {rows[-1]['source_in_roi']}", flush=True)

    range_errors = np.array([row["abs_range_error_m"] for row in rows])
    spatial_errors = np.array([row["localization_error_m"] for row in rows])
    summary = dict(
        n_mc=N_MC, seed=seed, config_path=str(HERE.parent / "config.py"),
        assisted_prt=False, assisted_hrt=False, matched_filter=False,
        template_refinement=False, template_free_hrt_refinement=True,
        refinement_method="bounded damped Newton",
        f0_hz=float(SYSTEM.fc),
        p_grid_nodes=N_P, p_grid_step_s_per_m=dp,
        q_grid_nodes=len(q_grid), q_grid_spacing="uniform",
        q_grid_mode=Q_GRID_MODE, q_grid_step_s_per_m2=dq,
        q_grid_max_step_from_study_s_per_m2=q_max_step,
        q_error_target_from_study_s_per_m2=q_target,
        q_error_bound_at_actual_step_s_per_m2=(Q_CONT_ERROR_BOUND + dq / 2
                                                + Q_MODEL_ERROR_BOUND),
        q_min_s_per_m2=float(q_grid[0]), q_max_s_per_m2=float(q_grid[-1]),
        valid_pq_nodes=int(np.count_nonzero(valid)),
        prt_time_oversample=PRT_TIME_OVERSAMPLE,
        hrt_range_step_m=HRT_RANGE_STEP_M,
        hrt_theta_step_deg=HRT_THETA_STEP_DEG,
        hrt_time_oversample=HRT_TIME_OVERSAMPLE,
        bandwidth_hz=float(SYSTEM.bandwidth), n_antennas=len(x),
        noise_variance=variance, reference_range_m=reference_range,
        reference_peak_snr_db=reference_snr_db,
        acquisition="Arrival-centered recording window spans every antenna's pulse plus 8 sigma; PRT searches its full 4x-interpolated time trace",
        snr_definition="Peak power at the center antenna / fixed complex noise variance",
        range_min_m=R_MIN, range_max_m=R_MAX,
        theta_min_deg=THETA_MIN, theta_max_deg=THETA_MAX,
        roi_coverage=float(np.mean([row["source_in_roi"] for row in rows])),
        range_roi_coverage=float(np.mean([row["range_in_roi"] for row in rows])),
        theta_roi_coverage=float(np.mean([row["theta_in_roi"] for row in rows])),
        tau_roi_coverage=float(np.mean([row["tau_in_roi"] for row in rows])),
        ale_m=float(np.mean(spatial_errors)),
        localization_rmse_m=float(np.sqrt(np.mean(spatial_errors ** 2))),
        range_bias_m=float(np.mean([row["range_error_m"] for row in rows])),
        range_rmse_m=float(np.sqrt(np.mean(range_errors ** 2))),
        median_abs_range_error_m=float(np.median(range_errors)),
        p95_abs_range_error_m=float(np.quantile(range_errors, .95)),
        mean_prt_seconds=float(np.mean([row["prt_seconds"] for row in rows])),
        mean_hrt_seconds=float(np.mean([row["hrt_seconds"] for row in rows])),
        mean_refinement_seconds=float(np.mean([row["refinement_seconds"] for row in rows])),
        mean_pipeline_seconds=float(np.mean([row["pipeline_seconds"] for row in rows])),
        median_pipeline_seconds=float(np.median([row["pipeline_seconds"] for row in rows])),
        elapsed_seconds=time.perf_counter() - started,
        roi_k_q=float(K_Q), roi_theta_half_width_deg=float(THETA_HALF_WIDTH_DEG),
        roi_tau_half_width_samples=float(TAU_HALF_WIDTH_SAMPLES),
        note="Global PRT p/q/time search with uniform q; FFT HRT restricted to PRT-derived ROI; no truth hints.",
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ordered = np.sort(range_errors)
    ax.step(np.r_[0, ordered], np.r_[0, np.arange(1, N_MC + 1) / N_MC],
            where="post", linewidth=2)
    ax.set(xlabel="Absolute Range Error (m)", ylabel="Empirical CDF",
           xlim=(0, None), ylim=(0, 1),
           title=f"PRT → ROI → HRT: Range Error ({N_MC} trials)")
    ax.grid(True, alpha=.3)
    fig.savefig(OUTPUT_DIR / "range_error_cdf.png", dpi=180)

    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    sc = ax.scatter([row["x_true_m"] for row in rows],
                    [row["y_true_m"] for row in rows],
                    c=spatial_errors, cmap="viridis", s=30, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Absolute 2-D Localization Error (m)")
    ax.plot(x, np.zeros_like(x), color="black", lw=2, label="Receiving array")
    xmax = R_MAX * np.sin(np.deg2rad(max(abs(THETA_MIN), abs(THETA_MAX))))
    ax.set_xlim(-xmax - 25, xmax + 25)
    ax.set_ylim(0, R_MAX + 25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("True User Positions Colored by Absolute Localization Error")
    ax.grid(True, alpha=.3)
    ax.text(.02, .98,
            f"N={N_MC}\nROI coverage={summary['roi_coverage']:.1%}\n"
            f"ALE={summary['ale_m']:.3f} m\n"
            f"RMSE={summary['localization_rmse_m']:.3f} m",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.9,
                      edgecolor="gray"))
    fig.savefig(OUTPUT_DIR / "true_positions_localization_error.png", dpi=180)
    print(f"Saved trials, summary, and plots to {OUTPUT_DIR}", flush=True)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
