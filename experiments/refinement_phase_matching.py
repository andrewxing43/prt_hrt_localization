import numpy as np

from scipy.constants import speed_of_light as C

import csv
import json
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, resample
from datetime import datetime

try:
    import cupy as cp
except ImportError:
    cp = None

NOISELESS = False
USE_MATCHED_FILTER = True
WAVEFORM = 'gaussian'       # 'gaussian' or 'qam'
QAM_PW_SEC = 2e-9
QAM_M = 16

# 'full_matrix' or 'one_value'
DATA_MODE = 'one_value'     
USE_CUPY = True
GPU_COMPLEX_DTYPE = 'complex64'

# used only for MF + one_value
PEAK_TIME_OVERSAMPLE = 4      
GRID_CHUNK = 2048

output_dir = Path(__file__).resolve().parent/(
    f'phase_matching_results_{DATA_MODE}_'
    f'{"MF" if USE_MATCHED_FILTER else "raw"}_'
    f'{"noiseless" if NOISELESS else "noisy"}_'
    f'{WAVEFORM}__{datetime.now():%Y%m%d_%H%M%S}'
)
method_name = f'{WAVEFORM.upper()}: {"MF + " if USE_MATCHED_FILTER else ""}' + (
    'Full-Matrix Phase Matching' if DATA_MODE == 'full_matrix'
    else 'One-Value Phase Matching'
)

here = Path(__file__).resolve().parent
root = here if (here/'config.py').is_file() else here.parent
sys.path.insert(0, str(root))

from config import C as config_c, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.qam_source import MultiSourceGen


# same Monte Carlo domain as experiments/single_user.py.
N_MC = 50



R_MIN, R_MAX = 50.0, 400.0
# R_MIN, R_MAX = 1.0, 50.0
THETA_MIN, THETA_MAX = -60.0, 60.0

SEARCH_R_MIN, SEARCH_R_MAX = R_MIN, R_MAX
SEARCH_THETA_MIN, SEARCH_THETA_MAX = THETA_MIN, THETA_MAX

CLOCK_OFFSET = 0 #10e-9
RANGE_STEP_M = 1.0
THETA_STEP_DEG = 0.1
NEWTON_HESSIAN_STEPS = np.array([0.01/RANGE_STEP_M, 1e-3])

def bounded_damped_newton(evaluate, bounds, maxiter=25):
    """Maximize a score with analytic gradient and finite-difference Hessian."""
    lower, upper = np.asarray(bounds, float).T
    u = np.zeros(2)
    initial_score = evaluate(u)[0]
    scale = max(initial_score, 1e-30)
    radius = 1.0
    evaluations = 1
    accepted_steps = 0
    reason = 'maximum iterations'

    for _ in range(maxiter):
        score, raw_gradient = evaluate(u)
        evaluations += 1
        gradient = raw_gradient/scale
        hessian = np.empty((2, 2))
        for axis, h in enumerate(NEWTON_HESSIAN_STEPS):
            shift = np.zeros(2)
            shift[axis] = h
            plus = evaluate(u+shift)[1]
            minus = evaluate(u-shift)[1]
            evaluations += 2
            hessian[:, axis] = (plus-minus)/(2*h*scale)
        hessian = (hessian+hessian.T)/2

        negative_hessian = -hessian
        metric = np.diag(np.maximum(np.abs(np.diag(negative_hessian)), 1e-8))
        damping = 0.0
        for _ in range(12):
            try:
                np.linalg.cholesky(negative_hessian+damping*metric)
                break
            except np.linalg.LinAlgError:
                damping = 1e-3 if damping == 0 else 10*damping
        else:
            reason = 'no positive definite damped Hessian'
            break

        direction = np.linalg.solve(negative_hessian+damping*metric, gradient)
        direction *= min(1.0, radius/max(np.max(np.abs(direction)), 1e-100))
        candidate = np.clip(u+direction, lower, upper)
        displacement = candidate-u
        if np.max(np.abs(displacement)) < 1e-8:
            reason = 'Newton step below scaled tolerance'
            break

        candidate_score, candidate_gradient = evaluate(candidate)
        evaluations += 1
        actual_gain = (candidate_score-score)/scale
        predicted_gain = gradient@displacement + 0.5*displacement@hessian@displacement
        ratio = actual_gain/predicted_gain if predicted_gain > 0 else -np.inf
        roundoff = (abs(actual_gain) < 10*np.finfo(float).eps and
                    np.linalg.norm(candidate_gradient/scale) <
                    np.linalg.norm(gradient))
        if (predicted_gain > 0 and ratio > 0.1) or roundoff:
            u = candidate
            accepted_steps += 1
            if ratio > 0.75 and np.isclose(
                np.max(np.abs(direction)), radius, rtol=0, atol=1e-8
            ):
                radius = min(16.0, 2*radius)
            if np.max(np.abs(displacement)) < 1e-6:
                reason = 'small accepted Newton step'
                break
        else:
            radius *= 0.5
            if radius < 1e-8:
                reason = 'trust radius reached numerical tolerance'
                break

    final_score = evaluate(u)[0]
    evaluations += 1
    return u, final_score, evaluations, accepted_steps, reason

def steering(x, f0, ranges, angles):
    """
    Generate the theoretical phase shifts the array should see for a source across all ranges/angles.
    """
    x = np.asarray(x, float).reshape(-1, 1)
    r,  th = np.broadcast_arrays(np.asarray(ranges, float), np.asarray(angles, float))
    r, th = r.reshape(1, -1), th.reshape(1, -1)

    # exact geometric distance from source to each antenna
    d = np.sqrt(r*r + x*x - 2*r*x*np.sin(th))
    # dr = path length difference relative to center i.e. d - R
    # multiplying top and bottom by d + r to prevent floating point error
    dr = (x*x - 2*r*x*np.sin(th)) / (d+r)

    # convert dr into a phase shift
    return np.exp(-2j*np.pi*f0*dr/C) / np.sqrt(x.size)


def steering_cupy(x_gpu, f0, ranges_gpu, angles_gpu):
    """
    Generate steering vector for all candidate ranges/angles
    Use Cupy
    """
    xg = x_gpu[:, None]
    r = ranges_gpu[None, :]
    th = angles_gpu[None, :]
    d = cp.sqrt(r*r + xg*xg - 2*r*xg*cp.sin(th))
    dr = (xg*xg - 2*r*xg*cp.sin(th)) / (d+r)
    return cp.exp(-2j*cp.pi*f0*dr/C) / np.sqrt(x_gpu.size)

f0 = float(SYSTEM.fc)
x = np.asarray(ARRAY.positions(SYSTEM), float)
seed = int(getattr(SIM, 'rng_seed', 0))
reference_range = float(getattr(SIM, 'reference_range', 300.0))
reference_snr_db = float(getattr(SIM, 'reference_snr_db', 10.0))

# calibrate signal to the reference range and reference SNR
reference_index = int(np.argmin(np.abs(x)))
if WAVEFORM == 'gaussian':
    ref_delays = propagation_delays(reference_range, 0.0,
                                    clock_offset=CLOCK_OFFSET, antenna_x=x)
    ref_clean, _ = received_signal(np.array([ref_delays[reference_index]]),
                                    reference_range, 0.0,
                                    clock_offset=CLOCK_OFFSET, antenna_x=x)
    reference_peak_power = float(abs(ref_clean[reference_index, 0])**2)
else:
    # One independent reference QAM burst sets a fixed noise variance for all trials.
    qam_gen = MultiSourceGen(pw_sec=QAM_PW_SEC, M=QAM_M,
                             antenna_x=x, f0=f0, seed=seed+2)
    qam_cal = MultiSourceGen(pw_sec=QAM_PW_SEC, M=QAM_M,
                             antenna_x=x[reference_index:reference_index+1],
                             f0=f0, seed=seed+3)
    ref_clean, _, _ = qam_cal.generate([dict(
        range_m=reference_range, theta_deg=0.0, clock_offset=CLOCK_OFFSET
    )])
    reference_peak_power = float(np.max(np.abs(ref_clean[0])**2))

# NOISELESS: change to 0
variance = 0 if NOISELESS else reference_peak_power / 10.0**(reference_snr_db/10.0)

noise_scale = np.sqrt(variance/2.0)

rng_pos, rng_noise = np.random.default_rng(seed), np.random.default_rng(seed+1)
output_dir.mkdir(parents=True, exist_ok=True)
ranges = np.unique(np.r_[np.arange(R_MIN, R_MAX, RANGE_STEP_M), R_MAX])
angles_deg = np.unique(np.r_[np.arange(THETA_MIN, THETA_MAX, THETA_STEP_DEG), THETA_MAX])
angles = np.deg2rad(angles_deg)
rows = []
nr, nt = len(ranges), len(angles)
total = nr * nt
if USE_CUPY:
    x_gpu = cp.asarray(x, dtype=cp.float64)
    ranges_gpu = cp.asarray(ranges, dtype=cp.float64)
    angles_gpu = cp.asarray(angles, dtype=cp.float64)
if USE_MATCHED_FILTER:
    if WAVEFORM == 'gaussian':
        half = int(np.ceil(6 * SYSTEM.gaussian_sigma / SYSTEM.dt))
        lag = np.arange(-half, half + 1) * SYSTEM.dt
        pulse = np.exp(-0.5 * (lag / SYSTEM.gaussian_sigma)**2)
        pulse /= np.linalg.norm(pulse)
started = time.perf_counter()
print(f'Config: {root / "config.py"}', flush=True)
print(f'{N_MC} independent single-source trials; f0={f0/1e9:g} GHz, '
        f'B={SYSTEM.bandwidth/1e9:g} GHz, N={len(x)}, reference peak SNR='
        f'{reference_snr_db:g} dB at {reference_range:g} m', flush=True)
print(f'Fixed complex noise variance: {variance:.6g}; source amplitude unchanged across trials.', flush=True)
print(f'Data mode: {DATA_MODE}; MF: {USE_MATCHED_FILTER}; GPU: {USE_CUPY}', flush=True)
print(f'Waveform: {WAVEFORM}', flush=True)
if WAVEFORM == 'qam' and USE_MATCHED_FILTER:
    print('QAM MF uses the generated symbols as a known reference.', flush=True)

# generate N_MC ranges/thetas
for i in range(N_MC):
    trial_started = time.perf_counter()
    r_true = float(rng_pos.uniform(R_MIN,R_MAX))
    th_true = float(rng_pos.uniform(THETA_MIN,THETA_MAX))
    if WAVEFORM == 'gaussian':
        delays = propagation_delays(r_true, th_true, clock_offset=CLOCK_OFFSET, antenna_x=x)
        margin = 8.0*SYSTEM.gaussian_sigma
        n0 = int(np.floor((delays.min()-margin)/SYSTEM.dt))
        n1 = int(np.ceil((delays.max()+margin)/SYSTEM.dt))
        t = np.arange(n0,n1+1)*SYSTEM.dt

        # gaussian pulse
        clean, _ = received_signal(t, r_true, th_true, clock_offset=CLOCK_OFFSET, antenna_x=x)
    else:
        # one independent RRC-shaped QAM burst, shared across this source's antennas
        clean, t, source_info = qam_gen.generate([dict(
            range_m=r_true, theta_deg=th_true, clock_offset=CLOCK_OFFSET
        )])

    # add AWGN to signal
    # NOISELESS: change to clean.copy()
    if NOISELESS:
        noisy = clean.copy()
    else:
        noisy = clean + noise_scale*(rng_noise.standard_normal(clean.shape) + 1j*rng_noise.standard_normal(clean.shape))

    signal_seconds = time.perf_counter()-trial_started
    preprocessing_started = time.perf_counter()
    if USE_MATCHED_FILTER:
        if WAVEFORM == 'qam':
            # Known-QAM reference: generated symbols are available to the receiver here.
            reference = source_info[0]['waveform_reference']
            pulse = np.conj(reference[::-1]) / np.linalg.norm(reference)
        filtered = fftconvolve(noisy, pulse[None, :], mode='same', axes=1)
    else:
        filtered = noisy
    if DATA_MODE == 'full_matrix':
        data = filtered
    elif USE_MATCHED_FILTER:
        # one independently selected complex MF peak per antenna
        peaks_in = (resample(filtered, PEAK_TIME_OVERSAMPLE*len(t), axis=1)
                    if PEAK_TIME_OVERSAMPLE > 1 else filtered)
        peak_idx = np.argmax(np.abs(peaks_in)**2, axis=1)
        data = peaks_in[np.arange(len(x)), peak_idx][:, None]
    else:
        # one baseband f0 coefficient per antenna
        data = (noisy.sum(axis=1) / np.sqrt(len(t)))[:, None]
    preprocessing_seconds = time.perf_counter()-preprocessing_started

    sampled_peak_snr_db = None if NOISELESS else float(
        10*np.log10(np.max(abs(clean[reference_index])**2)/variance)
    )
    row = dict(trial=i, range_true_m=r_true, theta_true_deg=th_true,
               time_samples=len(t), noise_variance=float(variance),
               sampled_peak_snr_db=sampled_peak_snr_db)
    rows.append(row)

    search_started = time.perf_counter()
    if USE_CUPY:
        data_gpu = cp.asarray(data, dtype=GPU_COMPLEX_DTYPE)
        grid_scores_gpu = cp.empty(total, dtype=cp.float64)
        for start in range(0, total, GRID_CHUNK):
            ids = cp.arange(start, min(start + GRID_CHUNK, total))
            a = steering_cupy(
                x_gpu, f0, ranges_gpu[ids // nt], angles_gpu[ids % nt]
            ).astype(data_gpu.dtype)
            grid_scores_gpu[start:start+len(ids)] = cp.sum(
                cp.abs(a.conj().T @ data_gpu)**2, axis=1
            )
        best_index = int(cp.argmax(grid_scores_gpu).get())
    else:
        best_score = -np.inf
        best_index = 0
        for start in range(0, total, GRID_CHUNK):
            ids = np.arange(start, min(start + GRID_CHUNK, total))
            a = steering(x, f0, ranges[ids // nt], angles[ids % nt])
            scores = np.sum(np.abs(a.conj().T @ data)**2, axis=1)
            j = int(np.argmax(scores))
            if scores[j] > best_score:
                best_score = float(scores[j])
                best_index = int(ids[j])
    search_seconds = time.perf_counter()-search_started

    refinement_started = time.perf_counter()
    idx = best_index
    r0, th0 = float(ranges[idx // nt]), float(angles[idx % nt])
    energy = float(np.vdot(data, data).real)

    theta0_deg = float(np.rad2deg(th0))

    def phase_score_and_gradient(u):
        R = r0 + u[0]*RANGE_STEP_M
        theta = np.deg2rad(theta0_deg + u[1]*THETA_STEP_DEG)
        q = x*x - 2*R*x*np.sin(theta)
        dist = np.sqrt(R*R + q)
        delta_length = q/(dist+R)
        delay = delta_length/config_c
        d_delay_dR = -(x*x+x*np.sin(theta)*delta_length)/(
            config_c*dist*(dist+R))
        d_delay_dtheta = -R*x*np.cos(theta)/(config_c*dist)

        a_conj = np.exp(2j*np.pi*f0*delay)/np.sqrt(len(x))
        y = a_conj @ data
        dy_dR = (a_conj*(2j*np.pi*f0*d_delay_dR*RANGE_STEP_M)) @ data
        dy_dtheta = (a_conj*(2j*np.pi*f0*d_delay_dtheta
                   *np.deg2rad(THETA_STEP_DEG))) @ data
        score = float(np.vdot(y, y).real/energy)
        gradient = np.array([
            2*np.vdot(y, dy_dR).real/energy,
            2*np.vdot(y, dy_dtheta).real/energy,
        ])
        return score, gradient

    # Newton starts at the global phase-grid winner but has full-domain bounds.
    bounds = [
        ((R_MIN-r0)/RANGE_STEP_M, (R_MAX-r0)/RANGE_STEP_M),
        ((THETA_MIN-theta0_deg)/THETA_STEP_DEG,
         (THETA_MAX-theta0_deg)/THETA_STEP_DEG),
    ]
    best, score, n_eval, n_steps, stop_reason = bounded_damped_newton(
        phase_score_and_gradient, bounds
    )
    use_refined = bool(n_steps > 0 and score >= phase_score_and_gradient(
        np.zeros(2))[0]*(1-5*np.finfo(float).eps))

    # get the r and theta test
    rhat = r0 + best[0]*RANGE_STEP_M
    that = theta0_deg + best[1]*THETA_STEP_DEG
    theta_true = np.deg2rad(row['theta_true_deg'])
    xt,yt = row['range_true_m']*np.sin(theta_true),row['range_true_m']*np.cos(theta_true)
    xh,yh = rhat*np.sin(np.deg2rad(that)),rhat*np.cos(np.deg2rad(that))

    # localization error in range and overall
    range_error_m = float(rhat-row['range_true_m'])
    localization_error_m = float(np.hypot(xh-xt,yh-yt))
    refinement_seconds = time.perf_counter()-refinement_started
    trial_seconds = time.perf_counter()-trial_started

    row.update(range_est_m=float(rhat),theta_est_deg=float(that),
                range_error_m=range_error_m, abs_range_error_m=abs(range_error_m),
                localization_error_m=localization_error_m,
                x_true_m=float(xt),y_true_m=float(yt),
                coarse_range_m=r0,coarse_theta_deg=theta0_deg,
                range_at_boundary=bool(np.isclose(rhat,R_MIN) or np.isclose(rhat,R_MAX)),
                angle_at_boundary=bool(np.isclose(that, THETA_MIN) or np.isclose(that, THETA_MAX)), 
                refinement_converged=stop_reason in (
                    'Newton step below scaled tolerance',
                    'small accepted Newton step'),
                refinement_accepted=use_refined,
                refinement_evaluations=n_eval,
                newton_steps=n_steps,
                newton_reason=stop_reason,
                search_spatial_evaluations=total,
                signal_seconds=float(signal_seconds),
                preprocessing_seconds=float(preprocessing_seconds),
                search_seconds=float(search_seconds),
                refinement_seconds=float(refinement_seconds),
                estimator_seconds=float(trial_seconds-signal_seconds),
                trial_seconds=float(trial_seconds))

    
    if i == 0 or (i + 1) % 20 == 0:
        print(f'{i+1}/{N_MC}; elapsed {time.perf_counter()-started:.1f} s', flush=True)
   
# save the data
with (output_dir/'trials.csv').open('w', newline='') as handle:
    writer=csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
errors=np.array([row['abs_range_error_m'] for row in rows])
spatial_errors=np.array([row['localization_error_m'] for row in rows])
steady_rows = rows[1:] if N_MC > 1 else rows
summary=dict(n_mc=N_MC, seed=seed, config_path=str(root/'config.py'),
                waveform=WAVEFORM, qam_pw_sec=QAM_PW_SEC if WAVEFORM == 'qam' else None,
                qam_M=QAM_M if WAVEFORM == 'qam' else None,
                mf_reference=('generated QAM symbols (known waveform)' if WAVEFORM == 'qam'
                              and USE_MATCHED_FILTER else 'Gaussian pulse' if USE_MATCHED_FILTER
                              else None),
                matched_filter=USE_MATCHED_FILTER,
                data_mode=DATA_MODE, use_cupy=USE_CUPY,
                gpu_complex_dtype=GPU_COMPLEX_DTYPE if USE_CUPY else None,
                peak_time_oversample=PEAK_TIME_OVERSAMPLE if
                    DATA_MODE == 'one_value' and USE_MATCHED_FILTER else None,
                score_definition=('sum_t |steering(R,theta)^H data[:,t]|^2'
                                  if DATA_MODE == 'full_matrix' else
                                  '|steering(R,theta)^H data[:,0]|^2'),
                f0_hz=f0, bandwidth_hz=float(SYSTEM.bandwidth), n_antennas=len(x), 
                reference_peak_snr_db=reference_snr_db, reference_range_m=reference_range, 
                noise_variance=variance, reference_peak_power=reference_peak_power, 
                snr_definition='Peak signal power at reference antenna before matched filtering / complex noise variance', 
                range_min_m=R_MIN, range_max_m=R_MAX, 
                theta_min_deg=THETA_MIN, theta_max_deg=THETA_MAX, 
                range_step_m=RANGE_STEP_M, theta_step_deg=THETA_STEP_DEG, 
                mean_spatial_evaluations=float(np.mean([row['search_spatial_evaluations'] for row in rows])),
                range_bias_m=float(np.mean([row['range_error_m'] for row in rows])), 
                range_rmse_m=float(np.sqrt(np.mean(errors**2))), 
                median_abs_range_error_m=float(np.median(errors)), 
                p95_abs_range_error_m=float(np.quantile(errors, .95)), 
                localization_ale_m=float(np.mean(spatial_errors)),
                localization_rmse_m=float(np.sqrt(np.mean(spatial_errors**2))), 
                boundary_count=sum(row['range_at_boundary'] or row['angle_at_boundary'] for row in rows), 
                runtime_mean_signal_s=float(np.mean([row['signal_seconds'] for row in rows])),
                runtime_mean_preprocessing_s=float(np.mean([row['preprocessing_seconds'] for row in rows])),
                runtime_mean_search_s=float(np.mean([row['search_seconds'] for row in rows])),
                runtime_mean_refinement_s=float(np.mean([row['refinement_seconds'] for row in rows])),
                runtime_mean_estimator_s=float(np.mean([row['estimator_seconds'] for row in rows])),
                runtime_median_estimator_s=float(np.median([row['estimator_seconds'] for row in rows])),
                runtime_p95_estimator_s=float(np.quantile([row['estimator_seconds'] for row in rows], .95)),
                runtime_mean_estimator_excluding_first_s=float(np.mean([row['estimator_seconds'] for row in steady_rows])),
                runtime_mean_trial_s=float(np.mean([row['trial_seconds'] for row in rows])),
                elapsed_s=time.perf_counter()-started, 
                acquisition=('Same arrival-based bounds plus 8 sigma as single_user.py;'
                             ' not used as estimator search hints'
                             if WAVEFORM == 'gaussian' else
                             'All antenna arrivals plus 1.5 QAM pulse widths;'
                             ' not used as estimator search hints'))
(output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

# plot CDF
fig, ax=plt.subplots(figsize=(8, 5), layout='constrained')
sorted_errors=np.sort(errors)
ax.step(np.r_[0, sorted_errors], np.r_[0, np.arange(1, N_MC+1)/N_MC], where='post')
ax.set(xlabel='Absolute Range Error (m)', ylabel='Empirical CDF', ylim=(0, 1), xlim=(0, None), 
        title=f'{method_name}: CDF of Range Error ({N_MC} trials)')
ax.grid(True, alpha=.3)
fig.savefig(output_dir/'range_error_cdf.png', dpi=180)

# plot the 2-D colored locations + errors
mean_ale = np.mean(spatial_errors)
rmse = np.sqrt(np.mean(spatial_errors**2))

fig, ax=plt.subplots(figsize=(8, 7), layout='constrained')
sc=ax.scatter([row['x_true_m'] for row in rows], [row['y_true_m'] for row in rows], 
                c=spatial_errors, cmap='viridis', s=30, edgecolors='none')
fig.colorbar(sc, ax=ax, label='Absolute 2-D Localization Error (m)')
ax.plot(x, np.zeros_like(x), color='black', lw=2, label='Receiving array')
xmax = R_MAX * np.sin(np.deg2rad(max(abs(THETA_MIN),  abs(THETA_MAX))))

metrics_text = f"N={N_MC}\nMean ALE={mean_ale:.3f} m\nRMSE={rmse:.3f} m"
bbox_metrics = dict(boxstyle='round, pad=0.4',  facecolor='white',  edgecolor='gray',  alpha=0.9)
ax.text(0.02,  0.98,  metrics_text,  transform=ax.transAxes,  fontsize=10, 
        verticalalignment='top',  horizontalalignment='left',  bbox=bbox_metrics)

ax.set_xlim(-xmax-25, xmax+25)
ax.set_ylim(0, R_MAX+25)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'{method_name}: True User Positions Colored by Absolute Localization Error')
ax.grid(True, alpha=.3)
fig.savefig(output_dir/'true_positions_localization_error.png', dpi=180)
print(f'Saved all {N_MC} trials and both plots to {output_dir}', flush=True)
plt.show()
