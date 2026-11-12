import numpy as np

from scipy.constants import speed_of_light as C

import csv
import json
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve, resample
from datetime import datetime
try:
    import cupy as cp
except ImportError:
    cp = None

HRT_ALIGN_KERNEL = (
    cp.RawKernel(r'''
    extern "C" __global__
    void hrt_align(
        const double2* spectrum,  // [antenna, frequency]
        const double* freq,        // [frequency]
        const double* delay,       // [candidate, antenna]
        const double* weight,      // [candidate, antenna]
        const double f0,
        const int n_ant,
        const int nfft,
        const int n_candidates,
        float2* aligned           // [candidate, frequency]
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_candidates * nfft) return;

        int k = idx / nfft;
        int j = idx % nfft;
        double omega = 6.2831853071795864769 * (f0 + freq[j]);

        double real_sum = 0.0;
        double imag_sum = 0.0;

        for (int m = 0; m < n_ant; ++m) {
            int spatial_idx = k * n_ant + m;
            double phase = omega * delay[spatial_idx];
            double sine, cosine;
            sincos(phase, &sine, &cosine);

            double2 z = spectrum[m * nfft + j];
            double w = weight[spatial_idx];

            real_sum += w * (z.x * cosine - z.y * sine);
            imag_sum += w * (z.x * sine + z.y * cosine);
        }

        aligned[idx] = make_float2((float)real_sum, (float)imag_sum);
    }
    ''', 'hrt_align')
    if cp is not None else None
)

NOISELESS = False #False
WAVEFORM = 'gaussian'  # 'gaussian' or 'qam'
QAM_PW_SEC = 2e-9
QAM_M = 16
MF = True
if WAVEFORM not in ('gaussian', 'qam'):
    raise ValueError("WAVEFORM must be 'gaussian' or 'qam'")
output_dir = (Path(__file__).resolve().parent /
              f"hrt_time_domain_results_{WAVEFORM}_{'mf' if MF else 'no_mf'}_"
              f"{'noiseless' if NOISELESS else 'noisy'}_{datetime.now():%Y%m%d_%H%M%S}")

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
CLOCK_OFFSET = 0.0
RANGE_STEP_M = 1.0
THETA_STEP_DEG = 0.1
COARSE_RANGE_STEP_M = 10.0
COARSE_THETA_STEP_DEG = 0.2
N_COARSE_PEAKS = 5
PEAK_SEPARATION_R_M = 15.0
PEAK_SEPARATION_THETA_DEG = 0.4
FINE_HALF_RANGE_M = 15.0
FINE_HALF_THETA_DEG = 0.4
HRT_TILE_SIZE = 128
CENTER_TIME_HALF_SAMPLES = 4
FINE_TIME_OVERSAMPLE = 4
SHOW_PLOTS = True

# analytic gradient steps, in meter, deg, time samples
NEWTON_HESSIAN_STEPS = np.array([0.01/RANGE_STEP_M, 1e-3, 1e-3])

def bounded_damped_newton(evaluate, bounds, maxiter=25):
    """Maximize the continuous HRT score from its grid peak in local bounds."""
    lower, upper = np.asarray(bounds, float).T
    u = np.zeros(3)
    initial_score = evaluate(u)[0]
    scale = max(initial_score, 1e-30)
    radius = 1.0  # Initially, at most one (R, theta, time) grid step.
    evaluations = 1
    accepted_steps = 0
    reason = 'maximum iterations'

    for _ in range(maxiter):
        score, raw_gradient = evaluate(u)
        evaluations += 1
        gradient = raw_gradient/scale
        hessian = np.empty((3, 3))
        for axis, h in enumerate(NEWTON_HESSIAN_STEPS):
            shift = np.zeros(3)
            shift[axis] = h
            plus = evaluate(u+shift)[1]
            minus = evaluate(u-shift)[1]
            evaluations += 2
            hessian[:, axis] = (plus-minus)/(2*h*scale)
        hessian = (hessian+hessian.T)/2

        # Add damping until the local quadratic model has a maximum.
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

def prepare_hrt(hrt_input, x):
    """FFT the input waveforms once for both spatial passes."""
    xp = cp if cp is not None else np
    nt = hrt_input.shape[1]
    guard = int(np.ceil(np.max(np.abs(x))/config_c/SYSTEM.dt)) + 2
    nfft = next_fast_len(nt+2*guard)
    data = xp.asarray(hrt_input)
    spectrum = xp.fft.fft(xp.pad(data, ((0, 0),
                          (guard, nfft-nt-guard))), axis=1)
    freq = xp.fft.fftfreq(nfft, d=SYSTEM.dt)
    x_gpu = xp.asarray(x, dtype=xp.float64)[None, :]
    return spectrum, freq, x_gpu, guard, nt

def hrt_scores(prepared, f0, ranges, angles_deg):
    """GPU HRT scores at arbitrary points; retain each point's peak time."""
    if cp is None:
        raise ImportError("This HRT implementation requires CuPy")

    spectrum, freq, x_gpu, guard, nt = prepared

    # RawKernel expects these exact contiguous layouts and types.
    spectrum = cp.ascontiguousarray(spectrum, dtype=cp.complex128)
    freq = cp.ascontiguousarray(freq, dtype=cp.float64)
    ranges_gpu = cp.asarray(ranges, dtype=cp.float64)
    angles_gpu = cp.asarray(angles_deg, dtype=cp.float64)

    n_points = len(ranges)
    n_ant, nfft = spectrum.shape
    factor = FINE_TIME_OVERSAMPLE

    scores_gpu = cp.empty(n_points, dtype=cp.float64)
    times_gpu = cp.empty(n_points, dtype=cp.int64)

    for start in range(0, n_points, HRT_TILE_SIZE):
        stop = min(start + HRT_TILE_SIZE, n_points)
        n_candidates = stop - start

        r = ranges_gpu[start:stop, None]
        th = cp.deg2rad(angles_gpu[start:stop, None])

        dist = cp.sqrt(r*r + x_gpu*x_gpu - 2*r*x_gpu*cp.sin(th))
        delay = cp.ascontiguousarray((dist - r) / config_c)

        amp = r / dist
        weight = cp.ascontiguousarray(
            amp / cp.sqrt(cp.sum(amp*amp, axis=1, keepdims=True))
        )

        aligned = cp.empty((n_candidates, nfft), dtype=cp.complex64)
        threads = 256
        blocks = (n_candidates*nfft + threads - 1) // threads

        HRT_ALIGN_KERNEL(
            (blocks,), (threads,),
            (
                spectrum, freq, delay, weight,
                np.float64(f0),
                np.int32(n_ant),
                np.int32(nfft),
                np.int32(n_candidates),
                aligned,
            ),
        )

        if factor == 1:
            trace = cp.fft.ifft(aligned, axis=1)
        else:
            # Same frequency-bin layout as fftshift -> pad -> ifftshift.
            npos = (nfft + 1) // 2
            nneg = nfft - npos
            padded = cp.zeros(
                (n_candidates, factor*nfft), dtype=cp.complex64
            )
            padded[:, :npos] = aligned[:, :npos]
            if nneg:
                padded[:, -nneg:] = aligned[:, npos:]
            trace = cp.fft.ifft(padded, axis=1) * factor

        trace = trace[:, guard*factor:(guard+nt-1)*factor+1]
        power = cp.abs(trace)**2
        index = cp.argmax(power, axis=1)

        scores_gpu[start:stop] = power[cp.arange(n_candidates), index]
        times_gpu[start:stop] = index

    return cp.asnumpy(scores_gpu), cp.asnumpy(times_gpu)

# def hrt_scores(prepared, f0, ranges, angles_deg):
#     """HRT scores at arbitrary (range, angle) points; retain only peak time."""
#     xp = cp if cp is not None else np
#     spectrum, freq, x_gpu, guard, nt = prepared
#     scores = np.empty(len(ranges))
#     time_indices = np.empty(len(ranges), dtype=int)

#     for start in range(0, len(ranges), HRT_TILE_SIZE):
#         stop = min(start+HRT_TILE_SIZE, len(ranges))
#         r = xp.asarray(ranges[start:stop], dtype=xp.float64)[:, None]
#         th = xp.deg2rad(xp.asarray(angles_deg[start:stop], dtype=xp.float64))[:, None]
#         dist = xp.sqrt(r*r+x_gpu*x_gpu-2*r*x_gpu*xp.sin(th))
#         delay = (dist-r)/config_c
#         amp = r/dist
#         weight = amp/xp.sqrt(xp.sum(amp*amp, axis=1, keepdims=True))
#         phase = xp.exp(2j*xp.pi*(f0+freq[None, None, :])*delay[:, :, None])
#         aligned = xp.einsum('kmf,mf->kf', weight[:, :, None]*phase,
#                             spectrum, optimize=True).astype(xp.complex64)
#         factor = FINE_TIME_OVERSAMPLE
#         nfft = aligned.shape[1]

#         if factor == 1:
#             trace = xp.fft.ifft(aligned, axis=1)
#         else:
#             pad = nfft * (factor - 1)
#             left = (factor * nfft)//2 - nfft//2
#             shifted = xp.fft.fftshift(aligned, axes=1)
#             padded = xp.pad(shifted, ((0, 0), (left, pad-left)))
#             trace = xp.fft.ifft(
#                 xp.fft.ifftshift(padded, axes=1), axis=1
#             ) * factor

#         trace = trace[:, guard*factor:(guard+nt-1)*factor+1]
#         power = xp.abs(trace)**2
#         index = xp.argmax(power, axis=1)
#         scores[start:stop] = cp.asnumpy(xp.max(power, axis=1)) if cp is not None else xp.max(power, axis=1)
#         time_indices[start:stop] = cp.asnumpy(index) if cp is not None else index
#     return scores, time_indices


def search_hrt(hrt_input, t, x, f0, r_true, th_true):
    """Full-domain approximate HRT followed by exact local FFT HRT."""
    coarse_r = np.linspace(R_MIN, R_MAX,
                           int(round((R_MAX-R_MIN)/COARSE_RANGE_STEP_M))+1)
    coarse_th = np.linspace(THETA_MIN, THETA_MAX,
                            int(round((THETA_MAX-THETA_MIN)/COARSE_THETA_STEP_DEG))+1)
    rr, th = np.meshgrid(coarse_r, coarse_th, indexing='ij')

    preparation_started = time.perf_counter()
    prepared = prepare_hrt(hrt_input, x)
    preparation_seconds = time.perf_counter()-preparation_started
    coarse_started = time.perf_counter()
    coarse_scores, _ = hrt_scores(prepared, f0, rr.ravel(), th.ravel())
    coarse_seconds = time.perf_counter()-coarse_started
    coarse_map = coarse_scores.reshape(rr.shape)

    peaks = []
    for idx in np.argsort(coarse_scores)[::-1]:
        r, angle = float(rr.ravel()[idx]), float(th.ravel()[idx])
        if all(abs(r-prev_r) >= PEAK_SEPARATION_R_M or
               abs(angle-prev_th) >= PEAK_SEPARATION_THETA_DEG
               for prev_r, prev_th in peaks):
            peaks.append((r, angle))
            if len(peaks) == N_COARSE_PEAKS:
                break

    regions = []
    for r, angle in peaks:
        rlo, rhi = max(R_MIN, r-FINE_HALF_RANGE_M), min(R_MAX, r+FINE_HALF_RANGE_M)
        alo, ahi = max(THETA_MIN, angle-FINE_HALF_THETA_DEG), min(THETA_MAX, angle+FINE_HALF_THETA_DEG)
        local_r = np.linspace(rlo, rhi, int(round((rhi-rlo)/RANGE_STEP_M))+1)
        local_th = np.linspace(alo, ahi, int(round((ahi-alo)/THETA_STEP_DEG))+1)
        local_rr, local_angles = np.meshgrid(local_r, local_th, indexing='ij')
        regions.append((rlo, rhi, alo, ahi, local_rr.ravel(), local_angles.ravel()))

    # deduplicate overlapping search regions before evaluating the expensive HRT
    points = np.unique(np.concatenate([
        np.column_stack((region[4], region[5])) for region in regions]), axis=0)
    fine_started = time.perf_counter()
    fine_scores, fine_times = hrt_scores(prepared, f0, points[:, 0], points[:, 1])
    fine_seconds = time.perf_counter()-fine_started
    best = int(np.argmax(fine_scores))
    r0, th0 = map(float, points[best])
    roi_bounds = (min(region[0] for region in regions),
                  max(region[1] for region in regions),
                  min(region[2] for region in regions),
                  max(region[3] for region in regions))
    coverage = any(region[0] <= r_true <= region[1] and
                   region[2] <= th_true <= region[3] for region in regions)
    return dict(r0=r0, th0=th0, tau0=float(t[0] + fine_times[best] * SYSTEM.dt / FINE_TIME_OVERSAMPLE),
                roi_bounds=roi_bounds, coverage=bool(coverage),
                coarse_map=coarse_map, coarse_ranges=coarse_r,
                coarse_angles=coarse_th, coarse_peaks=peaks,
                n_spatial=len(coarse_scores)+len(points),
                n_coarse=len(coarse_scores), n_fine=len(points),
                preparation_seconds=preparation_seconds,
                coarse_seconds=coarse_seconds, fine_seconds=fine_seconds)

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
variance = 0.0 if NOISELESS else reference_peak_power / 10.0**(reference_snr_db/10.0)

noise_scale = np.sqrt(variance/2.0)
rng_pos, rng_noise = np.random.default_rng(seed), np.random.default_rng(seed+1)
output_dir.mkdir(parents=True, exist_ok=True)
rows = []
rt_plot = None
started = time.perf_counter()
print(f'Config: {root / "config.py"}', flush=True)
print(f'{N_MC} independent single-source trials; f0={f0/1e9:g} GHz, '
      f'B={SYSTEM.bandwidth/1e9:g} GHz, N={len(x)}, reference peak SNR='
      f'{"noiseless" if NOISELESS else str(reference_snr_db) + " dB at " + str(reference_range) + " m"}', flush=True)
print(f'Fixed complex noise variance: {variance:.6g}; source amplitude unchanged across trials.', flush=True)
print(f'Waveform: {WAVEFORM}; MF: {MF}', flush=True)
if WAVEFORM == 'qam' and MF:
    print('QAM MF uses the generated symbols as a known reference.', flush=True)
if WAVEFORM == 'gaussian' and MF:
    half = int(np.ceil(6 * SYSTEM.gaussian_sigma / SYSTEM.dt))
    lag = np.arange(-half, half + 1) * SYSTEM.dt
    pulse = np.exp(-0.5 * (lag / SYSTEM.gaussian_sigma)**2)
    pulse /= np.linalg.norm(pulse)

# generate N_MC ranges/thetas and evaluate the RT for each waveform
for i in range(N_MC):
    trial_started = time.perf_counter()
    r_true = float(rng_pos.uniform(R_MIN, R_MAX))
    th_true = float(rng_pos.uniform(THETA_MIN, THETA_MAX))
    if WAVEFORM == 'gaussian':
        delays = propagation_delays(r_true, th_true, clock_offset=CLOCK_OFFSET, antenna_x=x)
        margin = 8.0*SYSTEM.gaussian_sigma
        n0 = int(np.floor((delays.min()-margin)/SYSTEM.dt))
        n1 = int(np.ceil((delays.max()+margin)/SYSTEM.dt))
        t = np.arange(n0,n1+1)*SYSTEM.dt

        # gaussian pulse
        clean, _ = received_signal(t, r_true, th_true, clock_offset=CLOCK_OFFSET, antenna_x=x)
    else:
        # one independent RRC-shaped QAM burst
        clean, t, source_info = qam_gen.generate([dict(
            range_m=r_true, theta_deg=th_true, clock_offset=CLOCK_OFFSET
        )])

    # add AWGN to signal
    noisy = clean.copy() if NOISELESS else clean + noise_scale*(
        rng_noise.standard_normal(clean.shape) + 1j*rng_noise.standard_normal(clean.shape))

    signal_seconds = time.perf_counter()-trial_started
    preprocessing_started = time.perf_counter()
    # matched filtering
    if MF:
        if WAVEFORM == 'qam':
            # known QAM reference symbols
            reference = source_info[0]['waveform_reference']
            pulse = np.conj(reference[::-1]) / np.linalg.norm(reference)
        hrt_input = fftconvolve(noisy, pulse[None, :], mode='same', axes=1)
    else:
        hrt_input = noisy

    preprocessing_seconds = time.perf_counter()-preprocessing_started

    # full time domain HRT, exact FFT HRT around peaks
    hrt_started = time.perf_counter()
    search = search_hrt(hrt_input, t, x, f0, r_true, th_true)
    hrt_seconds = time.perf_counter()-hrt_started
    r0, th0, tau0 = search['r0'], search['th0'], search['tau0']

    if i == 0:
        rt_plot = (search['coarse_map'], search['coarse_ranges'],
                   search['coarse_angles'], th_true, r_true, th0, r0)

    sampled_peak_snr_db = None if NOISELESS else float(
        10*np.log10(np.max(abs(clean[reference_index])**2)/variance))
    row = dict(trial=i,range_true_m=r_true,theta_true_deg=th_true,
               time_samples=len(t),noise_variance=float(variance),
               sampled_peak_snr_db=sampled_peak_snr_db)

    refinement_started = time.perf_counter()
    # continous HRT refinement, range/angle/ref arrival time
    # does not require template WF knowledge
    dr, dtheta = RANGE_STEP_M, THETA_STEP_DEG
    dt = SYSTEM.dt
    guard = int(np.ceil(np.max(np.abs(x))/config_c/dt))+2
    nfft = next_fast_len(len(t)+2*guard)
    spectrum = np.fft.fft(
        np.pad(hrt_input, ((0, 0), (guard, nfft-len(t)-guard))), axis=1)
    freq = np.fft.fftfreq(nfft,dt)
    total_freq = freq[None,:]+f0

    def hrt_peak_and_gradient(u):
        R,theta_deg,tau = r0+u[0]*dr,th0+u[1]*dtheta,tau0+u[2]*dt
        theta = np.deg2rad(theta_deg)
        q = x*x-2*R*x*np.sin(theta)
        dist = np.sqrt(R*R+q)
        delta_length = q/(dist+R)
        d_dist_dR = (R-x*np.sin(theta))/dist
        d_dist_dtheta = -R*x*np.cos(theta)/dist
        delta = delta_length/config_c
        d_delta_dR = -(x*x+x*np.sin(theta)*delta_length)/(
            config_c*dist*(dist+R))
        d_delta_dtheta = d_dist_dtheta/config_c

        amplitude = R/dist
        d_amp_dR = 1/dist-R*d_dist_dR/dist**2
        d_amp_dtheta = -R*d_dist_dtheta/dist**2
        norm = np.linalg.norm(amplitude)
        weight = amplitude/norm
        d_weight_dR = (d_amp_dR-weight*np.dot(weight,d_amp_dR))/norm
        d_weight_dtheta = (d_amp_dtheta-weight*np.dot(weight,d_amp_dtheta))/norm

        aligned = spectrum*np.exp(2j*np.pi*total_freq*delta[:,None])
        U_f = np.sum(weight[:,None]*aligned,axis=0)
        dU_f_dR = np.sum(aligned*(d_weight_dR[:,None]
            + weight[:,None]*2j*np.pi*total_freq*d_delta_dR[:,None]),axis=0)*dr
        dU_f_dtheta = np.sum(aligned*(d_weight_dtheta[:,None]
            + weight[:,None]*2j*np.pi*total_freq*d_delta_dtheta[:,None]),axis=0)
        dU_f_dtheta *= np.deg2rad(dtheta)

        time_phase = np.exp(2j*np.pi*freq*(tau-t[0]+guard*dt))
        value = np.dot(U_f,time_phase)/nfft
        derivatives = np.array([
            np.dot(dU_f_dR,time_phase),
            np.dot(dU_f_dtheta,time_phase),
            np.dot(U_f,2j*np.pi*freq*dt*time_phase)
        ])/nfft
        score = float(abs(value)**2)
        gradient = 2*np.real(np.conj(value)*derivatives)
        return score,gradient

    initial_score = hrt_peak_and_gradient((0.0,0.0,0.0))[0]
    roi_rlo, roi_rhi, roi_alo, roi_ahi = search['roi_bounds']
    bounds = [
        ((roi_rlo-r0)/dr,(roi_rhi-r0)/dr),
        (max(-1.0,(roi_alo-th0)/dtheta),min(1.0,(roi_ahi-th0)/dtheta)),
        (max(-1.0,(t[0]-tau0)/dt),min(1.0,(t[-1]-tau0)/dt))
    ]

    best, best_score, newton_evaluations, newton_steps, newton_reason = (
        bounded_damped_newton(hrt_peak_and_gradient, bounds)
    )
    refinement_accepted = bool(best_score > initial_score*(
        1-5*np.finfo(float).eps) and newton_steps > 0)
    refinement_seconds = time.perf_counter()-refinement_started
    rhat,that,tau_hat = r0+best[0]*dr,th0+best[1]*dtheta,tau0+best[2]*dt
    trial_seconds = time.perf_counter()-trial_started

    theta_true = np.deg2rad(row['theta_true_deg'])
    xt,yt = row['range_true_m']*np.sin(theta_true),row['range_true_m']*np.cos(theta_true)
    xh,yh = rhat*np.sin(np.deg2rad(that)),rhat*np.cos(np.deg2rad(that))
    range_error_m = float(rhat-row['range_true_m'])
    localization_error_m = float(np.hypot(xh-xt,yh-yt))

    row.update(range_est_m=float(rhat),theta_est_deg=float(that),
               range_error_m=range_error_m,abs_range_error_m=abs(range_error_m),
               localization_error_m=localization_error_m,
               x_true_m=float(xt),y_true_m=float(yt),
               coarse_range_m=r0,coarse_theta_deg=th0,
               coarse_tau_s=tau0,tau_est_s=tau_hat,
               hrt_grid_range_m=r0,
               global_coarse_range_m=search['coarse_peaks'][0][0],
               global_coarse_theta_deg=search['coarse_peaks'][0][1],
               roi_contains_truth=search['coverage'],
               search_spatial_evaluations=search['n_spatial'],
               coarse_hrt_evaluations=search['n_coarse'],
               fine_hrt_evaluations=search['n_fine'],
               range_at_boundary=bool(np.isclose(rhat,R_MIN) or np.isclose(rhat,R_MAX)),
               angle_at_boundary=bool(np.isclose(that,THETA_MIN) or np.isclose(that,THETA_MAX)),
               refinement_converged=newton_reason in (
                   'Newton step below scaled tolerance',
                   'small accepted Newton step'),
               refinement_evaluations=int(newton_evaluations),
               newton_evaluations=int(newton_evaluations),
               newton_accepted=bool(newton_steps > 0),
               newton_steps=int(newton_steps),
               newton_reason=newton_reason,
               signal_seconds=float(signal_seconds),
               preprocessing_seconds=float(preprocessing_seconds),
               hrt_preparation_seconds=search['preparation_seconds'],
               coarse_hrt_seconds=search['coarse_seconds'],
               fine_hrt_seconds=search['fine_seconds'],
               hrt_seconds=float(hrt_seconds),
               search_seconds=float(hrt_seconds),
               refinement_seconds=float(refinement_seconds),
               estimator_seconds=float(trial_seconds-signal_seconds),
               trial_seconds=float(trial_seconds),
               refinement_accepted=refinement_accepted)
    rows.append(row)
    if i % 20 == 0:
        print(f'  {i+1}/{N_MC}; elapsed {time.perf_counter()-started:.1f} s; '
              f'signal {signal_seconds:.2f} s; coarse HRT {search["coarse_seconds"]:.2f} s; '
              f'fine HRT {search["fine_seconds"]:.2f} s; '
              f'refinement {refinement_seconds:.2f} s; '
              f'evaluations {newton_evaluations}; '
              f'Newton {newton_reason}',flush=True)

# save the data
with (output_dir/'trials.csv').open('w', newline='') as handle:
    writer=csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
errors=np.array([row['abs_range_error_m'] for row in rows])
spatial_errors=np.array([row['localization_error_m'] for row in rows])
steady_rows = rows[1:] if N_MC > 1 else rows
summary=dict(n_mc=N_MC, seed=seed, config_path=str(root/'config.py'),
             waveform=WAVEFORM, matched_filter=MF,
             qam_pw_sec=QAM_PW_SEC if WAVEFORM == 'qam' else None,
             qam_M=QAM_M if WAVEFORM == 'qam' else None,
             mf_reference=('generated QAM symbols (known waveform)' if WAVEFORM == 'qam' and MF
                           else 'Gaussian pulse' if MF else None),
             f0_hz=f0, bandwidth_hz=float(SYSTEM.bandwidth), n_antennas=len(x),
             reference_peak_snr_db=None if NOISELESS else reference_snr_db,
             reference_range_m=reference_range, noise_variance=variance,
             reference_peak_power=reference_peak_power,
             snr_definition='Noiseless' if NOISELESS else 'Peak signal power at reference antenna / complex noise variance',
             range_min_m=R_MIN,range_max_m=R_MAX,
             theta_min_deg=THETA_MIN,theta_max_deg=THETA_MAX,
             range_step_m=RANGE_STEP_M,theta_step_deg=THETA_STEP_DEG,
             coarse_range_step_m=COARSE_RANGE_STEP_M,
             coarse_theta_step_deg=COARSE_THETA_STEP_DEG,
             n_coarse_peaks=N_COARSE_PEAKS,
             peak_separation_range_m=PEAK_SEPARATION_R_M,
             peak_separation_theta_deg=PEAK_SEPARATION_THETA_DEG,
             fine_half_range_m=FINE_HALF_RANGE_M,
             fine_half_theta_deg=FINE_HALF_THETA_DEG,
             roi_coverage_rate=float(np.mean([row['roi_contains_truth'] for row in rows])),
             mean_spatial_evaluations=float(np.mean([row['search_spatial_evaluations'] for row in rows])),
             mean_coarse_hrt_evaluations=float(np.mean([row['coarse_hrt_evaluations'] for row in rows])),
             mean_fine_hrt_evaluations=float(np.mean([row['fine_hrt_evaluations'] for row in rows])),
             range_bias_m=float(np.mean([row['range_error_m'] for row in rows])),
             range_rmse_m=float(np.sqrt(np.mean(errors**2))),
             median_abs_range_error_m=float(np.median(errors)),
             p95_abs_range_error_m=float(np.quantile(errors,.95)),
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
             acquisition=('Truth-centered recording window plus 8 sigma' if WAVEFORM == 'gaussian'
                          else 'Recording window spans QAM arrival at all antennas')
                         + '; absolute endpoints do not restrict the HRT spatial search',
             hrt_range_roi='Full-domain time-domain interpolated HRT; separated data-selected peaks supply exact FFT HRT regions',
             coarse_time_window='Full time search; no center-antenna window used',
             fine_time_oversample=FINE_TIME_OVERSAMPLE)
(output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

# plot CDF
fig, ax=plt.subplots(figsize=(8, 5), layout='constrained')
sorted_errors=np.sort(errors)
ax.step(np.r_[0, sorted_errors], np.r_[0, np.arange(1, N_MC+1)/N_MC], where='post')
ax.set(xlabel='Absolute Range Error (m)', ylabel='Empirical CDF', ylim=(0, 1), xlim=(0, None),
       title=f'HRT + Refinement: CDF of Range Error ({N_MC} trials)')
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
xmax = R_MAX * np.sin(np.deg2rad(max(abs(THETA_MIN),abs(THETA_MAX))))

metrics_text = f"N={N_MC}\nMean ALE={mean_ale:.3f} m\nRMSE={rmse:.3f} m"
bbox_metrics = dict(boxstyle='round, pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left', bbox=bbox_metrics)

ax.set_xlim(-xmax-25, xmax+25)
ax.set_ylim(0, R_MAX+25)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('HRT + Refinement: True User Positions Colored by Absolute Localization Error')
ax.grid(True, alpha=.3)
fig.savefig(output_dir/'true_positions_localization_error.png', dpi=180)

# RT map from the first trial
rt_map, rt_ranges, rt_angles, th_true, r_true, th0, rt_peak_range = rt_plot
fig, ax=plt.subplots(figsize=(9, 5), layout='constrained')
im=ax.pcolormesh(rt_angles, rt_ranges, rt_map/max(float(rt_map.max()),1e-300),
                 shading='auto', cmap='viridis', rasterized=True)
fig.colorbar(im, ax=ax, label='Normalized |RT|²')
ax.plot(th_true,r_true,'rx',ms=9,mew=2,label='True source')
ax.plot(th0,rt_peak_range,'wo',ms=7,mec='k',label='RT peak')
ax.set(xlabel='Angle (deg)',ylabel='Range (m)',
       title=f'Global Coarse HRT of {"MF" if MF else "Raw"} {WAVEFORM.upper()} Waveform (Trial 0)')
ax.legend()
fig.savefig(output_dir/'radon_transform.png',dpi=180)

print(f'Saved all {N_MC} trials and plots to {output_dir}', flush=True)
if SHOW_PLOTS:
    plt.show()
