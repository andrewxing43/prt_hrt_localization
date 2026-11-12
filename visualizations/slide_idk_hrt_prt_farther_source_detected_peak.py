"""Four-panel PRT/HRT visual for one source at 300 m, 50 degrees.

Copy into experiments/ of the localization repository. The top panels show
R/theta maps, and the bottom panels show range profiles at the refined angle.
All four panels use the same raw noisy Gaussian observation. Each grid node
searches the entire recorded center-arrival-time axis; the winner is refined
continuously in R, theta, and tau with bounded damped Newton steps.
The spatial grid is the local range/angle window specified below.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len

try:
    import cupy as cp
except ImportError:
    cp = None


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
if not (ROOT / "config.py").is_file():
    ROOT = HERE.parent / "sic2_integration_fixture"  # Standalone copy only.
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal


R_TRUE = 300.0
THETA_TRUE_DEG = 50.0
CLOCK_OFFSET = 0.0
RANGES = np.arange(200.0, 400.0 + 1.0, 2.0)
ANGLES_DEG = np.arange(47.0, 53.0 + 0.025, 0.025)
CHUNK = 128
SHOW_PLOT = True
OUTPUT = HERE / "prt_hrt_four_panel_300m50.png"


ALIGN_KERNEL = (cp.RawKernel(r'''
extern "C" __global__ void align_trace(
    const double2* spectrum, const double* freq, const double* delay,
    const double f0, const int n_ant, const int nfft,
    const int n_candidates, float2* aligned
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates * nfft) return;
    int k = idx / nfft, j = idx % nfft;
    double omega = 6.283185307179586 * (f0 + freq[j]);
    double real_sum = 0.0, imag_sum = 0.0;
    for (int m = 0; m < n_ant; ++m) {
        double phase = omega * delay[k*n_ant + m];
        double sine, cosine;
        sincos(phase, &sine, &cosine);
        double2 z = spectrum[m*nfft + j];
        real_sum += z.x*cosine - z.y*sine;
        imag_sum += z.x*sine + z.y*cosine;
    }
    aligned[idx] = make_float2((float)(real_sum/n_ant),
                               (float)(imag_sum/n_ant));
}''', 'align_trace') if cp is not None else None)


def relative_delay(r, theta_deg, x, kind, xp=np):
    theta = xp.deg2rad(theta_deg)
    if kind == "HRT":
        return (xp.sqrt(r*r + x*x - 2*r*x*xp.sin(theta)) - r)/C
    return (-xp.sin(theta)*x/C
            + xp.cos(theta)**2*x*x/(2*C*r))


def prepare_transform(data, t, x):
    dt = float(t[1] - t[0])
    guard = int(np.ceil(np.max(np.abs(x))/C/dt)) + 4
    nfft = next_fast_len(len(t) + 2*guard)
    padded = np.pad(data, ((0, 0), (guard, nfft-len(t)-guard)))
    spectrum = np.ascontiguousarray(np.fft.fft(padded, axis=1))
    freq = np.fft.fftfreq(nfft, dt)
    return spectrum, freq, guard, len(t)


def score_map(prepared, t, x, kind, ranges=None, angles=None):
    """Profile every candidate over all recorded arrival times, no tau hint."""
    if ranges is None:
        ranges = RANGES
    if angles is None:
        angles = ANGLES_DEG
    spectrum, freq, guard, nt = prepared
    R, TH = np.meshgrid(ranges, angles, indexing="ij")
    rr, aa = R.ravel(), TH.ravel()
    scores = np.empty(len(rr))
    time_indices = np.empty(len(rr), dtype=int)
    nfft = len(freq)

    if cp is not None:
        spec_gpu = cp.ascontiguousarray(cp.asarray(spectrum), dtype=cp.complex128)
        freq_gpu = cp.ascontiguousarray(cp.asarray(freq), dtype=cp.float64)
        x_gpu = cp.asarray(x, dtype=cp.float64)[None, :]

    for start in range(0, len(rr), CHUNK if cp is not None else 8):
        stop = min(start + (CHUNK if cp is not None else 8), len(rr))
        if cp is not None:
            r = cp.asarray(rr[start:stop], dtype=cp.float64)[:, None]
            th = cp.asarray(aa[start:stop], dtype=cp.float64)[:, None]
            delay = cp.ascontiguousarray(relative_delay(r, th, x_gpu, kind, cp))
            n_candidates = stop-start
            aligned = cp.empty((n_candidates, nfft), dtype=cp.complex64)
            threads = 256
            blocks = (n_candidates*nfft + threads-1)//threads
            ALIGN_KERNEL((blocks,), (threads,),
                         (spec_gpu, freq_gpu, delay, np.float64(SYSTEM.fc),
                          np.int32(len(x)), np.int32(nfft),
                          np.int32(n_candidates), aligned))
            trace = cp.fft.ifft(aligned, axis=1)[:, guard:guard+nt]
            power = cp.abs(trace)**2
            index = cp.argmax(power, axis=1)
            scores[start:stop] = cp.asnumpy(power[cp.arange(n_candidates), index])
            time_indices[start:stop] = cp.asnumpy(index)
        else:
            delay = relative_delay(rr[start:stop, None],
                                   aa[start:stop, None], x[None, :], kind)
            phase = np.exp(2j*np.pi*(SYSTEM.fc + freq[None, None, :])
                           * delay[:, :, None])
            aligned = np.mean(spectrum[None, :, :]*phase, axis=1)
            power = np.abs(np.fft.ifft(aligned, axis=1)[:, guard:guard+nt])**2
            index = np.argmax(power, axis=1)
            scores[start:stop] = power[np.arange(stop-start), index]
            time_indices[start:stop] = index
    return scores.reshape(R.shape), time_indices.reshape(R.shape)


def continuous_score(prepared, t, x, kind, r, theta, tau):
    """Continuous Fourier evaluation of the same aligned transform trace."""
    spectrum, freq, guard, _ = prepared
    delay = relative_delay(r, theta, x, kind)
    aligned = np.mean(spectrum*np.exp(
        2j*np.pi*(SYSTEM.fc+freq[None, :])*delay[:, None]), axis=0)
    shift_time = tau - t[0] + guard*(t[1]-t[0])
    value = np.dot(aligned, np.exp(2j*np.pi*freq*shift_time))/len(freq)
    return float(abs(value)**2)


def refine_peak(prepared, t, x, kind, r0, theta0, tau0):
    """Bounded damped Newton maximization of the continuous R/theta/tau peak."""
    dr = float(RANGES[1] - RANGES[0])
    dtheta = float(ANGLES_DEG[1] - ANGLES_DEG[0])
    dt = float(t[1] - t[0])
    lower = np.array([max(-20., (RANGES[0]-r0)/dr),
                      max(-1., (ANGLES_DEG[0]-theta0)/dtheta),
                      max(-2., (t[0]-tau0)/dt)])
    upper = np.array([min(20., (RANGES[-1]-r0)/dr),
                      min(1., (ANGLES_DEG[-1]-theta0)/dtheta),
                      min(2., (t[-1]-tau0)/dt)])
    initial = continuous_score(prepared, t, x, kind, r0, theta0, tau0)
    evaluations = 0

    def objective(u):
        nonlocal evaluations
        evaluations += 1
        return -continuous_score(prepared, t, x, kind,
                                 r0+u[0]*dr, theta0+u[1]*dtheta,
                                 tau0+u[2]*dt)/max(initial, 1e-30)

    u = np.zeros(3)
    f = objective(u)
    radius = 1.0
    h = np.array([0.1, 0.05, 0.05])
    for _ in range(25):
        gradient = np.empty(3)
        hessian = np.empty((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = h[j]
            plus, minus = objective(u+step), objective(u-step)
            gradient[j] = (plus-minus)/(2*h[j])
            hessian[j, j] = (plus-2*f+minus)/(h[j]**2)
            for k in range(j):
                other = np.zeros(3)
                other[k] = h[k]
                hessian[j, k] = hessian[k, j] = (
                    objective(u+step+other)-objective(u+step-other)
                    -objective(u-step+other)+objective(u-step-other)
                )/(4*h[j]*h[k])
        damping = max(0., 1e-8-np.linalg.eigvalsh(hessian)[0])
        direction = -np.linalg.solve(hessian+damping*np.eye(3), gradient)
        norm = np.linalg.norm(direction)
        if norm > radius:
            direction *= radius/norm
        candidate = np.clip(u+direction, lower, upper)
        candidate_value = objective(candidate)
        if candidate_value < f:
            gain = f-candidate_value
            u, f = candidate, candidate_value
            radius = min(8., 2*radius)
            if gain < 1e-13 or np.linalg.norm(direction) < 1e-7:
                break
        else:
            radius *= 0.5
            if radius < 1e-5:
                break
    accepted = f < objective(np.zeros(3))
    return (r0+u[0]*dr, theta0+u[1]*dtheta, tau0+u[2]*dt,
            accepted, evaluations)


def main():
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    center = int(np.argmin(np.abs(x)))
    reference_range = float(getattr(SIM, "reference_range", 300.0))
    reference_snr_db = float(getattr(SIM, "reference_snr_db", 10.0))
    rng = np.random.default_rng(int(getattr(SIM, "rng_seed", 0)) + 1)

    reference_delay = propagation_delays(
        reference_range, 0.0, clock_offset=CLOCK_OFFSET, antenna_x=x
    )[center]
    reference, _ = received_signal(
        np.array([reference_delay]), reference_range, 0.0,
        clock_offset=CLOCK_OFFSET, antenna_x=x,
    )
    variance = (float(abs(reference[center, 0])**2)
                / 10.0**(reference_snr_db/10.0))

    delays = propagation_delays(
        R_TRUE, THETA_TRUE_DEG, clock_offset=CLOCK_OFFSET, antenna_x=x
    )
    margin = 8*SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min() - margin)/SYSTEM.dt))
    n1 = int(np.ceil((delays.max() + margin)/SYSTEM.dt))
    t = np.arange(n0, n1 + 1)*SYSTEM.dt
    clean, _ = received_signal(
        t, R_TRUE, THETA_TRUE_DEG,
        clock_offset=CLOCK_OFFSET, antenna_x=x,
    )
    noisy = clean + np.sqrt(variance/2)*(
        rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape)
    )
    prepared = prepare_transform(noisy, t, x)
    results = {name: score_map(prepared, t, x, name)
               for name in ("HRT", "PRT")}
    maps = {name: result[0] for name, result in results.items()}

    fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.2),
                           layout="constrained", height_ratios=[1.0, 0.72])
    labels = {"HRT": "Hyperbolic Radon Transform",
              "PRT": "Parabolic Radon Transform"}
    for column, (name, score) in enumerate(maps.items()):
        maximum = score.max()
        im = ax[0, column].imshow(
            10*np.log10(np.maximum(score/maximum, 1e-15)),
            origin="lower", aspect="auto", cmap="magma", vmin=-35, vmax=0,
            extent=(ANGLES_DEG[0], ANGLES_DEG[-1], RANGES[0], RANGES[-1]),
        )
        ir, ia = np.unravel_index(np.argmax(score), score.shape)
        r_peak, theta_peak = RANGES[ir], ANGLES_DEG[ia]
        tau_peak = t[results[name][1][ir, ia]]
        rhat, that, tau_hat, accepted, nfev = refine_peak(
            prepared, t, x, name, r_peak, theta_peak, tau_peak,
        )
        xt = R_TRUE*np.sin(np.deg2rad(THETA_TRUE_DEG))
        yt = R_TRUE*np.cos(np.deg2rad(THETA_TRUE_DEG))
        xhat = rhat*np.sin(np.deg2rad(that))
        yhat = rhat*np.cos(np.deg2rad(that))
        error_2d = np.hypot(xhat-xt, yhat-yt)
        ax[0, column].plot(THETA_TRUE_DEG, R_TRUE, "o", ms=9,
                           mfc="none", mec="cyan", mew=2.2,
                           label=f"Source: R={R_TRUE:g} m, θ={THETA_TRUE_DEG:g}°")
        ax[0, column].plot(that, rhat, "x", ms=9, mew=2,
                           color="#ef704d", label="Refined estimate")
        ax[0, column].set_title(labels[name], fontsize=25)
        ax[0, column].set_xlabel("Angle θ (deg)", fontsize=20)
        ax[0, column].set_ylabel("Range R (m)", fontsize=20)
        ax[0, column].legend(loc="upper right", fontsize=15,
                             facecolor="white", framealpha=0.85)

        profile, _ = score_map(prepared, t, x, name,
                               ranges=RANGES, angles=np.array([that]))
        profile = profile[:, 0]
        profile_db = 10*np.log10(profile/profile.max())
        refined_power = continuous_score(prepared, t, x, name,
                                         rhat, that, tau_hat)
        refined_db = 10*np.log10(refined_power/profile.max())
        ax[1, column].plot(RANGES, profile_db, color="#2874bc", lw=2)
        ax[1, column].axvline(R_TRUE, color="black", ls="--", lw=1.4,
                              label=f"True source: {R_TRUE:g} m, {THETA_TRUE_DEG:g}°")
        ax[1, column].plot(rhat, refined_db, "o",
                           color="#ef704d", ms=6,
                           label=f"Refined: {rhat:.2f} m, {that:.2f}°")
        ax[1, column].set_title(
            f"{name} range profile at peak θ={that:.0f}°\n"
            f"Refined 2-D error: {error_2d:.2f} m", fontsize=25,
        )
        ax[1, column].set_xlabel("Candidate range (m)", fontsize=20)
        ax[1, column].set_ylabel("Relative mag. (dB)", fontsize=20)
        ax[1, column].set_xlim(RANGES[0], RANGES[-1])
        ax[1, column].set_ylim(-0.11, 0.012)
        ax[1, column].grid(True, alpha=0.2)
        ax[1, column].legend(loc="lower left", fontsize=15)
        print(f"{name}: grid peak R={r_peak:g} m, theta={theta_peak:g} deg; "
              f"tau={tau_peak:.12g} s; refined R={rhat:.6f} m, "
              f"theta={that:.6f} deg, tau={tau_hat:.12g} s; "
              f"2-D error={error_2d:.4f} m; accepted={accepted}; "
              f"evaluations={nfev}")

    fig.colorbar(im, ax=ax[0, :], label="Magnitude normalized to each peak (dB)",
                 pad=0.015, shrink=0.83)
    fig.suptitle("Hyperbolic & Parabolic Radon Transforms Under Noise",
                 fontsize=25)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    print(f"Saved {OUTPUT}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
