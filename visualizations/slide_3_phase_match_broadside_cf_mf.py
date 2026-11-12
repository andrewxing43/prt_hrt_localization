"""Single-source narrowband phase matching with a normalized time sum.

Place in experiments/ of the localization repository and run as a script.
Uses the same noisy record, search grid, and bounded Newton refinement as
the original phase comparison, and saves one range-score figure.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
root = here if (here / 'config.py').is_file() else here.parent
sys.path.insert(0, str(root))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal


# Illustrative single source and a fixed noise realization.
R_TRUE, THETA_TRUE = 300.0, 50.0
NOISE_SEED = int(getattr(SIM, 'rng_seed', 0)) + 1

R_MIN, R_MAX, DR = 5.0, 400.0, 2.0
TH_MIN, TH_MAX, DTH = -60.0, 60.0, 0.5
CHUNK = 1024
DISPLAY_RANGE_STEP_M = 0.1
NEWTON_HESSIAN_STEPS = np.array([0.01/DR, 1e-3])
SHOW_PLOT = True
OUTPUT = here / 'slide_3_phase_matching_single_source.png'


def phase_maps(x, f0, ranges, angles_deg, data):
    """Normalized spherical carrier steering for the antenna values."""
    rr, th = np.meshgrid(ranges, np.deg2rad(angles_deg), indexing='ij')
    rs, angles = rr.ravel(), th.ravel()
    scores = np.empty((len(rs), data.shape[1]))
    xx = x[:, None]
    for start in range(0, len(rs), CHUNK):
        stop = min(start + CHUNK, len(rs))
        r = rs[None, start:stop]
        a = angles[None, start:stop]
        dist = np.sqrt(r*r + xx*xx - 2*r*xx*np.sin(a))
        delta = (xx*xx - 2*r*xx*np.sin(a)) / (dist + r)
        steering = np.exp(-2j*np.pi*f0*delta/C) / np.sqrt(len(x))
        scores[start:stop] = np.abs(steering.conj().T @ data)**2
    return scores.reshape(len(ranges), len(angles_deg), -1)


def bounded_damped_newton(evaluate, bounds, maxiter=25):
    """Same bounded damped Newton search as the phase-matching experiment."""
    lower, upper = np.asarray(bounds, float).T
    u = np.zeros(2)
    initial_score = evaluate(u)[0]
    scale = max(initial_score, 1e-30)
    radius = 1.0
    accepted_steps = 0
    for _ in range(maxiter):
        score, raw_gradient = evaluate(u)
        gradient = raw_gradient/scale
        hessian = np.empty((2, 2))
        for axis, h in enumerate(NEWTON_HESSIAN_STEPS):
            shift = np.zeros(2)
            shift[axis] = h
            plus = evaluate(u+shift)[1]
            minus = evaluate(u-shift)[1]
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
            break
        direction = np.linalg.solve(negative_hessian+damping*metric, gradient)
        direction *= min(1.0, radius/max(np.max(np.abs(direction)), 1e-100))
        candidate = np.clip(u+direction, lower, upper)
        displacement = candidate-u
        if np.max(np.abs(displacement)) < 1e-8:
            break
        candidate_score, candidate_gradient = evaluate(candidate)
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
                break
        else:
            radius *= 0.5
            if radius < 1e-8:
                break
    final_score = evaluate(u)[0]
    return u if accepted_steps and final_score >= initial_score*(
        1-5*np.finfo(float).eps) else np.zeros(2)


def refine_phase(x, f0, z, r0, theta0_deg):
    """Optimize exactly the same phase score in continuous (R, theta)."""
    data = z[:, None]
    energy = float(np.vdot(data, data).real)

    def evaluate(u):
        r = r0+u[0]*DR
        theta = np.deg2rad(theta0_deg+u[1]*DTH)
        q = x*x-2*r*x*np.sin(theta)
        dist = np.sqrt(r*r+q)
        delta = q/(dist+r)
        delay = delta/C
        d_delay_dr = -(x*x+x*np.sin(theta)*delta)/(C*dist*(dist+r))
        d_delay_dtheta = -r*x*np.cos(theta)/(C*dist)
        a_conj = np.exp(2j*np.pi*f0*delay)/np.sqrt(len(x))
        y = a_conj @ data
        dy_dr = (a_conj*(2j*np.pi*f0*d_delay_dr*DR)) @ data
        dy_dtheta = (a_conj*(2j*np.pi*f0*d_delay_dtheta*np.deg2rad(DTH))) @ data
        score = float(np.vdot(y, y).real/energy)
        gradient = np.array([
            2*np.vdot(y, dy_dr).real/energy,
            2*np.vdot(y, dy_dtheta).real/energy,
        ])
        return score, gradient

    bounds = [((R_MIN-r0)/DR, (R_MAX-r0)/DR),
              ((TH_MIN-theta0_deg)/DTH, (TH_MAX-theta0_deg)/DTH)]
    u = bounded_damped_newton(evaluate, bounds)
    return r0+u[0]*DR, theta0_deg+u[1]*DTH


def main():
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    delays = propagation_delays(R_TRUE, THETA_TRUE, antenna_x=x)
    margin = 8*SYSTEM.gaussian_sigma
    n0 = int(np.floor((delays.min()-margin)/SYSTEM.dt))
    n1 = int(np.ceil((delays.max()+margin)/SYSTEM.dt))
    t = np.arange(n0, n1+1)*SYSTEM.dt
    clean, _ = received_signal(t, R_TRUE, THETA_TRUE, antenna_x=x)

    reference_range = float(getattr(SIM, 'reference_range', 300.0))
    reference_snr_db = float(getattr(SIM, 'reference_snr_db', 10.0))
    center = int(np.argmin(np.abs(x)))
    reference_delays = propagation_delays(reference_range, 0.0, antenna_x=x)
    reference, _ = received_signal(
        np.array([reference_delays[center]]), reference_range, 0.0,
        antenna_x=x,
    )
    variance = float(abs(reference[center, 0])**2/10**(reference_snr_db/10))
    rng = np.random.default_rng(NOISE_SEED)
    noisy = clean + np.sqrt(variance/2)*(
        rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape)
    )

    # Exactly the one_value/no-MF extraction: one normalized time sum per antenna.
    z_raw = noisy.sum(axis=1)/np.sqrt(len(t))

    ranges = np.arange(R_MIN, R_MAX+DR/2, DR)
    angles_deg = np.arange(TH_MIN, TH_MAX+DTH/2, DTH)
    scores = phase_maps(
        x, float(SYSTEM.fc), ranges, angles_deg, z_raw[:, None]
    )[:, :, 0]
    ir, ia = np.unravel_index(np.argmax(scores), scores.shape)
    rhat, that = ranges[ir], angles_deg[ia]
    r_refined, th_refined = refine_phase(
        x, float(SYSTEM.fc), z_raw, rhat, that
    )

    # A 0.1 m slice at the refined angle draws a smooth curve; the estimator
    # is the full 2-D grid followed by the same bounded Newton refinement.
    display_ranges = np.arange(180.0, 400.0+DISPLAY_RANGE_STEP_M/2,
                               DISPLAY_RANGE_STEP_M)
    display_score = phase_maps(
        x, float(SYSTEM.fc), display_ranges, np.array([th_refined]),
        z_raw[:, None]
    )[:, 0, 0]
    slice_db = 10*np.log10(np.maximum(
        display_score/max(display_score.max(), 1e-300), 1e-300))
    error = np.hypot(
        r_refined*np.sin(np.deg2rad(th_refined))-
        R_TRUE*np.sin(np.deg2rad(THETA_TRUE)),
        r_refined*np.cos(np.deg2rad(th_refined))-
        R_TRUE*np.cos(np.deg2rad(THETA_TRUE)),
    )

    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    ax.plot(display_ranges, slice_db, lw=2.4, color='#2b6eaa')
    ax.axvline(R_TRUE, lw=1.7, color='black', ls='--',
               label=f'True source: {R_TRUE:g} m, {THETA_TRUE:g}°')
    ax.scatter([r_refined], [np.interp(r_refined, display_ranges, slice_db)],
               marker='o', s=105, facecolors='#e86140', edgecolors='white',
               linewidths=1.2,
               label=f'Refined: {r_refined:.2f} m, {th_refined:.2f}°', zorder=6)
    ax.set_title(f'Narrowband Phase Matching Under Noise\n'
                 f'Refined 2-D error: {error:.1f} m', fontsize=20)
    ax.set_xlabel('Candidate range (m)', fontsize=15)
    ax.set_ylabel('Phase correlation magnitude relative to maximum (dB)', fontsize=15) 
    ax.set_xlim(180, 400)
    ax.set_ylim(-0.1, .006)
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', fontsize=15)
    fig.savefig(OUTPUT, dpi=200)
    print(f'Normalized time sum: grid=({rhat:g} m, {that:g}°), '
          f'refined=({r_refined:.2f} m, {th_refined:.2f}°), '
          f'2-D refined error={error:.3f} m')
    print(f'Saved {OUTPUT}; N={len(x)}, Nt={len(t)}, seed={NOISE_SEED}')
    if SHOW_PLOT:
        plt.show()


if __name__ == '__main__':
    main()
