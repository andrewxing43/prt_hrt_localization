"""Peak-only FFT HRT within a PRT-derived (range, angle, time) ROI.

Uses the same fused equal-weight alignment as fast_prt_peak and 4x FFT
interpolation in time. The output is a peak, not a materialized HRT cube.
"""

import numpy as np
import cupy as cp
from scipy.fft import next_fast_len

from config import C
from transforms.fast_prt_peak import _PRT_ALIGN_KERNEL


def run_hyperbolic_radon_scores(data, time_axis, antenna_x, ranges,
                                angles_deg, tau_bounds, f0, time_factor=4,
                                tile_size=128):
    """Peak power and arrival time at each paired (range, angle) candidate."""
    if time_factor < 1 or int(time_factor) != time_factor:
        raise ValueError("time_factor must be a positive integer")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    time_axis = np.asarray(time_axis, dtype=float)
    antenna_x = np.asarray(antenna_x, dtype=float)
    ranges = np.asarray(ranges, dtype=float).ravel()
    angles = np.asarray(angles_deg, dtype=float).ravel()
    if ranges.shape != angles.shape or not len(ranges):
        raise ValueError("ranges and angles_deg must have the same nonzero length")
    nt = data.shape[1]
    dt = float(time_axis[1] - time_axis[0])
    guard = int(np.ceil(np.max(np.abs(antenna_x)) / (C * dt))) + 4
    nfft = next_fast_len(nt + 2 * guard)
    padded = cp.pad(cp.asarray(data), ((0, 0), (guard, nfft - nt - guard)))
    spectrum = cp.ascontiguousarray(cp.fft.fft(padded, axis=1),
                                    dtype=cp.complex128)
    freq = cp.ascontiguousarray(cp.fft.fftfreq(nfft, d=dt), dtype=cp.float64)
    x = cp.asarray(antenna_x, dtype=cp.float64)[None, :]
    factor = int(time_factor)

    # The FFT-interpolated trace is anchored at time_axis[0], as in the
    # HRT-only experiment. Search only time samples inside the PRT ROI.
    low = max(0, int(np.ceil((tau_bounds[0] - time_axis[0]) / dt * factor - 1e-8)))
    high = min((nt - 1) * factor,
               int(np.floor((tau_bounds[1] - time_axis[0]) / dt * factor + 1e-8)))
    if low > high:
        raise ValueError("The ROI has no HRT time sample at this interpolation")

    scores = np.empty(len(ranges), dtype=float)
    time_indices = np.empty(len(ranges), dtype=np.int64)
    for start in range(0, len(ranges), tile_size):
        stop = min(start + tile_size, len(ranges))
        r = cp.asarray(ranges[start:stop], dtype=cp.float64)[:, None]
        angle = cp.deg2rad(cp.asarray(angles[start:stop], dtype=cp.float64)[:, None])
        dist = cp.sqrt(r * r + x * x - 2 * r * x * cp.sin(angle))
        delay = cp.ascontiguousarray((dist - r) / C, dtype=cp.float64)
        aligned = cp.empty((stop - start, nfft), dtype=cp.complex64)
        count = (stop - start) * nfft
        _PRT_ALIGN_KERNEL(
            ((count + 255) // 256,), (256,),
            (spectrum, freq, delay, np.float64(f0),
             np.int32(len(antenna_x)), np.int32(nfft),
             np.int32(stop - start), aligned),
        )

        if factor == 1:
            trace = cp.fft.ifft(aligned, axis=1)
        else:
            npos = (nfft + 1) // 2
            padded_fft = cp.zeros((stop - start, factor * nfft),
                                  dtype=cp.complex64)
            padded_fft[:, :npos] = aligned[:, :npos]
            padded_fft[:, -(nfft - npos):] = aligned[:, npos:]
            trace = cp.fft.ifft(padded_fft, axis=1) * factor

        power = cp.abs(trace[:, guard * factor + low:
                             guard * factor + high + 1]) ** 2
        tile_time = cp.argmax(power, axis=1)
        tile_score = power[cp.arange(stop - start), tile_time]
        scores[start:stop] = cp.asnumpy(tile_score)
        time_indices[start:stop] = low + cp.asnumpy(tile_time)

    return scores, time_axis[0] + time_indices * dt / factor


def run_hyperbolic_radon_peak(data, time_axis, antenna_x, range_grid,
                              theta_grid, tau_bounds, f0, time_factor=4,
                              tile_size=128):
    """Return (R, theta_deg, tau, peak magnitude) within the given ROI."""
    ranges, angles = np.meshgrid(range_grid, theta_grid, indexing="ij")
    ranges, angles = ranges.ravel(), angles.ravel()
    scores, times = run_hyperbolic_radon_scores(
        data, time_axis, antenna_x, ranges, angles, tau_bounds, f0,
        time_factor=time_factor, tile_size=tile_size,
    )
    best = int(np.argmax(scores))
    return (float(ranges[best]), float(angles[best]),
            float(times[best]), float(np.sqrt(scores[best])))
