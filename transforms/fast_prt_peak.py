"""Peak-only batched PRT using the positive delay convention."""

import numpy as np
import cupy as cp
from scipy.fft import next_fast_len

_PRT_ALIGN_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void prt_align(const double2* spectrum, const double* freq,
               const double* delay, const double f0, const int n_ant,
               const int nfft, const int n_candidates, float2* output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates * nfft) return;
    const int k = idx / nfft;
    const int j = idx % nfft;
    const double omega = 6.2831853071795864769 * (f0 + freq[j]);
    double real_sum = 0.0, imag_sum = 0.0;
    for (int m = 0; m < n_ant; ++m) {
        const double phase = omega * delay[k * n_ant + m];
        double sine, cosine;
        sincos(phase, &sine, &cosine);
        const double2 z = spectrum[m * nfft + j];
        real_sum += z.x * cosine - z.y * sine;
        imag_sum += z.x * sine + z.y * cosine;
    }
    const double inverse_ant = 1.0 / n_ant;
    output[idx] = make_float2((float)(real_sum * inverse_ant),
                              (float)(imag_sum * inverse_ant));
}
''', 'prt_align')


def run_parabolic_radon_peak(input_matrix, tx_xpos, ts, f0, px, qx, cpo, candidate_batch=256, tau_indices=None, valid_spatial_mask=None, implementation="fused", time_factor=1):
    """Return the global peak; time index is on the ``ts / time_factor`` grid."""
    if implementation not in ("fused", "reference"):
        raise ValueError("implementation must be 'fused' or 'reference'")
    if not isinstance(time_factor, (int, np.integer)) or time_factor < 1:
        raise ValueError("time_factor must be a positive integer")
    x = np.asarray(tx_xpos, dtype=float)
    p = np.asarray(px, dtype=float)
    q = np.asarray(qx, dtype=float)
    nt_original = input_matrix.shape[1]
    # tau_indices, when provided, are indices on the interpolated time grid.
    times = (np.arange((nt_original - 1) * time_factor + 1)
             if tau_indices is None else np.unique(np.asarray(tau_indices, dtype=int)))
    if len(times) == 0 or times[0] < 0 or times[-1] > (nt_original - 1) * time_factor:
        raise ValueError("tau_indices must lie within the recorded time window")
    max_delay = 0.0
    for p_edge in (float(np.min(p)), float(np.max(p))):
        for q_edge in (float(np.min(q)), float(np.max(q))):
            max_delay = max(max_delay, float(np.max(np.abs(x * p_edge + x**2 * q_edge))))
    guard_samples = int(np.ceil(max_delay / ts)) + 2
    nfft = next_fast_len(nt_original + 2 * guard_samples)
    pad_left = guard_samples
    pad_right = nfft - nt_original - pad_left
    d_gpu = cp.asarray(input_matrix)
    d_padded = cp.pad(d_gpu, ((0, 0), (pad_left, pad_right)))
    D_f = cp.ascontiguousarray(cp.fft.fft(d_padded, axis=1), dtype=cp.complex128)
    tx_gpu = cp.asarray(x)
    px_gpu = cp.asarray(p)
    qx_gpu = cp.asarray(q)
    freqs = cp.ascontiguousarray(cp.fft.fftfreq(nfft, d=ts), dtype=cp.float64)
    times_gpu = cp.asarray(times)
    n_spatial = len(p) * len(q)
    if valid_spatial_mask is None:
        valid_spatial_mask = np.ones((len(p), len(q)), dtype=bool)
    valid_spatial_mask = np.asarray(valid_spatial_mask, dtype=bool)
    if valid_spatial_mask.shape != (len(p), len(q)) or not np.any(valid_spatial_mask):
        raise ValueError("valid_spatial_mask must have shape (Np, Nq) and contain at least one valid candidate")
    valid_flat = np.flatnonzero(valid_spatial_mask.ravel())
    valid_flat_gpu = cp.asarray(valid_flat, dtype=cp.int64)
    best_power = -np.inf
    best_flat_index = len(times) * n_spatial
    best = None

    for start in range(0, len(valid_flat), int(candidate_batch)):
        stop = min(start + int(candidate_batch), len(valid_flat))
        spatial_idx = valid_flat_gpu[start:stop]
        P_flat = px_gpu[spatial_idx // len(q)]
        Q_flat = qx_gpu[spatial_idx % len(q)]
        delay = cp.ascontiguousarray(
            (P_flat[:, None] * tx_gpu[None, :]
             + Q_flat[:, None] * tx_gpu[None, :] ** 2), dtype=cp.float64,
        )
        U_f = cp.empty((stop - start, nfft), dtype=cp.complex64)
        if implementation == "fused":
            count = (stop - start) * nfft
            _PRT_ALIGN_KERNEL(
                ((count + 255) // 256,), (256,),
                (D_f, freqs, delay, np.float64(f0 if cpo else 0.0),
                 np.int32(len(x)), np.int32(nfft), np.int32(stop - start), U_f),
            )
        else:
            for j in range(nfft):
                correction = cp.exp(2j * cp.pi *
                                    (f0 + freqs[j] if cpo else freqs[j]) * delay)
                U_f[:, j] = (D_f[:, j] @ correction.T) / len(x)
        if time_factor == 1:
            U_t = cp.fft.ifft(U_f, axis=1)
        else:
            # Fourier interpolation of each PRT output trace, after antenna
            # alignment. This is the same bin placement as the 4x HRT search.
            npos = (nfft + 1) // 2
            nneg = nfft - npos
            padded = cp.zeros((stop - start, time_factor * nfft),
                              dtype=U_f.dtype)
            padded[:, :npos] = U_f[:, :npos]
            if nneg:
                padded[:, -nneg:] = U_f[:, npos:]
            U_t = cp.fft.ifft(padded, axis=1) * time_factor
        selected = U_t[:, pad_left * time_factor + times_gpu].T
        power = cp.abs(selected) ** 2
        local_flat_index = int(cp.argmax(power).item())
        time_local, spatial_local = divmod(local_flat_index, stop - start)
        peak_power = float(power[time_local, spatial_local].item())
        time_index = int(times[time_local])
        spatial_index = int(valid_flat[start + spatial_local])
        flat_index = time_index * n_spatial + spatial_index
        if peak_power > best_power or (peak_power == best_power and flat_index < best_flat_index):
            best_power = peak_power
            best_flat_index = flat_index
            best = (time_index, spatial_index // len(q), spatial_index % len(q), peak_power)
    return best
