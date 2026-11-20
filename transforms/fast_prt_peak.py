import numpy as np
import cupy as cp
from scipy.fft import next_fast_len


def run_parabolic_radon_peak(input_matrix, tx_xpos, ts, f0, px, qx, cpo, candidate_batch=256, tau_indices=None):
    x = np.asarray(tx_xpos, dtype=float)
    p = np.asarray(px, dtype=float)
    q = np.asarray(qx, dtype=float)
    nt_original = input_matrix.shape[1]
    times = np.arange(nt_original) if tau_indices is None else np.unique(np.asarray(tau_indices, dtype=int))
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
    D_f = cp.fft.fft(d_padded, axis=1)
    tx_gpu = cp.asarray(x)
    px_gpu = cp.asarray(p)
    qx_gpu = cp.asarray(q)
    freqs = cp.fft.fftfreq(nfft, d=ts)
    times_gpu = cp.asarray(times)
    inv_Nx = cp.float32(1.0 / len(x))
    n_spatial = len(p) * len(q)
    best_power = -np.inf
    best_flat_index = nt_original * n_spatial
    best = None

    # batch spatial candidates so GPU memory does not scale with the complete p/q grid.
    for start in range(0, n_spatial, int(candidate_batch)):
        stop = min(start + int(candidate_batch), n_spatial)
        spatial_idx = cp.arange(start, stop)
        P_flat = px_gpu[spatial_idx // len(q)]
        Q_flat = qx_gpu[spatial_idx % len(q)]
        tau_matrix = cp.outer(tx_gpu, P_flat) + cp.outer(tx_gpu**2, Q_flat)
        scale_matrix = 2.0 * cp.pi * tau_matrix
        U_f = cp.zeros((nfft, stop - start), dtype=cp.complex64)
        for it in range(nfft):
            f_total = f0 + freqs[it] if cpo else freqs[it]
            total_corr = cp.exp(1j * f_total * scale_matrix)
            U_f[it, :] = (D_f[:, it] @ total_corr) * inv_Nx
        U_t = cp.fft.ifft(U_f, axis=0)
        selected = U_t[pad_left + times_gpu, :]
        power = cp.abs(selected)**2
        local_flat_index = int(cp.argmax(power).item())
        time_local, spatial_local = divmod(local_flat_index, stop - start)
        peak_power = float(power[time_local, spatial_local].item())
        time_index = int(times[time_local])
        spatial_index = start + spatial_local
        flat_index = time_index * n_spatial + spatial_index
        if peak_power > best_power or (peak_power == best_power and flat_index < best_flat_index):
            best_power = peak_power
            best_flat_index = flat_index
            best = (time_index, spatial_index // len(q), spatial_index % len(q), peak_power)
    return best
