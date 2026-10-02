import numpy as np
import cupy as cp
from scipy.fft import next_fast_len
from scipy.constants import c  # Imported speed of light to resolve the undefined 'c' in delay_matrix

def run_hyperbolic_radon_transform(input_matrix, tx_xpos, ts, f0, range_vals, theta_vals, cpo):
    """
    Calculate the exact hyperbolic Radon transform over a local range-angle grid.
    """
    if cp is None:
        raise ImportError("cupy is required for run_hyperbolic_radon_transform")

    d_gpu = cp.asarray(input_matrix)
    tx_gpu = cp.asarray(tx_xpos)
    range_gpu = cp.asarray(range_vals)
    theta_gpu = cp.asarray(theta_vals)

    R, THETA = cp.meshgrid(range_gpu, theta_gpu, indexing="ij")
    R_flat = R.ravel()
    THETA_flat = THETA.ravel()
    n_spatial = len(R_flat)

    X0_flat = R_flat * cp.sin(THETA_flat)
    Z0_flat = R_flat * cp.cos(THETA_flat)
    distance_matrix = cp.sqrt((tx_gpu[:, None] - X0_flat[None, :])**2 + Z0_flat[None, :]**2)
    delay_matrix = (distance_matrix - R_flat[None, :]) / c

    nt_original = d_gpu.shape[1]
    max_delay = float(cp.max(cp.abs(delay_matrix)).get())
    guard_samples = int(np.ceil(max_delay / ts)) + 2
    nfft = next_fast_len(nt_original + 2 * guard_samples)
    pad_left = guard_samples
    pad_right = nfft - nt_original - pad_left

    d_gpu = cp.pad(d_gpu, ((0, 0), (pad_left, pad_right)))
    D_f = cp.fft.fft(d_gpu, axis=1)
    Nx, nt = D_f.shape
    freqs = cp.fft.fftfreq(nt, d=ts)

    relative_amplitude = R_flat[None, :] / distance_matrix
    amplitude_norm = cp.sqrt(cp.sum(relative_amplitude**2, axis=0, keepdims=True))

    if cpo:
        carrier_corr = cp.exp(1j * 2 * cp.pi * f0 * delay_matrix)
    else:
        carrier_corr = cp.ones_like(delay_matrix, dtype=cp.complex64)

    antenna_weight_matrix = relative_amplitude * carrier_corr / amplitude_norm
    U_f_flat = cp.zeros((nt, n_spatial), dtype=cp.complex64)

    for it in range(nt):
        time_corr = cp.exp(1j * 2 * cp.pi * freqs[it] * delay_matrix)
        total_corr = time_corr * antenna_weight_matrix
        U_f_flat[it, :] = total_corr.T @ D_f[:, it]

    U_f = U_f_flat.reshape(nt, len(range_vals), len(theta_vals))
    U_t_gpu = cp.fft.ifft(U_f, axis=0)
    U_t_gpu = U_t_gpu[pad_left:pad_left + nt_original, :, :]

    return U_t_gpu.get()