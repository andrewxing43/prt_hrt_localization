import numpy as np
import cupy as cp
from scipy.fft import next_fast_len


from config import C, SYSTEM

def run_linear_rt_cupy(input_matrix, tx_xpos, ts, px, f0, cpo=True, chunk_size=32):
    """
    Reusable function that calculates linear radon transform on nx x nt input signal matrix.
    Following steps in https://wiki.seg.org/wiki/The_parabolic_Radon_transform but for LRT.
    But using cupy.
    """
    if cp is None:
        raise ImportError("cupy is required for run_linear_rt_cupy")

    # move data to gpu
    d_gpu = cp.asarray(input_matrix, dtype=cp.complex64)
    tx_gpu = cp.asarray(tx_xpos, dtype=cp.float32)
    p_gpu = cp.asarray(px, dtype=cp.float32)
    n_nodes = len(p_gpu)

    # max time shift for padded FFT axis
    tau_max = float(cp.max(cp.abs(p_gpu)).get()) * float(cp.max(cp.abs(tx_gpu)).get())

    # convert to samples and get number of FFT points
    nt_original = d_gpu.shape[1]
    guard_samples = int(np.ceil(tau_max / ts)) + 2
    N_fft = next_fast_len(nt_original + 2 * guard_samples)
    pad_left = guard_samples
    pad_right = N_fft - nt_original - pad_left

    d_gpu = cp.pad(d_gpu, ((0, 0), (pad_left, pad_right)))

    # FFT along time axis
    D_f = cp.fft.fft(d_gpu, axis=1)
    Nx, nt = D_f.shape

    # for fourier shift later
    freqs = cp.fft.fftfreq(nt, d=ts).astype(cp.float32)

    tau_matrix = cp.outer(tx_gpu, p_gpu)
    carrier_corr = cp.exp(1j * 2 * cp.pi * f0 * tau_matrix) if cpo else cp.ones_like(tau_matrix, dtype=cp.complex64)
    antenna_weight_matrix = carrier_corr / Nx

    U_f_flat = cp.zeros((nt, n_nodes), dtype=cp.complex64)
    for start_idx in range(0, nt, chunk_size):
        end_idx = min(start_idx + chunk_size, nt)
        freqs_chunk = freqs[start_idx:end_idx]

        # fourier shift theorem to turn time domain delay into mult by exp in freq domain
        phase = 1j * 2 * cp.pi * freqs_chunk[:, None, None] * tau_matrix[None, :, :]
        total_corr_chunk = cp.exp(phase) * antenna_weight_matrix[None, :, :]
        D_f_chunk = D_f[:, start_idx:end_idx].T
        U_f_flat[start_idx:end_idx, :] = cp.einsum('cxn,cx->cn', total_corr_chunk, D_f_chunk)

    U_t_gpu = cp.fft.ifft(U_f_flat, axis=0)

    # chop off the padding
    U_t_gpu = U_t_gpu[pad_left:pad_left + nt_original, :]

    return U_t_gpu.get()
