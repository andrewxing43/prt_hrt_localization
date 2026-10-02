import numpy as np
import cupy as cp
from scipy.fft import next_fast_len

def run_parabolic_radon_transform(input_matrix, tx_xpos, ts, f0, px, qx, cpo):
    """
    Reusable function that calculates parabolic radon transform on nt x npx input signal matrix.
    Following steps in https://wiki.seg.org/wiki/The_parabolic_Radon_transform.
    Also used "Inverse velocity stacking for multiple elimination", Dan Hampson (1986).
    """
    if cp is None:
        raise ImportError("cupy is required for run_parabolic_radon_transform")

    # move data to GPU
    d_gpu = cp.asarray(input_matrix)
    tx_gpu = cp.asarray(tx_xpos)
    px_gpu = cp.asarray(px)
    qx_gpu = cp.asarray(qx)

    # p/q matrix of all combinations instead of p/q loops
    P, Q = cp.meshgrid(px_gpu, qx_gpu, indexing='ij')
    P_flat = P.ravel()
    Q_flat = Q.ravel()
    n_spatial = len(P_flat)

    # tau matrix (Nx x n_spatial) with each entry as tau = -px - qx^2
    tau_matrix = -cp.outer(tx_gpu, P_flat) - cp.outer(tx_gpu**2, Q_flat)

    # pad for circular
    nt_original = d_gpu.shape[1]

    max_delay = float(cp.max(cp.abs(tau_matrix)).get())
    guard_samples = int(np.ceil(max_delay / ts)) + 2

    nfft = next_fast_len(nt_original + 2 * guard_samples)
    pad_left = guard_samples
    pad_right = nfft - nt_original - pad_left

    d_gpu = cp.pad(d_gpu, ((0, 0), (pad_left, pad_right)))

    # cupy FFT along time axis
    D_f = cp.fft.fft(d_gpu, axis=1)
    Nx, nt = D_f.shape

    # for fourier shift later
    freqs = cp.fft.fftfreq(nt, d=ts)

    # carrier freq offset pre-calculated using above
    if cpo:
        carrier_corr = cp.exp(1j * 2 * cp.pi * f0 * tau_matrix)
    else:
        carrier_corr = cp.ones_like(tau_matrix, dtype=cp.complex64)

    # pre-divide by Nx to avoid division at every iteration of loop
    antenna_weight_matrix = carrier_corr / Nx

    # pre-allocate output on GPU
    U_f_flat = cp.zeros((nt, n_spatial), dtype=cp.complex64)

    # compute RT freq slice by freq slice
    for it in range(nt):
        # time_corr: (Nx, n_spatial)
        time_corr = cp.exp(1j * 2 * cp.pi * freqs[it] * tau_matrix)

        # combine time shift and CPO
        total_corr = time_corr * antenna_weight_matrix

        # matrix multiplication -> sum over the antennas for all slownesses and hardware done on GPU very quickly
        U_f_flat[it, :] = total_corr.T @ D_f[:, it]
    
    # reshape and IFFT along freq axis get back to time domain
    U_f = U_f_flat.reshape(nt, len(px), len(qx))
    U_t_gpu = cp.fft.ifft(U_f, axis=0)

    # strip away the padding added for FFT
    U_t_gpu = U_t_gpu[pad_left:pad_left + nt_original, :, :]

    # get result on RAM
    U_t_cpu = U_t_gpu.get()

    return np.transpose(U_t_cpu, (0, 1, 2))