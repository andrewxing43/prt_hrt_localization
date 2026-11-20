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
    
    from config import C  # fixed lowercase 'c' reference
    delay_matrix = (distance_matrix - R_flat[None, :]) / C

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
    spatial_weights = relative_amplitude / amplitude_norm

    scale_matrix = (2.0 * cp.pi) * delay_matrix
    
    U_f_flat = cp.zeros((nt, n_spatial), dtype=cp.complex64)

    for it in range(nt):
        f_total = f0 + freqs[it] if cpo else freqs[it]
        weighted_phase = cp.exp(1j * f_total * scale_matrix) * spatial_weights
        U_f_flat[it, :] = D_f[:, it] @ weighted_phase

    U_f = U_f_flat.reshape(nt, len(range_vals), len(theta_vals))
    U_t_gpu = cp.fft.ifft(U_f, axis=0)
    U_t_gpu = U_t_gpu[pad_left:pad_left + nt_original, :, :]

    return U_t_gpu.get()

def resample_hrt(input_hrt, K):
    """
    Upsample HRT (input_hrt) by K times
    Basically implement scipy resample but by hand
    """
    N = input_hrt.shape[0]
    N_upsampled = N * K
    
    # FFT along time axis
    X_f = np.fft.fft(input_hrt, axis=0)
    
    # Empty freq array of size post upsampling
    upsamp_shape = list(X_f.shape)
    upsamp_shape[0] = N_upsampled
    X_f_padded = np.zeros(upsamp_shape, dtype=np.complex128)
    
    # map the frequencies from old to new freq array
    if N % 2 == 0:
        Nyq = N // 2

        # copy positive frequencies
        X_f_padded[0:Nyq] = X_f[0:Nyq]

        # for even size, split the nyquist bin in half to neg and pos freq.
        X_f_padded[Nyq] = X_f[Nyq] / 2.0
        X_f_padded[-Nyq] = X_f[Nyq] / 2.0

        # copy negative frequencies
        X_f_padded[-Nyq + 1:] = X_f[Nyq + 1:]
    else:
        Nyq = (N - 1) // 2

        # copy positive frequencies
        X_f_padded[0:Nyq + 1] = X_f[0:Nyq + 1]

        # copy negative frequencies
        X_f_padded[-Nyq:] = X_f[-Nyq:]
        
    # take FFT and scale amplitude by upsampling factor
    x_interpolated = np.fft.ifft(X_f_padded, axis=0) * K
    
    return x_interpolated