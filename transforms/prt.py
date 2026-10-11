"""
GPU-accelerated Parabolic Radon Transform using CuPy.

This module preserves the original public API:
    parabolic_radon_transform(...)
    prt_peak(...)

The PRT calculation runs on an NVIDIA GPU, while the returned PRT array
is converted back to numpy.ndarray for compatibility with existing code.
"""

import numpy as np
import cupy as cp
from scipy.fft import next_fast_len

from config import SYSTEM


def parabolic_radon_transform(
    data: np.ndarray,
    antenna_x: np.ndarray,
    p_grid: np.ndarray,
    q_grid: np.ndarray,
    fs: float = SYSTEM.fs,
    fc: float = SYSTEM.fc,
    carrier_phase_correction: bool = True,
) -> np.ndarray:
    """
    Compute the GPU-accelerated 3-D PRT P(tau, p, q).

    Candidate propagation trajectory:
        t(x) = tau + p*x + q*x^2

    Parameters
    ----------
    data : ndarray, shape (M, Nt)
        Complex-baseband antenna/time data.
    antenna_x : ndarray, shape (M,)
        Antenna positions [m].
    p_grid : ndarray
        Slowness grid [s/m].
    q_grid : ndarray
        Curvature grid [s/m^2].
    fs : float
        Sampling rate [Hz].
    fc : float
        Carrier frequency [Hz].
    carrier_phase_correction : bool
        Compensate spatial carrier phase before coherent summation.

    Returns
    -------
    prt : numpy.ndarray, shape (Nt, Np, Nq)
        Complex PRT returned on the CPU for compatibility with the
        existing localization and DBSCAN code.
    """
    data = np.asarray(data, dtype=np.complex128)
    antenna_x = np.asarray(antenna_x, dtype=np.float64)
    p_grid = np.asarray(p_grid, dtype=np.float64)
    q_grid = np.asarray(q_grid, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("data must have shape (num_antennas, num_time_samples)")
    if antenna_x.ndim != 1 or data.shape[0] != antenna_x.size:
        raise ValueError("antenna_x must match the antenna dimension of data")
    if p_grid.ndim != 1 or p_grid.size == 0:
        raise ValueError("p_grid must be a non-empty 1-D array")
    if q_grid.ndim != 1 or q_grid.size == 0:
        raise ValueError("q_grid must be a non-empty 1-D array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and > 0")
    if not np.isfinite(fc):
        raise ValueError("fc must be finite")
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA-capable GPU was detected by CuPy.")

    num_antennas, nt_original = data.shape
    num_p, num_q = p_grid.size, q_grid.size

    data_gpu = cp.asarray(data)
    antenna_x_gpu = cp.asarray(antenna_x)
    p_grid_gpu = cp.asarray(p_grid)
    q_grid_gpu = cp.asarray(q_grid)

    P, Q = cp.meshgrid(p_grid_gpu, q_grid_gpu, indexing="ij")
    p_flat, q_flat = P.ravel(), Q.ravel()
    num_pq = int(p_flat.size)

    delay_offset = (
        cp.outer(antenna_x_gpu, p_flat)
        + cp.outer(antenna_x_gpu**2, q_flat)
    )

    dt = 1.0 / fs
    max_delay = float(cp.max(cp.abs(delay_offset)).item())
    guard_samples = int(np.ceil(max_delay / dt)) + 2
    nfft = next_fast_len(nt_original + 2 * guard_samples)

    pad_left = guard_samples
    pad_right = nfft - nt_original - pad_left
    padded = cp.pad(data_gpu, ((0, 0), (pad_left, pad_right)))

    data_f = cp.fft.fft(padded, axis=1)
    freqs = cp.fft.fftfreq(nfft, d=dt)

    if carrier_phase_correction:
        weights = cp.exp(
            1j * 2.0 * cp.pi * fc * delay_offset
        ) / num_antennas
    else:
        weights = cp.full(
            delay_offset.shape,
            1.0 / num_antennas,
            dtype=cp.complex128,
        )

    prt_f = cp.empty((nfft, num_pq), dtype=cp.complex128)

    for k in range(nfft):
        time_corr = cp.exp(
            1j * 2.0 * cp.pi * freqs[k] * delay_offset
        )
        time_corr *= weights
        prt_f[k] = time_corr.T @ data_f[:, k]

    prt_gpu = cp.fft.ifft(prt_f, axis=0)
    prt_gpu = prt_gpu[pad_left:pad_left + nt_original]
    prt_gpu = prt_gpu.reshape(nt_original, num_p, num_q)

    return cp.asnumpy(prt_gpu)


def prt_peak(
    prt: np.ndarray,
    time_axis: np.ndarray,
    p_grid: np.ndarray,
    q_grid: np.ndarray,
) -> tuple[float, float, float, float, tuple[int, int, int]]:
    """Return the global 3-D PRT peak."""
    prt = np.asarray(prt)
    time_axis = np.asarray(time_axis, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)

    expected_shape = (time_axis.size, p_grid.size, q_grid.size)
    if prt.shape != expected_shape:
        raise ValueError(
            f"prt shape must be {expected_shape}, got {prt.shape}"
        )

    magnitude = np.abs(prt)
    idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    tau_hat = float(time_axis[idx[0]])
    p_hat = float(p_grid[idx[1]])
    q_hat = float(q_grid[idx[2]])
    peak_value = float(magnitude[idx])

    return tau_hat, p_hat, q_hat, peak_value, idx


if __name__ == "__main__":
    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]

    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    print("GPU PRT module loaded successfully.")
    print(f"CUDA device: {device_name}")