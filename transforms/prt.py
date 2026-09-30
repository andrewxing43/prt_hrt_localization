import numpy as np
from scipy.fft import next_fast_len
from config import SYSTEM


def parabolic_radon_transform(data: np.ndarray, antenna_x: np.ndarray,
                              p_grid: np.ndarray, q_grid: np.ndarray,
                              fs: float = SYSTEM.fs, fc: float = SYSTEM.fc,
                              carrier_phase_correction: bool = True) -> np.ndarray:
    """
    Compute the 3-D PRT P(tau, p, q).

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
    prt : ndarray, shape (Nt, Np, Nq)
        Complex PRT. The first axis corresponds exactly to the input
        time-sample axis.
    """
    data = np.asarray(data, dtype=np.complex128)
    antenna_x = np.asarray(antenna_x, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must have shape (num_antennas, num_time_samples)")
    if data.shape[0] != antenna_x.size:
        raise ValueError("antenna_x must match the antenna dimension of data")

    num_antennas, nt_original = data.shape

    # All (p, q) combinations.
    P, Q = np.meshgrid(p_grid, q_grid, indexing="ij")
    p_flat, q_flat = P.ravel(), Q.ravel()
    num_pq = p_flat.size

    # Relative propagation delay:
    # delta_tau(x) = p*x + q*x^2
    delay_offset = (
        np.outer(antenna_x, p_flat)
        + np.outer(antenna_x**2, q_flat)
    )

    # Zero-padding prevents circular wrap-around during Fourier shifting.
    dt = 1.0 / fs
    max_delay = float(np.max(np.abs(delay_offset)))
    guard_samples = int(np.ceil(max_delay / dt)) + 2
    nfft = next_fast_len(nt_original + 2 * guard_samples)

    pad_left = guard_samples
    pad_right = nfft - nt_original - pad_left
    padded = np.pad(data, ((0, 0), (pad_left, pad_right)))

    # FFT along time.
    data_f = np.fft.fft(padded, axis=1)
    freqs = np.fft.fftfreq(nfft, d=dt)

    # Received carrier phase contains:
    # exp(-j*2*pi*fc*(tau + delta_tau)).
    # Remove the spatial part exp(-j*2*pi*fc*delta_tau).
    if carrier_phase_correction:
        carrier_corr = np.exp(1j * 2.0 * np.pi * fc * delay_offset)
    else:
        carrier_corr = np.ones_like(delay_offset, dtype=np.complex128)

    weights = carrier_corr / num_antennas
    prt_f = np.empty((nfft, num_pq), dtype=np.complex128)

    # r(t + delta_tau) aligns each antenna trajectory back to tau.
    # Fourier-domain time advance:
    # r(t + d) <-> R(f) exp(+j*2*pi*f*d)
    for k, freq in enumerate(freqs):
        time_corr = np.exp(1j * 2.0 * np.pi * freq * delay_offset)
        prt_f[k] = (time_corr * weights).T @ data_f[:, k]

    # Return to tau/time domain.
    prt = np.fft.ifft(prt_f, axis=0)
    prt = prt[pad_left:pad_left + nt_original]
    return prt.reshape(nt_original, len(p_grid), len(q_grid))


def prt_peak(prt: np.ndarray, time_axis: np.ndarray,
             p_grid: np.ndarray, q_grid: np.ndarray) -> tuple[float, float, float, float, tuple[int, int, int]]:
    """Return the global 3-D PRT peak."""
    magnitude = np.abs(prt)
    idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    tau_hat = float(time_axis[idx[0]])
    p_hat = float(p_grid[idx[1]])
    q_hat = float(q_grid[idx[2]])
    peak_value = float(magnitude[idx])

    return tau_hat, p_hat, q_hat, peak_value, idx


if __name__ == "__main__":
    print("PRT module loaded successfully.")