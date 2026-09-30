import numpy as np
from config import SYSTEM, SIM
from signal_model.gaussian_source import reference_pulse


def matched_filter(rx: np.ndarray, sigma: float = SYSTEM.gaussian_sigma,
                   fs: float = SYSTEM.fs,
                   span_sigma: float = SIM.pulse_span_sigma,
                   normalize: bool = True) -> np.ndarray:
    """
    Apply a Gaussian matched filter independently to every antenna channel.

    Parameters
    ----------
    rx : ndarray, shape (num_antennas, num_time_samples)
        Noisy complex-baseband received signal.
    normalize : bool
        Normalize the matched-filter template to unit energy.

    Returns
    -------
    mf_rx : ndarray
        Matched-filter output with the same shape as rx.
    """
    rx = np.asarray(rx, dtype=np.complex128)

    if rx.ndim != 2:
        raise ValueError("rx must have shape (num_antennas, num_time_samples)")

    _, pulse = reference_pulse(sigma, fs, span_sigma)

    if normalize:
        pulse = pulse / np.linalg.norm(pulse)

    h = np.conj(pulse[::-1])
    mf_rx = np.empty_like(rx)

    for m in range(rx.shape[0]):
        mf_rx[m] = np.convolve(rx[m], h, mode="same")

    return mf_rx


if __name__ == "__main__":
    from config import C
    from signal_model.propagation import received_signal
    from signal_model.noise import add_awgn

    range_m, theta_deg = 200.0, -18.0
    center_delay = range_m / C
    margin = 30 * SYSTEM.gaussian_sigma
    t = np.arange(center_delay - margin, center_delay + margin + SYSTEM.dt, SYSTEM.dt)

    rx, _ = received_signal(t, range_m, theta_deg)
    noisy_rx, _ = add_awgn(rx, snr_db=10.0)
    mf_rx = matched_filter(noisy_rx)

    center = rx.shape[0] // 2
    peak_index = np.argmax(np.abs(mf_rx[center]))

    print(f"Matched-filter output shape: {mf_rx.shape}")
    print(f"Center antenna peak time: {t[peak_index] * 1e9:.6f} ns")
    print(f"True center delay: {center_delay * 1e9:.6f} ns")