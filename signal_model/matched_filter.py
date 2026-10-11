import numpy as np
from config import SYSTEM
from signal_model.gaussian_source import gaussian_pulse


def matched_filter(rx: np.ndarray) -> np.ndarray:
    """Apply the Gaussian pulse matched filter independently to each antenna."""
    rx = np.asarray(rx, dtype=np.complex128)

    half_span = int(np.ceil(6.0 * SYSTEM.gaussian_sigma / SYSTEM.dt))
    t_kernel = np.arange(-half_span, half_span + 1) * SYSTEM.dt
    kernel = gaussian_pulse(-t_kernel)
    kernel = kernel / np.sqrt(np.sum(np.abs(kernel) ** 2))

    return np.array([np.convolve(channel, kernel, mode="same") for channel in rx])


if __name__ == "__main__":
    from signal_model.propagation import received_signal
    from signal_model.noise import add_awgn

    range_m, theta_deg = 300.0, -18.0
    center_delay = range_m / 299_792_458.0
    margin = 20 * SYSTEM.gaussian_sigma
    t = np.arange(center_delay - margin, center_delay + margin + SYSTEM.dt, SYSTEM.dt)

    rx, _ = received_signal(t, range_m, theta_deg)
    noisy_rx, _ = add_awgn(rx)
    mf_rx = matched_filter(noisy_rx)

    print(f"Input shape: {noisy_rx.shape}")
    print(f"Matched-filter output shape: {mf_rx.shape}")