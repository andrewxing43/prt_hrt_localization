import numpy as np
from config import SIM


def thermal_noise_power() -> float:
    """Return receiver thermal noise power kTB."""
    return SIM.noise_power


def add_awgn(rx: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """
    Add iid complex white Gaussian noise to all antenna channels.

    The same physical thermal-noise power kTB is used as the variance
    of every complex baseband sample.

    Returns
    -------
    noisy_rx : ndarray
        Received signal plus complex AWGN.
    noise_variance : float
        E[|n|^2] for each complex noise sample.
    """
    rx = np.asarray(rx, dtype=np.complex128)

    if rng is None:
        rng = np.random.default_rng(SIM.rng_seed)

    noise_variance = thermal_noise_power()
    sigma_component = np.sqrt(noise_variance / 2.0)

    noise = sigma_component * (
        rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
    )

    return rx + noise, noise_variance


if __name__ == "__main__":
    test_rx = np.zeros((3, 100), dtype=np.complex128)
    noisy_rx, noise_var = add_awgn(test_rx)

    print(f"Thermal noise power: {noise_var:.6e} W")
    print(f"Thermal noise power: {10.0 * np.log10(noise_var / 1e-3):.3f} dBm")
    print(f"Noisy signal shape: {noisy_rx.shape}")