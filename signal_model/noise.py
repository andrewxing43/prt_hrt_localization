import numpy as np
from config import SIM


def reference_signal_energy(rx: np.ndarray, reference_index: int | None = None) -> float:
    """
    Return the discrete-time pulse energy of one reference antenna.

    Parameters
    ----------
    rx : ndarray, shape (num_antennas, num_time_samples)
        Noiseless received complex-baseband signal.
    reference_index : int or None
        Reference antenna index. If None, use the center antenna.
    """
    rx = np.asarray(rx)
    if rx.ndim != 2:
        raise ValueError("rx must have shape (num_antennas, num_time_samples)")

    if reference_index is None:
        reference_index = rx.shape[0] // 2

    if not 0 <= reference_index < rx.shape[0]:
        raise IndexError("reference_index is out of range")

    return float(np.sum(np.abs(rx[reference_index]) ** 2))


def noise_variance_from_snr(rx: np.ndarray, snr_db: float,
                            reference_index: int | None = None) -> float:
    """
    Compute complex AWGN variance from pulse-energy SNR.

    SNR = E_ref / sigma_n^2
    """
    energy = reference_signal_energy(rx, reference_index)
    snr_linear = 10.0 ** (snr_db / 10.0)

    if snr_linear <= 0.0:
        raise ValueError("snr_db must correspond to a positive linear SNR")

    return energy / snr_linear


def add_awgn(rx: np.ndarray, snr_db: float = SIM.snr_db,
             rng: np.random.Generator | None = None,
             reference_index: int | None = None) -> tuple[np.ndarray, float]:
    """
    Add iid complex white Gaussian noise to all antenna channels.

    The same noise variance is used for every antenna. SNR is defined
    using the discrete-time pulse energy at the reference antenna.

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

    noise_variance = noise_variance_from_snr(rx, snr_db, reference_index)
    sigma_component = np.sqrt(noise_variance / 2.0)

    noise = sigma_component * (
        rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
    )

    return rx + noise, noise_variance


if __name__ == "__main__":
    test_rx = np.zeros((3, 100), dtype=np.complex128)
    test_rx[1, 45:55] = 1.0

    noisy_rx, noise_var = add_awgn(test_rx, snr_db=10.0)

    print(f"Reference signal energy: {reference_signal_energy(test_rx):.6f}")
    print(f"Noise variance: {noise_var:.6f}")
    print(f"Noisy signal shape: {noisy_rx.shape}")