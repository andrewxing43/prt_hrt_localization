import numpy as np
from config import SYSTEM, SIM


def gaussian_pulse(t: np.ndarray, sigma: float = SYSTEM.gaussian_sigma) -> np.ndarray:
    """Unit-amplitude complex-baseband Gaussian pulse centered at t=0."""
    t = np.asarray(t, dtype=float)
    return np.exp(-0.5 * (t / sigma) ** 2).astype(np.complex128)


def pulse_time_axis(sigma: float = SYSTEM.gaussian_sigma,
                    fs: float = SYSTEM.fs,
                    span_sigma: float = SIM.pulse_span_sigma) -> np.ndarray:
    """Time samples covering approximately [-span_sigma*sigma, +span_sigma*sigma]."""
    half_span = span_sigma * sigma
    n_half = int(np.ceil(half_span * fs))
    return np.arange(-n_half, n_half + 1) / fs


def reference_pulse(sigma: float = SYSTEM.gaussian_sigma,
                    fs: float = SYSTEM.fs,
                    span_sigma: float = SIM.pulse_span_sigma) -> tuple[np.ndarray, np.ndarray]:
    """Return the sampled reference Gaussian pulse and its time axis."""
    t = pulse_time_axis(sigma, fs, span_sigma)
    return t, gaussian_pulse(t, sigma)


if __name__ == "__main__":
    t, pulse = reference_pulse()
    print(f"Pulse samples: {len(pulse)}")
    print(f"Sampling rate: {SYSTEM.fs / 1e9:.1f} GHz")
    print(f"Gaussian sigma: {SYSTEM.gaussian_sigma * 1e12:.2f} ps")
    print(f"Pulse span: {t[0] * 1e12:.2f} to {t[-1] * 1e12:.2f} ps")