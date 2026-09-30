import numpy as np
from config import C, SYSTEM, ARRAY
from signal_model.gaussian_source import gaussian_pulse


def source_position(range_m: float, theta_deg: float) -> tuple[float, float]:
    """Source coordinates for a ULA along x; theta=0 is array broadside."""
    theta = np.deg2rad(theta_deg)
    return range_m * np.sin(theta), range_m * np.cos(theta)


def propagation_distances(range_m: float, theta_deg: float,
                          antenna_x: np.ndarray | None = None) -> np.ndarray:
    """Exact spherical-wave distance from the source to every antenna."""
    if antenna_x is None:
        antenna_x = ARRAY.positions(SYSTEM)

    x_src, z_src = source_position(range_m, theta_deg)
    return np.sqrt((x_src - antenna_x) ** 2 + z_src ** 2)


def propagation_delays(range_m: float, theta_deg: float, clock_offset: float = 0.0,
                       antenna_x: np.ndarray | None = None) -> np.ndarray:
    """Absolute propagation delay at every antenna."""
    distances = propagation_distances(range_m, theta_deg, antenna_x)
    return distances / C + clock_offset


def received_signal(t: np.ndarray, range_m: float, theta_deg: float,
                    clock_offset: float = 0.0, amplitude: complex = 1.0,
                    include_path_loss: bool = True,
                    antenna_x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate noiseless complex-baseband received signals for one source.

    Returns
    -------
    rx : ndarray, shape (num_antennas, num_time_samples)
        Received complex-baseband signal.
    delays : ndarray, shape (num_antennas,)
        Exact arrival time at each antenna.
    """
    if antenna_x is None:
        antenna_x = ARRAY.positions(SYSTEM)

    t = np.asarray(t, dtype=float)
    distances = propagation_distances(range_m, theta_deg, antenna_x)
    delays = distances / C + clock_offset

    envelope = gaussian_pulse(t[None, :] - delays[:, None])
    carrier_phase = np.exp(-1j * 2.0 * np.pi * SYSTEM.fc * delays)

    gain = amplitude / distances if include_path_loss else amplitude
    rx = gain[:, None] * carrier_phase[:, None] * envelope

    return rx, delays


if __name__ == "__main__":
    range_m, theta_deg = 200.0, -18.0
    center_delay = range_m / C
    margin = 20 * SYSTEM.gaussian_sigma
    t = np.arange(center_delay - margin, center_delay + margin + SYSTEM.dt, SYSTEM.dt)

    rx, delays = received_signal(t, range_m, theta_deg)

    print(f"rx shape: {rx.shape}")
    print(f"Center propagation delay: {center_delay * 1e9:.3f} ns")
    print(f"Delay span across array: {(delays.max() - delays.min()) * 1e12:.3f} ps")