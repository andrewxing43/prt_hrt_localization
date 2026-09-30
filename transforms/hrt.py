import numpy as np
from config import C, SYSTEM


def relative_spherical_delays(range_m, theta_deg, antenna_x):
    """Exact spherical-wave delays relative to the array-center arrival time."""
    antenna_x = np.asarray(antenna_x, dtype=float)
    R = np.asarray(range_m, dtype=float)
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))

    if np.any(~np.isfinite(R)) or np.any(R <= 0):
        raise ValueError("range_m must be finite and > 0")
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta_deg must be finite")

    R, theta = np.broadcast_arrays(R, theta)
    R = R.ravel()
    theta = theta.ravel()

    x = antenna_x[:, None]
    d = np.sqrt(R[None, :]**2 + x**2 - 2.0 * R[None, :] * x * np.sin(theta)[None, :])

    return (d - R[None, :]) / C


def tau_from_clock_offset(range_m, clock_offset=0.0):
    """Reference arrival time: tau = R/c + user/device delay."""
    tau = np.asarray(range_m, dtype=float) / C + np.asarray(clock_offset, dtype=float)
    return float(tau) if tau.ndim == 0 else tau


def clock_offset_from_tau(tau, range_m):
    """Recover user/device delay: b = tau - R/c."""
    offset = np.asarray(tau, dtype=float) - np.asarray(range_m, dtype=float) / C
    return float(offset) if offset.ndim == 0 else offset


def hyperbolic_radon_transform(data, time_axis, antenna_x, range_grid, theta_grid,
                               tau_grid, fc=SYSTEM.fc, carrier_phase_correction=True,
                               batch_size=128):
    """
    Exact spherical-wave HRT.

    Output shape:
        (N_tau, N_range, N_theta)

    Candidate trajectory:
        d_m = sqrt(R^2 + x_m^2 - 2 R x_m sin(theta))
        t_m = tau + (d_m - R) / c

    tau is independent of R, so an unknown user/device delay is supported.
    """
    data = np.asarray(data, dtype=np.complex128)
    time_axis = np.asarray(time_axis, dtype=float)
    antenna_x = np.asarray(antenna_x, dtype=float)
    range_grid = np.asarray(range_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must have shape (num_antennas, num_time_samples)")
    if time_axis.ndim != 1 or time_axis.size != data.shape[1]:
        raise ValueError("time_axis must match the time dimension of data")
    if antenna_x.ndim != 1 or antenna_x.size != data.shape[0]:
        raise ValueError("antenna_x must match the antenna dimension of data")
    if range_grid.ndim != 1 or range_grid.size == 0 or np.any(range_grid <= 0):
        raise ValueError("range_grid must be a non-empty positive 1-D array")
    if theta_grid.ndim != 1 or theta_grid.size == 0:
        raise ValueError("theta_grid must be a non-empty 1-D array")
    if tau_grid.ndim != 1 or tau_grid.size == 0:
        raise ValueError("tau_grid must be a non-empty 1-D array")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    dt_all = np.diff(time_axis)
    dt = float(np.mean(dt_all))
    if dt <= 0 or not np.allclose(dt_all, dt, rtol=1e-8, atol=max(1e-18, abs(dt) * 1e-10)):
        raise ValueError("time_axis must be uniformly sampled")

    nr, nt_theta, nt_tau = len(range_grid), len(theta_grid), len(tau_grid)

    R, theta = np.meshgrid(range_grid, theta_grid, indexing="ij")
    R_flat = R.ravel()
    theta_flat = theta.ravel()

    n_candidates = len(R_flat)
    out = np.zeros((nt_tau, n_candidates), dtype=np.complex128)

    antenna_idx = np.arange(data.shape[0])[:, None, None]
    t0 = float(time_axis[0])
    nt = data.shape[1]

    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)

        rel_delay = relative_spherical_delays(
            R_flat[start:stop], theta_flat[start:stop], antenna_x
        )

        sample_pos = (
            tau_grid[None, None, :] + rel_delay[:, :, None] - t0
        ) / dt

        i0 = np.floor(sample_pos).astype(np.int64)
        frac = sample_pos - i0
        valid = (i0 >= 0) & (i0 < nt - 1)
        i0_safe = np.clip(i0, 0, nt - 2)

        s0 = data[antenna_idx, i0_safe]
        s1 = data[antenna_idx, i0_safe + 1]
        samples = ((1.0 - frac) * s0 + frac * s1) * valid

        if carrier_phase_correction:
            phase = np.exp(1j * 2.0 * np.pi * fc * rel_delay)[:, :, None]
            samples *= phase

        out[:, start:stop] = np.sum(samples, axis=0).T / data.shape[0]

    return out.reshape(nt_tau, nr, nt_theta)


def hrt_peak(hrt, tau_grid, range_grid, theta_grid):
    """Return (tau_hat, R_hat, theta_hat, peak_value, index)."""
    hrt = np.asarray(hrt)
    tau_grid = np.asarray(tau_grid, dtype=float)
    range_grid = np.asarray(range_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)

    expected = (len(tau_grid), len(range_grid), len(theta_grid))
    if hrt.shape != expected:
        raise ValueError(f"hrt shape must be {expected}, got {hrt.shape}")

    magnitude = np.abs(hrt)
    idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    tau_hat = float(tau_grid[idx[0]])
    R_hat = float(range_grid[idx[1]])
    theta_hat = float(theta_grid[idx[2]])
    peak = float(magnitude[idx])

    return tau_hat, R_hat, theta_hat, peak, idx


if __name__ == "__main__":
    print("HRT module loaded successfully.")