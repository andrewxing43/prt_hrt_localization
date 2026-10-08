"""
GPU-accelerated Hyperbolic Radon Transform using CuPy.

Public API preserved:
    relative_spherical_delays(...)
    tau_from_clock_offset(...)
    clock_offset_from_tau(...)
    hyperbolic_radon_transform(...)
    hrt_peak(...)

The HRT calculation runs on an NVIDIA GPU. Results are converted back
to numpy.ndarray for compatibility with the existing project.
"""

import numpy as np
import cupy as cp

from config import C, SYSTEM


# ============================================================
# Public geometry helpers
# ============================================================

def relative_spherical_delays(range_m, theta_deg, antenna_x):
    """
    Exact spherical-wave delays relative to the array-center arrival time.

    This public helper remains NumPy-based to preserve its original behavior
    and return type.
    """
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
    distance = np.sqrt(
        R[None, :]**2
        + x**2
        - 2.0 * R[None, :] * x * np.sin(theta)[None, :]
    )

    return (distance - R[None, :]) / C


def tau_from_clock_offset(range_m, clock_offset=0.0):
    """Reference arrival time: tau = R/c + user/device delay."""
    tau = np.asarray(range_m, dtype=float) / C + np.asarray(
        clock_offset,
        dtype=float,
    )
    return float(tau) if tau.ndim == 0 else tau


def clock_offset_from_tau(tau, range_m):
    """Recover user/device delay: b = tau - R/c."""
    offset = np.asarray(tau, dtype=float) - np.asarray(
        range_m,
        dtype=float,
    ) / C
    return float(offset) if offset.ndim == 0 else offset


# ============================================================
# Internal GPU geometry helper
# ============================================================

def _relative_spherical_delays_gpu(
    range_m_gpu,
    theta_deg_gpu,
    antenna_x_gpu,
):
    """GPU version used internally by the HRT."""
    R = cp.asarray(range_m_gpu, dtype=cp.float64)
    theta = cp.deg2rad(cp.asarray(theta_deg_gpu, dtype=cp.float64))

    R, theta = cp.broadcast_arrays(R, theta)
    R = R.ravel()
    theta = theta.ravel()

    x = antenna_x_gpu[:, None]
    distance_squared = (
        R[None, :]**2
        + x**2
        - 2.0 * R[None, :] * x * cp.sin(theta)[None, :]
    )

    distance = cp.sqrt(cp.maximum(distance_squared, 0.0))
    return (distance - R[None, :]) / C


# ============================================================
# GPU HRT
# ============================================================

def hyperbolic_radon_transform(
    data,
    time_axis,
    antenna_x,
    range_grid,
    theta_grid,
    tau_grid,
    fc=SYSTEM.fc,
    carrier_phase_correction=True,
    batch_size=128,
):
    """
    GPU-accelerated exact spherical-wave HRT.

    Output shape:
        (N_tau, N_range, N_theta)

    Candidate trajectory:
        d_m = sqrt(R^2 + x_m^2 - 2 R x_m sin(theta))
        t_m = tau + (d_m - R) / c

    Parameters
    ----------
    data : ndarray, shape (num_antennas, num_time_samples)
        Complex matched-filtered array data.
    time_axis : ndarray
        Uniformly sampled time axis.
    antenna_x : ndarray
        Antenna positions [m].
    range_grid : ndarray
        Candidate ranges [m].
    theta_grid : ndarray
        Candidate angles [deg].
    tau_grid : ndarray
        Candidate reference arrival times [s].
    fc : float
        Carrier frequency [Hz].
    carrier_phase_correction : bool
        Apply carrier phase compensation when True.
    batch_size : int
        Number of (range, theta) candidates processed per GPU batch.

    Returns
    -------
    hrt : numpy.ndarray
        Complex HRT with shape (N_tau, N_range, N_theta).
    """
    data = np.asarray(data, dtype=np.complex128)
    time_axis = np.asarray(time_axis, dtype=float)
    antenna_x = np.asarray(antenna_x, dtype=float)
    range_grid = np.asarray(range_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            "data must have shape (num_antennas, num_time_samples)"
        )
    if time_axis.ndim != 1 or time_axis.size != data.shape[1]:
        raise ValueError(
            "time_axis must match the time dimension of data"
        )
    if antenna_x.ndim != 1 or antenna_x.size != data.shape[0]:
        raise ValueError(
            "antenna_x must match the antenna dimension of data"
        )
    if (
        range_grid.ndim != 1
        or range_grid.size == 0
        or np.any(~np.isfinite(range_grid))
        or np.any(range_grid <= 0)
    ):
        raise ValueError(
            "range_grid must be a non-empty positive finite 1-D array"
        )
    if (
        theta_grid.ndim != 1
        or theta_grid.size == 0
        or np.any(~np.isfinite(theta_grid))
    ):
        raise ValueError(
            "theta_grid must be a non-empty finite 1-D array"
        )
    if (
        tau_grid.ndim != 1
        or tau_grid.size == 0
        or np.any(~np.isfinite(tau_grid))
    ):
        raise ValueError(
            "tau_grid must be a non-empty finite 1-D array"
        )
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if not np.isfinite(fc):
        raise ValueError("fc must be finite")
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA-capable GPU was detected by CuPy.")

    dt_all = np.diff(time_axis)
    dt = float(np.mean(dt_all))

    if (
        dt <= 0
        or not np.allclose(
            dt_all,
            dt,
            rtol=1e-8,
            atol=max(1e-18, abs(dt) * 1e-10),
        )
    ):
        raise ValueError("time_axis must be uniformly sampled")

    num_antennas, num_time_samples = data.shape
    num_ranges = range_grid.size
    num_angles = theta_grid.size
    num_tau = tau_grid.size

    data_gpu = cp.asarray(data)
    antenna_x_gpu = cp.asarray(antenna_x)
    range_grid_gpu = cp.asarray(range_grid)
    theta_grid_gpu = cp.asarray(theta_grid)
    tau_grid_gpu = cp.asarray(tau_grid)

    R_gpu, theta_gpu = cp.meshgrid(
        range_grid_gpu,
        theta_grid_gpu,
        indexing="ij",
    )
    R_flat = R_gpu.ravel()
    theta_flat = theta_gpu.ravel()

    num_candidates = int(R_flat.size)
    output_gpu = cp.zeros(
        (num_tau, num_candidates),
        dtype=cp.complex128,
    )

    antenna_indices = cp.arange(
        num_antennas,
        dtype=cp.int64,
    )[:, None, None]

    t0 = float(time_axis[0])

    for start in range(0, num_candidates, batch_size):
        stop = min(start + batch_size, num_candidates)

        relative_delay = _relative_spherical_delays_gpu(
            R_flat[start:stop],
            theta_flat[start:stop],
            antenna_x_gpu,
        )

        sample_position = (
            tau_grid_gpu[None, None, :]
            + relative_delay[:, :, None]
            - t0
        ) / dt

        index0 = cp.floor(sample_position).astype(cp.int64)
        fraction = sample_position - index0

        valid = (
            (index0 >= 0)
            & (index0 < num_time_samples - 1)
        )
        index0_safe = cp.clip(
            index0,
            0,
            num_time_samples - 2,
        )

        sample0 = data_gpu[antenna_indices, index0_safe]
        sample1 = data_gpu[antenna_indices, index0_safe + 1]

        samples = (
            (1.0 - fraction) * sample0
            + fraction * sample1
        )
        samples *= valid

        if carrier_phase_correction:
            phase = cp.exp(
                1j
                * 2.0
                * cp.pi
                * fc
                * relative_delay
            )[:, :, None]
            samples *= phase

        output_gpu[:, start:stop] = (
            cp.sum(samples, axis=0).T / num_antennas
        )

    output_gpu = output_gpu.reshape(
        num_tau,
        num_ranges,
        num_angles,
    )

    return cp.asnumpy(output_gpu)


# ============================================================
# Peak detection
# ============================================================

def hrt_peak(hrt, tau_grid, range_grid, theta_grid):
    """Return (tau_hat, R_hat, theta_hat, peak_value, index)."""
    hrt = np.asarray(hrt)
    tau_grid = np.asarray(tau_grid, dtype=float)
    range_grid = np.asarray(range_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)

    expected_shape = (
        len(tau_grid),
        len(range_grid),
        len(theta_grid),
    )

    if hrt.shape != expected_shape:
        raise ValueError(
            f"hrt shape must be {expected_shape}, got {hrt.shape}"
        )

    magnitude = np.abs(hrt)
    index = np.unravel_index(
        np.argmax(magnitude),
        magnitude.shape,
    )

    tau_hat = float(tau_grid[index[0]])
    range_hat = float(range_grid[index[1]])
    theta_hat = float(theta_grid[index[2]])
    peak_value = float(magnitude[index])

    return (
        tau_hat,
        range_hat,
        theta_hat,
        peak_value,
        index,
    )


# ============================================================
# Quick GPU check
# ============================================================

if __name__ == "__main__":
    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]

    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    free_memory, total_memory = cp.cuda.runtime.memGetInfo()

    print("GPU HRT module loaded successfully.")
    print(f"CUDA device: {device_name}")
    print(f"Free GPU memory: {free_memory / 1024**3:.2f} GiB")
    print(f"Total GPU memory: {total_memory / 1024**3:.2f} GiB")