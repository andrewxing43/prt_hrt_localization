import numpy as np
from config import C


def p_to_theta(p: float | np.ndarray) -> float | np.ndarray:
    """
    Convert PRT slowness p [s/m] to angle theta [rad].

    p = -sin(theta) / c
    """
    x = np.clip(-np.asarray(p, dtype=float) * C, -1.0, 1.0)
    theta = np.arcsin(x)
    return float(theta) if np.ndim(theta) == 0 else theta


def p_to_theta_deg(p: float | np.ndarray) -> float | np.ndarray:
    """Convert PRT slowness p [s/m] to angle theta [deg]."""
    return np.rad2deg(p_to_theta(p))


def pq_to_range(p: float | np.ndarray,
                q: float | np.ndarray) -> float | np.ndarray:
    """
    Convert PRT parameters (p, q) to range R [m].

    R = (1 - (p c)^2) / (2 q c)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    numerator = 1.0 - (p * C) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        range_m = numerator / (2.0 * q * C)

    return float(range_m) if np.ndim(range_m) == 0 else range_m


def range_theta_to_pq(range_m: float | np.ndarray,
                      theta_deg: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert physical source parameters (R, theta) to PRT parameters (p, q).

    p = -sin(theta) / c
    q = cos^2(theta) / (2 R c)
    """
    range_m = np.asarray(range_m, dtype=float)
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))

    p = -np.sin(theta) / C
    q = np.cos(theta) ** 2 / (2.0 * range_m * C)

    return p, q


if __name__ == "__main__":
    range_true, theta_true = 200.0, -18.0

    p, q = range_theta_to_pq(range_true, theta_true)
    range_back = pq_to_range(p, q)
    theta_back = p_to_theta_deg(p)

    print(f"True range: {range_true:.3f} m")
    print(f"Recovered range: {range_back:.3f} m")
    print(f"True angle: {theta_true:.3f} deg")
    print(f"Recovered angle: {theta_back:.3f} deg")
    print(f"p: {float(p):.6e} s/m")
    print(f"q: {float(q):.6e} s/m^2")