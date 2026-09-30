from dataclasses import dataclass
import numpy as np

from config import C, PRT


# ============================================================
# ROI settings
# ============================================================

DQ = PRT.dq
K_Q = 4.5

THETA_HALF_WIDTH_DEG = 0.2
TAU_HALF_WIDTH_SAMPLES = 2


# ============================================================
# ROI result
# ============================================================

@dataclass(frozen=True)
class ROI:
    R_min: float
    R_max: float

    theta_min: float
    theta_max: float

    tau_min: float
    tau_max: float

    W_R: float
    W_theta: float
    W_tau: float

    @property
    def range_bounds(self):
        return self.R_min, self.R_max

    @property
    def theta_bounds(self):
        return self.theta_min, self.theta_max

    @property
    def tau_bounds(self):
        return self.tau_min, self.tau_max


# ============================================================
# Range ROI
# ============================================================

def range_roi_half_width(
    R_hat,
    theta_hat_deg,
    dq=DQ,
    k_q=K_Q,
    theta_half_width_deg=THETA_HALF_WIDTH_DEG
):
    """
    Calculate range ROI half-width.

    W_R =
        2*c*R_hat^2/cos^2(theta_hat) * (K_q * dq)
        +
        2*R_hat*|tan(theta_hat)| * delta_theta
    """

    R_hat = float(R_hat)
    theta_hat_deg = float(theta_hat_deg)

    if not np.isfinite(R_hat) or R_hat <= 0:
        raise ValueError("R_hat must be finite and > 0")

    if not np.isfinite(theta_hat_deg):
        raise ValueError("theta_hat_deg must be finite")

    theta = np.deg2rad(theta_hat_deg)
    cos2 = np.cos(theta) ** 2

    if cos2 <= np.finfo(float).eps:
        raise ValueError("theta_hat is too close to +/-90 deg")

    dq_eff = k_q * dq
    dtheta = np.deg2rad(theta_half_width_deg)

    W_q = 2.0 * C * R_hat**2 / cos2 * dq_eff
    W_theta = 2.0 * R_hat * abs(np.tan(theta)) * dtheta

    return float(W_q + W_theta)


# ============================================================
# Build complete ROI
# ============================================================

def build_roi(
    R_hat,
    theta_hat_deg,
    tau_hat,
    dq=DQ,
    k_q=K_Q,
    theta_half_width_deg=THETA_HALF_WIDTH_DEG,
    tau_half_width_samples=TAU_HALF_WIDTH_SAMPLES,
    range_bounds=None,
    theta_bounds=None,
    tau_bounds=None
):
    """
    Build the (R, theta, tau) ROI around the PRT estimate.

    Parameters
    ----------
    R_hat : float
        PRT range estimate [m].

    theta_hat_deg : float
        PRT angle estimate [deg].

    tau_hat : float
        PRT tau estimate [s].

    range_bounds : tuple or None
        Optional global physical range bounds.

    theta_bounds : tuple or None
        Optional global physical angle bounds.

    tau_bounds : tuple or None
        Optional global tau bounds.
    """

    tau_hat = float(tau_hat)

    if not np.isfinite(tau_hat):
        raise ValueError("tau_hat must be finite")

    W_R = range_roi_half_width(
        R_hat,
        theta_hat_deg,
        dq=dq,
        k_q=k_q,
        theta_half_width_deg=theta_half_width_deg
    )

    W_theta = float(theta_half_width_deg)
    W_tau = float(tau_half_width_samples * PRT.tau_step)

    R_min = float(R_hat - W_R)
    R_max = float(R_hat + W_R)

    theta_min = float(theta_hat_deg - W_theta)
    theta_max = float(theta_hat_deg + W_theta)

    tau_min = float(tau_hat - W_tau)
    tau_max = float(tau_hat + W_tau)

    if range_bounds is not None:
        R_min, R_max = _clip_interval(
            R_min, R_max, range_bounds, "range_bounds"
        )

    if theta_bounds is not None:
        theta_min, theta_max = _clip_interval(
            theta_min, theta_max, theta_bounds, "theta_bounds"
        )

    if tau_bounds is not None:
        tau_min, tau_max = _clip_interval(
            tau_min, tau_max, tau_bounds, "tau_bounds"
        )

    return ROI(
        R_min=R_min,
        R_max=R_max,
        theta_min=theta_min,
        theta_max=theta_max,
        tau_min=tau_min,
        tau_max=tau_max,
        W_R=W_R,
        W_theta=W_theta,
        W_tau=W_tau
    )


# ============================================================
# Helper
# ============================================================

def _clip_interval(lo, hi, bounds, name):
    b0, b1 = map(float, bounds)

    if not (
        np.isfinite(b0)
        and np.isfinite(b1)
        and b0 <= b1
    ):
        raise ValueError(f"{name} must satisfy min <= max")

    lo = max(lo, b0)
    hi = min(hi, b1)

    if lo > hi:
        raise ValueError(f"ROI does not intersect {name}")

    return lo, hi


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    R_hat = 200.0
    theta_hat = 30.0
    tau_hat = R_hat / C

    roi = build_roi(
        R_hat,
        theta_hat,
        tau_hat,
        range_bounds=(50.0, 400.0),
        theta_bounds=(-60.0, 60.0)
    )

    print(roi)