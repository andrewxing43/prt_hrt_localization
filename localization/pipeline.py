"""
PRT -> ROI -> HRT near-field localization pipeline.

Purpose:
1. Apply matched filtering to the received array signal.
2. Run PRT over the global parameter space for coarse estimation of tau, R and theta.
3. Use the PRT estimate to construct a physically motivated ROI.
4. Run exact spherical-wave HRT only inside that ROI.
5. Return the final HRT estimate in both polar (R, theta, tau) and Cartesian (x, y) coordinates.

This file only connects existing modules. It does not generate signals, add noise,
or use any ground-truth user position.
"""

from dataclasses import dataclass
import numpy as np

from config import C, SYSTEM, PRT
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from transforms.hrt import hyperbolic_radon_transform, hrt_peak
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import build_roi


# ============================================================
# Global PRT search space
# ============================================================

PRT_THETA_MIN, PRT_THETA_MAX = -65.0, 65.0
PRT_R_MIN, PRT_R_MAX = 40.0, 500.0

THETA_PRT = np.arange(PRT_THETA_MIN, PRT_THETA_MAX + 0.5 * PRT.theta_step_deg, PRT.theta_step_deg)
P_GRID = -np.sin(np.deg2rad(THETA_PRT)) / C

Q_MIN = np.cos(np.deg2rad(max(abs(PRT_THETA_MIN), abs(PRT_THETA_MAX))))**2 / (2.0 * PRT_R_MAX * C)
Q_MAX = 1.0 / (2.0 * PRT_R_MIN * C)
Q_GRID = np.arange(Q_MIN - 2 * PRT.dq, Q_MAX + 2 * PRT.dq, PRT.dq)


# ============================================================
# HRT settings
# ============================================================

HRT_DR = 1.0
HRT_DTHETA = 0.005
HRT_DTAU = SYSTEM.dt / 4


# ============================================================
# Result
# ============================================================

@dataclass
class LocalizationResult:
    R: float
    theta: float
    tau: float
    x: float
    y: float

    prt_R: float
    prt_theta: float
    prt_tau: float

    roi: object
    hrt_peak: float


# ============================================================
# Helpers
# ============================================================

def fixed_grid(lo, hi, step):
    k0, k1 = int(np.ceil(lo / step)), int(np.floor(hi / step))
    if k1 >= k0:
        return np.arange(k0, k1 + 1) * step
    return np.array([(lo + hi) / 2.0])


def prt_peak(prt, time_axis):
    i = np.unravel_index(np.argmax(np.abs(prt)), prt.shape)
    return float(time_axis[i[0]]), float(P_GRID[i[1]]), float(Q_GRID[i[2]])


def polar_to_xy(R, theta_deg):
    theta = np.deg2rad(theta_deg)
    return float(R * np.sin(theta)), float(R * np.cos(theta))


# ============================================================
# PRT coarse localization
# ============================================================

def run_prt(mf_rx, time_axis, antenna_x):
    prt = parabolic_radon_transform(mf_rx, antenna_x, P_GRID, Q_GRID)
    tau_hat, p_hat, q_hat = prt_peak(prt, time_axis)

    theta_hat = float(p_to_theta_deg(p_hat))
    R_hat = float(pq_to_range(p_hat, q_hat))

    return R_hat, theta_hat, tau_hat


# ============================================================
# HRT fine localization
# ============================================================

def run_hrt(mf_rx, time_axis, antenna_x, roi):
    range_grid = fixed_grid(roi.R_min, roi.R_max, HRT_DR)
    theta_grid = fixed_grid(roi.theta_min, roi.theta_max, HRT_DTHETA)
    tau_grid = fixed_grid(roi.tau_min, roi.tau_max, HRT_DTAU)

    hrt = hyperbolic_radon_transform(mf_rx, time_axis, antenna_x, range_grid, theta_grid, tau_grid)
    tau_hat, R_hat, theta_hat, peak, _ = hrt_peak(hrt, tau_grid, range_grid, theta_grid)

    return float(R_hat), float(theta_hat), float(tau_hat), float(abs(peak))


# ============================================================
# Complete localization pipeline
# ============================================================

def localize(rx, time_axis, antenna_x):
    # 1. Matched filter
    mf_rx = matched_filter(rx)

    # 2. PRT coarse localization
    R_prt, theta_prt, tau_prt = run_prt(mf_rx, time_axis, antenna_x)

    # 3. Construct ROI from PRT estimate
    roi = build_roi(R_prt, theta_prt, tau_prt)

    # 4. Exact spherical-wave HRT inside ROI
    R_hat, theta_hat, tau_hat, peak = run_hrt(mf_rx, time_axis, antenna_x, roi)

    # 5. Polar -> Cartesian
    x_hat, y_hat = polar_to_xy(R_hat, theta_hat)

    return LocalizationResult(
        R=R_hat, theta=theta_hat, tau=tau_hat, x=x_hat, y=y_hat,
        prt_R=R_prt, prt_theta=theta_prt, prt_tau=tau_prt,
        roi=roi, hrt_peak=peak
    )