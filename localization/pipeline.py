"""
PRT -> ROI -> HRT localization pipeline.

Modes:
1. localize(rx, time_axis, antenna_x)
   -> blind/global PRT: global p, global tau, global q.

2. localize(rx, time_axis, antenna_x, theta_hint, tau_hint)
   -> truth-assisted PRT:
      4 p bins around p_hint, 4 tau bins around tau_hint, global q.

After PRT, ROI and HRT do not use hints.
"""

from dataclasses import dataclass
import numpy as np

from config import C, SYSTEM, PRT
from signal_model.matched_filter import matched_filter
from transforms.prt import parabolic_radon_transform
from transforms.hrt import hyperbolic_radon_transform
from localization.parameter_mapping import pq_to_range, p_to_theta_deg
from localization.roi import build_roi


# ============================================================
# PRT settings
# ============================================================

PRT_THETA_MIN, PRT_THETA_MAX = -60.0, 60.0
PRT_R_MIN, PRT_R_MAX = 50.0, 400.0
N_LOCAL_P, N_LOCAL_TAU = 4, 4

P_MIN = -np.sin(np.deg2rad(PRT_THETA_MAX)) / C
P_MAX = -np.sin(np.deg2rad(PRT_THETA_MIN)) / C
P_GRID_GLOBAL = np.linspace(P_MIN, P_MAX, PRT.num_p_points)
DP = float(P_GRID_GLOBAL[1] - P_GRID_GLOBAL[0])

Q_MIN = np.cos(np.deg2rad(60.0))**2 / (2.0 * PRT_R_MAX * C)
Q_MAX = 1.0 / (2.0 * PRT_R_MIN * C)
Q_GRID = np.arange(Q_MIN - 2 * PRT.dq, Q_MAX + 2 * PRT.dq, PRT.dq)


# ============================================================
# HRT settings
# ============================================================

HRT_DR = 2.0
HRT_DTHETA = 0.01
HRT_DTAU = SYSTEM.dt / 4


# ============================================================
# Result
# ============================================================

@dataclass
class LocalizationResult:
    R: float; theta: float; tau: float; x: float; y: float
    prt_R: float; prt_theta: float; prt_tau: float
    roi: object; prt_peak: float; hrt_peak: float


# ============================================================
# Helpers
# ============================================================

def nearest_indices(grid, value, n):
    i = np.searchsorted(grid, value)
    start = int(np.clip(i - n // 2, 0, len(grid) - n))
    return np.arange(start, start + n)


def local_p_grid(theta_hint):
    p_hint = -np.sin(np.deg2rad(theta_hint)) / C
    return P_GRID_GLOBAL[nearest_indices(P_GRID_GLOBAL, p_hint, N_LOCAL_P)]


def local_tau_indices(time_axis, tau_hint):
    return nearest_indices(time_axis, tau_hint, N_LOCAL_TAU)


def fixed_grid(lo, hi, step):
    k0, k1 = int(np.ceil(lo / step)), int(np.floor(hi / step))
    return np.arange(k0, k1 + 1) * step if k1 >= k0 else np.array([(lo + hi) / 2.0])


def polar_to_xy(R, theta_deg):
    t = np.deg2rad(theta_deg)
    return float(R * np.sin(t)), float(R * np.cos(t))


def valid_pq_mask(p_grid):
    P, Q = np.meshgrid(p_grid, Q_GRID, indexing="ij")
    R = pq_to_range(P, Q)
    return np.isfinite(R) & (R >= PRT_R_MIN) & (R <= PRT_R_MAX)


# ============================================================
# PRT peak selection
# ============================================================

def find_prt_peak_local(prt, time_axis, p_grid, tau_hint):
    tau_idx = local_tau_indices(time_axis, tau_hint)
    valid = valid_pq_mask(p_grid)
    score = np.where(valid[None, :, :], np.abs(prt[tau_idx]), -np.inf)
    i = np.unravel_index(np.argmax(score), score.shape)
    return float(time_axis[tau_idx[i[0]]]), float(p_grid[i[1]]), float(Q_GRID[i[2]]), float(score[i])


def find_prt_peak_global(prt, time_axis, p_grid):
    valid = valid_pq_mask(p_grid)
    score = np.where(valid[None, :, :], np.abs(prt), -np.inf)
    i = np.unravel_index(np.argmax(score), score.shape)
    return float(time_axis[i[0]]), float(p_grid[i[1]]), float(Q_GRID[i[2]]), float(score[i])


# ============================================================
# PRT
# ============================================================

def run_prt(mf_rx, time_axis, antenna_x, theta_hint=None, tau_hint=None):
    local_mode = theta_hint is not None and tau_hint is not None
    p_grid = local_p_grid(theta_hint) if local_mode else P_GRID_GLOBAL

    prt = parabolic_radon_transform(mf_rx, antenna_x, p_grid, Q_GRID)

    if local_mode:
        tau_hat, p_hat, q_hat, peak = find_prt_peak_local(prt, time_axis, p_grid, tau_hint)
    else:
        tau_hat, p_hat, q_hat, peak = find_prt_peak_global(prt, time_axis, p_grid)

    return float(pq_to_range(p_hat, q_hat)), float(p_to_theta_deg(p_hat)), tau_hat, peak


# ============================================================
# HRT
# ============================================================

def find_hrt_peak(hrt, tau_grid, range_grid, theta_grid):
    score = np.abs(hrt)
    i = np.unravel_index(np.argmax(score), score.shape)
    return float(range_grid[i[1]]), float(theta_grid[i[2]]), float(tau_grid[i[0]]), float(score[i])


def run_hrt(mf_rx, time_axis, antenna_x, roi):
    r_grid = fixed_grid(roi.R_min, roi.R_max, HRT_DR)
    theta_grid = fixed_grid(roi.theta_min, roi.theta_max, HRT_DTHETA)
    tau_grid = fixed_grid(roi.tau_min, roi.tau_max, HRT_DTAU)

    hrt = hyperbolic_radon_transform(mf_rx, time_axis, antenna_x, r_grid, theta_grid, tau_grid)
    return find_hrt_peak(hrt, tau_grid, r_grid, theta_grid)


# ============================================================
# Complete pipeline
# ============================================================

def localize(rx, time_axis, antenna_x, theta_hint=None, tau_hint=None):
    if (theta_hint is None) != (tau_hint is None):
        raise ValueError("theta_hint and tau_hint must be provided together.")

    mf_rx = matched_filter(rx)
    R_prt, theta_prt, tau_prt, prt_peak = run_prt(mf_rx, time_axis, antenna_x, theta_hint, tau_hint)

    roi = build_roi(
        R_prt, theta_prt, tau_prt, dp=DP,
        range_bounds=(PRT_R_MIN, PRT_R_MAX),
        theta_bounds=(PRT_THETA_MIN, PRT_THETA_MAX)
    )

    R_hat, theta_hat, tau_hat, hrt_peak = run_hrt(mf_rx, time_axis, antenna_x, roi)
    x_hat, y_hat = polar_to_xy(R_hat, theta_hat)

    return LocalizationResult(R_hat, theta_hat, tau_hat, x_hat, y_hat,
                              R_prt, theta_prt, tau_prt, roi, prt_peak, hrt_peak)