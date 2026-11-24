
"""Visualize LRT, PRT, and HRT responses for a near-field UWB user.

Uses the project's spherical-wave signal, AWGN, and matched-filter models.
Displays three independent Matplotlib figures without saving any files.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import received_signal
from signal_model.noise import add_awgn
from signal_model.matched_filter import matched_filter

# ==================== Settings ====================
R_TRUE, THETA_TRUE, TAU_NS = 5.0, 20.0, 1500.0
REF_RANGE, REF_SNR_DB = 5.0, 10.0
NOISE_SEED = 2026

NP, NQ, NR, NTH = 201, 201, 201, 201
P_LRT_MIN, P_LRT_MAX = -3.2, 3.2
TAU_SPAN_NS = 3.5

P_SPAN, Q_SPAN = 0.25, 0.10
R_SPAN, THETA_SPAN = 1.5, 4.0

CMAP = "viridis"
DB_MIN, DB_MAX = -30, 0

# ==================== System ====================
x = ARRAY.positions(SYSTEM)
tau0 = TAU_NS*1e-9
sigma_n2 = SIM.noise_power

p_true = -np.sin(np.deg2rad(THETA_TRUE))/C
q_true = np.cos(np.deg2rad(THETA_TRUE))**2/(2*R_TRUE*C)

gain_ref = (SYSTEM.wavelength/(4*np.pi*REF_RANGE))**2
tx_power = 10**(REF_SNR_DB/10)*sigma_n2/gain_ref

# ==================== Signal Generation ====================
clock_offset = tau0-R_TRUE/C
t = np.arange(tau0-6e-9, tau0+6e-9+SYSTEM.dt, SYSTEM.dt)

rx, _ = received_signal(
    t, R_TRUE, THETA_TRUE,
    clock_offset=clock_offset,
    tx_power=tx_power,
    antenna_x=x
)

noisy_rx, _ = add_awgn(rx, rng=np.random.default_rng(NOISE_SEED))
mf = matched_filter(noisy_rx)

# ==================== Generalized RT ====================
def rt_at_tau(g, tau):
    g = np.atleast_2d(g)
    nc = len(g)
    out = np.empty(nc, dtype=np.complex128)
    antennas = np.arange(len(x))[:, None]

    for start in range(0, nc, 256):
        block = g[start:start+256].T
        indices = (tau+block-t[0])/SYSTEM.dt
        coords = np.broadcast_to(antennas, block.shape)

        samples = map_coordinates(
            mf, [coords, indices],
            order=1, mode="constant", cval=0
        )

        phase = np.exp(1j*2*np.pi*SYSTEM.fc*block)
        out[start:start+256] = np.mean(samples*phase, axis=0)

    return out

def normalize_db(z):
    return 20*np.log10(np.maximum(z/max(z.max(), 1e-30), 1e-6))

# ==================== LRT ====================
p_lrt = np.linspace(P_LRT_MIN, P_LRT_MAX, NP)*1e-9
tau_lrt = np.linspace(TAU_NS-TAU_SPAN_NS, TAU_NS+TAU_SPAN_NS, 241)*1e-9

g_lrt = p_lrt[:, None]*x[None, :]
lrt = np.abs(np.array([rt_at_tau(g_lrt, tau) for tau in tau_lrt]))
lrt_db = normalize_db(lrt)

# ==================== PRT ====================
p_grid = np.linspace(p_true*1e9-P_SPAN, p_true*1e9+P_SPAN, NP)*1e-9
q_grid = np.linspace(q_true*1e9-Q_SPAN, q_true*1e9+Q_SPAN, NQ)*1e-9

P, Q = np.meshgrid(p_grid, q_grid, indexing="xy")
g_prt = P.ravel()[:, None]*x+Q.ravel()[:, None]*x**2

prt = np.abs(rt_at_tau(g_prt, tau0)).reshape(NQ, NP)
prt_db = normalize_db(prt)

# ==================== HRT ====================
r_grid = np.linspace(R_TRUE-R_SPAN, R_TRUE+R_SPAN, NR)
th_grid = np.linspace(THETA_TRUE-THETA_SPAN, THETA_TRUE+THETA_SPAN, NTH)

RR, TH = np.meshgrid(r_grid, th_grid, indexing="xy")
r = RR.ravel()[:, None]
th = np.deg2rad(TH.ravel())[:, None]

distance = np.sqrt(r**2+x**2-2*r*x*np.sin(th))
g_hrt = (distance-r)/C

hrt = np.abs(rt_at_tau(g_hrt, tau0)).reshape(NTH, NR)
hrt_db = normalize_db(hrt)

# ==================== Plot Settings ====================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot_rt(X, Y, Z, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im = ax.pcolormesh(
        X, Y, Z, shading="auto",
        cmap=CMAP, vmin=DB_MIN, vmax=DB_MAX,
        rasterized=True
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(direction="in")

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label("Normalized RT magnitude (dB)")

    fig.tight_layout()
    return fig, ax

# ==================== Three Separate Figures ====================
fig1, ax1 = plot_rt(
    p_lrt*1e9, tau_lrt*1e9, lrt_db,
    r"$p$ (ns/m)", r"$\tau$ (ns)", "LRT"
)

fig2, ax2 = plot_rt(
    p_grid*1e9, q_grid*1e9, prt_db,
    r"$p$ (ns/m)", r"$q$ (ns/m$^2$)", "PRT"
)

fig3, ax3 = plot_rt(
    r_grid, th_grid, hrt_db,
    r"$R$ (m)", r"$\theta$ (deg)", "HRT"
)

plt.show()

# ==================== Information ====================
print(f"True R = {R_TRUE:.2f} m")
print(f"True theta = {THETA_TRUE:.2f} deg")
print(f"True TOA = {TAU_NS:.3f} ns")
print(f"True p = {p_true*1e9:.6f} ns/m")
print(f"True q = {q_true*1e9:.6f} ns/m^2")
print(f"Reference SNR = {REF_SNR_DB:.1f} dB")
