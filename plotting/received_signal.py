
"""Visualize the actual noisy received signal of three near-field UWB users.

Uses the project's spherical-wave signal model, free-space path loss,
carrier phase, and complex AWGN. No theoretical trajectories are overlaid.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_delays, received_signal
from signal_model.noise import add_awgn

# ==================== Settings ====================
USERS = [
    {"R": 3.0, "theta": -40.0, "tau_ns": 1500.0},
    {"R": 5.0, "theta": 0.0,   "tau_ns": 1500.0},
    {"R": 8.0, "theta": 40.0,  "tau_ns": 1500.0},
    {"R": 10.0, "theta": 30.0,  "tau_ns": 1500.0},
    {"R": 6.0, "theta": -20.0,  "tau_ns": 1500.0},
]

REF_RANGE = 5.0
REF_SNR_DB = 10.0
NOISE_SEED = 2026

# ==================== System ====================
x = ARRAY.positions(SYSTEM)
noise_var = SIM.noise_power

path_gain = (SYSTEM.wavelength / (4*np.pi*REF_RANGE))**2
tx_power = 10**(REF_SNR_DB/10) * noise_var / path_gain

# ==================== Time Axis ====================
all_delays = [
    propagation_delays(
        u["R"], u["theta"],
        clock_offset=u["tau_ns"]*1e-9-u["R"]/C,
        antenna_x=x
    )
    for u in USERS
]

margin = 6*SYSTEM.gaussian_sigma
t_min = min(v.min() for v in all_delays) - margin
t_max = max(v.max() for v in all_delays) + margin
t = np.arange(t_min, t_max + SYSTEM.dt, SYSTEM.dt)

# ==================== Multi-User Reception ====================
rx = np.zeros((len(x), len(t)), dtype=np.complex128)

for u in USERS:
    clock_offset = u["tau_ns"]*1e-9-u["R"]/C
    signal, _ = received_signal(
        t, u["R"], u["theta"],
        clock_offset=clock_offset,
        tx_power=tx_power,
        antenna_x=x
    )
    rx += signal

noisy_rx, noise_var = add_awgn(
    rx, rng=np.random.default_rng(NOISE_SEED)
)

# ==================== Visualization ====================
power_db = 10*np.log10(
    np.maximum(np.abs(noisy_rx)**2, 1e-30)/noise_var
)

fig, ax = plt.subplots(figsize=(8, 5))

im = ax.imshow(
    power_db.T, origin="lower", aspect="auto",
    extent=[x.min(), x.max(), t[0]*1e9, t[-1]*1e9],
    cmap="magma", vmin=-5, vmax=25,
    interpolation="nearest"
)

ax.set_xlabel("Antenna position (m)", fontsize=12)
ax.set_ylabel("Time (ns)", fontsize=12)
ax.tick_params(labelsize=10)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Received power / noise power (dB)", fontsize=11)

plt.tight_layout()
plt.show()

# ==================== Information ====================
print(f"Noise variance: {noise_var:.4e} W")
print(f"Transmit power: {tx_power:.4e} W")
print(f"Reference SNR: {REF_SNR_DB:.1f} dB at {REF_RANGE:.1f} m")
