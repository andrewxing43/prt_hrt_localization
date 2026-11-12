import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light as c
from matplotlib.legend_handler import HandlerTuple

# Array apertures & Range defined as functions of L_long
L_long = 12.0            # Long aperture: [-L_long/2, L_long/2]
L_short = L_long / 5.0   # Short aperture: 1/5 * L_long (2.4 m)

# --- 1. Define Two Sources ---
# Source 1
theta_deg_1 = -18
theta_1 = np.deg2rad(theta_deg_1)
R_1 = 3.0 * L_long
x_s1 = R_1 * np.sin(theta_1)
z_s1 = R_1 * np.cos(theta_1)

# Source 2
theta_deg_2 = 25
theta_2 = np.deg2rad(theta_deg_2)
R_2 = 2.5 * L_long
x_s2 = R_2 * np.sin(theta_2)
z_s2 = R_2 * np.cos(theta_2)

N_ant = 11               # Number of discrete antenna elements
x_ant = np.linspace(-L_long/2, L_long/2, N_ant)
x_cont = np.linspace(-L_long/2, L_long/2, 500)

# --- 2. Compute Distance and Delay Models for Both Sources ---
# Source 1
d_cont_1 = np.sqrt(R_1**2 - 2*R_1*x_cont*np.sin(theta_1) + x_cont**2)
d_ant_1  = np.sqrt(R_1**2 - 2*R_1*x_ant*np.sin(theta_1) + x_ant**2)

dt_HRT_1 = (d_cont_1 - R_1) / c * 1e9
dt_PRT_1 = (-np.sin(theta_1)/c * x_cont + (np.cos(theta_1)**2)/(2*R_1*c) * x_cont**2) * 1e9
dt_LRT_1 = (-np.sin(theta_1)/c * x_cont) * 1e9
dt_ant_1 = (d_ant_1 - R_1) / c * 1e9

# Source 2
d_cont_2 = np.sqrt(R_2**2 - 2*R_2*x_cont*np.sin(theta_2) + x_cont**2)
d_ant_2  = np.sqrt(R_2**2 - 2*R_2*x_ant*np.sin(theta_2) + x_ant**2)

dt_HRT_2 = (d_cont_2 - R_2) / c * 1e9
dt_PRT_2 = (-np.sin(theta_2)/c * x_cont + (np.cos(theta_2)**2)/(2*R_2*c) * x_cont**2) * 1e9
dt_LRT_2 = (-np.sin(theta_2)/c * x_cont) * 1e9
dt_ant_2 = (d_ant_2 - R_2) / c * 1e9

# ==========================================
# PLOT 1: XZ Space Geometry & Wavefronts
# ==========================================
fig1, ax1 = plt.subplots(figsize=(10, 10))

x_neg_lim = -1.5
x_pos_lim = 1.5 
phi = np.linspace(-np.pi, 0, 600)

# Helper function to plot source geometry (SIMPLIFIED)
def plot_source_geometry(x_s, z_s, R, theta, color_wf, color_pw, label_prefix):
    # Just ONE intermediate circular arc for minimal context
    r_wf = 0.5 * R
    wx = x_s + r_wf * np.cos(phi)
    wz = z_s + r_wf * np.sin(phi)
    mask = (wz >= -1.0) & (wx >= x_neg_lim * L_long) & (wx <= x_pos_lim * L_long)
    ax1.plot(wx[mask], wz[mask], color=color_wf, alpha=0.35, lw=1.2, ls='--')

    # Solid arc: "One wavefront at a fixed time" (r = R)
    wx_ref = x_s + R * np.cos(phi)
    wz_ref = z_s + R * np.sin(phi)
    mask_ref = (wx_ref >= x_neg_lim * L_long) & (wx_ref <= x_pos_lim * L_long)
    ax1.plot(wx_ref[mask_ref], wz_ref[mask_ref], color=color_wf, lw=2.0, alpha=0.9,
             label='Spherical Wavefront')

    # Dotted line: "Local plane-wave approximation"
    x_pw = np.linspace(x_neg_lim * L_long, x_pos_lim * L_long, 100)
    z_pw = -x_pw * np.tan(theta)
    ax1.plot(x_pw, z_pw, color=color_pw, ls=':', lw=2.2, alpha=0.85,
             label='Plane-Wave Approx')

    # Single ray to the center antenna only (reduces crossing lines)
    ax1.plot([x_s, 0], [z_s, 0], color=color_wf, alpha=0.6, lw=1.5)

    # Plot and label the star "Point source"
    ax1.scatter([x_s], [z_s], marker='*', s=280, color='gold', edgecolor='black', zorder=10)
    ax1.annotate(f'Source {label_prefix}', xy=(x_s, z_s), xytext=(x_s - 0.10 * L_long, z_s + 0.025 * R),
                 fontsize=15,
                 bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

# Plot Source 1 (Blues/Reds)
plot_source_geometry(x_s1, z_s1, R_1, theta_1, '#1f77b4', '#d62728', '1')

# Plot Source 2 (Purples/Greens)
plot_source_geometry(x_s2, z_s2, R_2, theta_2, '#9467bd', '#2ca02c', '2')

# Draw antenna array along z = 0
ax1.axhline(0, color='black', lw=1.2, alpha=0.5)
ax1.axvline(0, color='black', ls='-.', alpha=0.3, lw=1.0)
short_mask = np.abs(x_ant) <= L_short/2 + 1e-5
ax1.scatter(x_ant[~short_mask], np.zeros(np.sum(~short_mask)), marker='^', s=95,
            facecolor='#ff7f0e', edgecolor='black', zorder=5, label='Long ULA (Length L)')
ax1.scatter(x_ant[short_mask], np.zeros(np.sum(short_mask)), marker='^', s=110,
            facecolor='#2ca02c', edgecolor='black', zorder=6, label='Short ULA (Length L/5)')

ax1.axvspan(-L_short/2, L_short/2, color='#2ca02c', alpha=0.12)

# Explicit limits to crop the plot cleanly
ax1.set_xlim(x_neg_lim * L_long, x_pos_lim * L_long)
ax1.set_ylim(-0.35 * L_long, 3.2 * L_long)
ax1.set_aspect('equal', adjustable='box')

ax1.set_xticks([-0.5 * L_long, -0.25 * L_long, 0, 0.25 * L_long, 0.5 * L_long])
ax1.set_xticklabels([r'$-L/2$', r'$-L/4$', r'$0$', r'$L/4$', r'$L/2$'])
ax1.set_yticks([0, 0.5 * L_long, 1.0 * L_long, 1.5 * L_long, 2.0 * L_long, 2.5 * L_long, 3.0 * L_long])
ax1.set_yticklabels([r'$0$', r'$L/2$', r'$L$', r'$3L/2$', r'$2L$', r'$5L/2$',  r'$3L$'])

ax1.set_xlabel('Antenna Position $x$', fontsize=20)
ax1.set_ylabel('Propagation Axis $z$', fontsize=20)
ax1.set_title('Wavefront Sampled across Uniform Linear Array', fontsize=25, pad=12)

# Group handles into tuples to show both colors side-by-side in the legend
handles, labels = ax1.get_legend_handles_labels()
grouped_handles = [
    (handles[0], handles[2]),  # S1 & S2 Spherical Wavefronts
    (handles[1], handles[3]),  # S1 & S2 Plane-Wave Approximations
    handles[4],                # Long ULA
    handles[5]                 # Short ULA
]
grouped_labels = ['Spherical Wavefront', 'Plane-Wave Approx', 'Long ULA (Length L)', 'Short ULA (Length L/5)']

ax1.legend(grouped_handles, grouped_labels, loc='upper right', fontsize=15, framealpha=0.95,
           handler_map={tuple: HandlerTuple(ndivide=None)})

ax1.grid(True, alpha=0.25)

fig1.tight_layout()
fig1.savefig('plot_1_clean_two_sources.png', dpi=150, bbox_inches='tight')

# ==========================================
# PLOT 2: Relative Delay Models vs Position x
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 4.5))

# Plot Source 1 Delays
ax2.plot(x_cont, dt_HRT_1, color='#1f77b4', lw=2.6, label='S1: Exact (HRT)')
ax2.plot(x_cont, dt_PRT_1, color='#ff7f0e', lw=2.2, ls='--', label='S1: Quad Approx (PRT)')
ax2.plot(x_cont, dt_LRT_1, color='#d62728', lw=2.0, ls=':', label='S1: Plane wave (LRT)')

# Plot Source 2 Delays
ax2.plot(x_cont, dt_HRT_2, color='#9467bd', lw=2.6, label='S2: Exact (HRT)')
ax2.plot(x_cont, dt_PRT_2, color='#8c564b', lw=2.2, ls='--', label='S2: Quad Approx (PRT)')
ax2.plot(x_cont, dt_LRT_2, color='#2ca02c', lw=2.0, ls=':', label='S2: Plane wave (LRT)')

# Discrete antenna samples 
ax2.scatter(x_ant, dt_ant_1, color='#1f77b4', edgecolor='black', zorder=6, s=40)
ax2.scatter(x_ant, dt_ant_2, color='#9467bd', edgecolor='black', zorder=6, s=40)

# Shade short aperture
ax2.axvspan(-L_short/2, L_short/2, color='#2ca02c', alpha=0.14)

ax2.set_xlim(-0.625 * L_long, 0.625 * L_long)

ax2.set_xticks([-0.5 * L_long, -0.25 * L_long, 0, 0.25 * L_long, 0.5 * L_long])
ax2.set_xticklabels([r'$-L/2$', r'$-L/4$', r'$0$', r'$L/4$', r'$L/2$'])

ax2.set_xlabel('Position along array', fontsize=11)
ax2.set_ylabel('Delay relative to center antenna (ns)', fontsize=11)
ax2.set_title('Relative Time Delay Models Across Apertures (Two Sources)', fontsize=12,  pad=12)

ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
           fontsize=15, framealpha=0.95)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig('plot_2_clean_two_sources.png', dpi=150, bbox_inches='tight')
print("Both multi-source clean plots generated.")