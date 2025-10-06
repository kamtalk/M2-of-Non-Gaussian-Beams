# @title generate_figure_2.py
# FINAL VERSION 3: Using a simple y-axis scale increase to fix label overlap.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from numpy.linalg import lstsq
import matplotlib.colors as mcolors
import math

# --- Parameters ---
grid_size = 256
xy_max = 15.0 
w0 = 3.0
P2 = 2.0
P4 = 1.0
max_mode_sum_fit = 6

# --- Grid and Beam Generation ---
x = np.linspace(-xy_max, xy_max, grid_size)
y = np.linspace(-xy_max, xy_max, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
gaussian_base = np.exp(-(R/w0)**2)
aberration_phase = P2 * (R/w0)**2 + P4 * (R/w0)**4
psi_data = gaussian_base * np.exp(1j * aberration_phase)

# --- HG Mode Functions and Fitting ---
def psi_nm_HG(x, y, n, m, w0):
    try:
        norm_val = math.factorial(n) * math.factorial(m)
        norm_factor = 1 / (w0 * np.sqrt(np.pi * 2**(n+m) * norm_val)) if norm_val > 0 else 0
    except (ValueError, OverflowError): 
        norm_factor = 0
    if norm_factor == 0: return np.zeros_like(x, dtype=complex)
    xi = np.sqrt(2) * x / w0; eta = np.sqrt(2) * y / w0
    Hn = hermite(n)(xi); Hm = hermite(m)(eta)
    return norm_factor * Hn * Hm * np.exp(-(x**2 + y**2) / w0**2)

hg_modes, mode_indices = [], []
for n in range(max_mode_sum_fit + 1):
    for m in range(max_mode_sum_fit + 1 - n):
        hg_modes.append(psi_nm_HG(X, Y, n, m, w0))
        mode_indices.append((n, m))

Psi_matrix = np.array([mode.flatten() for mode in hg_modes]).T
coefficients, _, _, _ = lstsq(Psi_matrix, psi_data.flatten(), rcond=1e-8)
psi_reconstructed = (Psi_matrix @ coefficients).reshape(grid_size, grid_size)
original_intensity = np.abs(psi_data)**2
reconstructed_intensity = np.abs(psi_reconstructed)**2
r_squared = 1 - np.sum((original_intensity - reconstructed_intensity)**2) / np.sum((original_intensity - np.mean(original_intensity))**2)
normalized_modal_power = np.abs(coefficients)**2 / np.sum(np.abs(coefficients)**2)

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={'width_ratios': [1, 1.2]})
fig.subplots_adjust(wspace=0.4, hspace=0.4)
zoom_extent_3x = xy_max / 3.0
zoom_extent_2x = xy_max / 2.0

# Panel A
ax1 = axes[0, 0]
im1 = ax1.imshow(original_intensity, cmap='viridis', extent=[-xy_max, xy_max, -xy_max, xy_max], origin='lower')
ax1.set_title('(a)', loc='left', fontsize=18, fontweight='bold')
ax1.set_xlabel('x (a.u.)', fontsize=16); ax1.set_ylabel('y (a.u.)', fontsize=16)
ax1.set_xlim(-zoom_extent_3x, zoom_extent_3x); ax1.set_ylim(-zoom_extent_3x, zoom_extent_3x)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=12)

# Panel B
ax2 = axes[0, 1]
phase_cmap = mcolors.LinearSegmentedColormap.from_list("hsv_like", plt.cm.hsv(np.linspace(0, 1, 256)))
intensity_mask = original_intensity > 1e-4 * np.max(original_intensity)
original_phase_masked = np.where(intensity_mask, np.angle(psi_data), np.nan)
im2 = ax2.imshow(original_phase_masked, cmap=phase_cmap, extent=[-xy_max, xy_max, -xy_max, xy_max], origin='lower', vmin=-np.pi, vmax=np.pi)
ax2.set_title('(b)', loc='left', fontsize=18, fontweight='bold')
ax2.set_xlabel('x (a.u.)', fontsize=16); ax2.set_ylabel('y (a.u.)', fontsize=16)
ax2.set_xlim(-zoom_extent_2x, zoom_extent_2x); ax2.set_ylim(-zoom_extent_2x, zoom_extent_2x)
cbar2 = plt.colorbar(im2, ax=ax2, ticks=[-np.pi, 0, np.pi], fraction=0.046, pad=0.04)
cbar2.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'], fontsize=12)

# Panel C
ax3 = axes[1, 0]
im3 = ax3.imshow(reconstructed_intensity, cmap='viridis', extent=[-xy_max, xy_max, -xy_max, xy_max], origin='lower')
ax3.set_title(f'(c)', loc='left', fontsize=18, fontweight='bold')
ax3.set_xlabel('x (a.u.)', fontsize=16); ax3.set_ylabel('y (a.u.)', fontsize=16)
ax3.set_xlim(-zoom_extent_3x, zoom_extent_3x); ax3.set_ylim(-zoom_extent_3x, zoom_extent_3x)
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.ax.tick_params(labelsize=12)

# Panel D
ax4 = axes[1, 1]
sorted_indices = sorted(range(len(hg_modes)), key=lambda k: (mode_indices[k][0] + mode_indices[k][1], mode_indices[k][0]))
sorted_power = normalized_modal_power[sorted_indices]
ax4.bar(np.arange(len(hg_modes)), sorted_power)
ax4.set_title('(d)', loc='left', fontsize=18, fontweight='bold')
ax4.set_ylabel('Normalized Power', fontsize=16)
ax4.set_xlabel('Mode Index (Sorted by $n+m$; key modes labeled)', fontsize=16)
ax4.grid(axis='y', linestyle='--'); ax4.xaxis.set_major_locator(plt.NullLocator())

# --- THIS IS THE CORRECTED SECTION ---
# Increase the upper y-axis limit by 10% to create space
current_ylim = ax4.get_ylim()
ax4.set_ylim(current_ylim[0], current_ylim[1] * 1.1)
# --- END OF CORRECTION ---

# Labeling Logic for Panel D
labeled_modes = {(0,0), (2,0), (0,2), (4,0), (0,4), (1,1)}
for i, original_idx in enumerate(sorted_indices):
    n, m = mode_indices[original_idx]
    if (n,m) in labeled_modes:
        height = sorted_power[i]
        ax4.text(i, height + 0.01, f'({n},{m})', ha='center', va='bottom', rotation=90, fontsize=10, weight='bold')

for ax in axes.flatten():
    ax.tick_params(axis='both', which='major', labelsize=14)

plt.savefig('aberrated_tem00_hg_fit_improved_fixed.png', dpi=300, bbox_inches='tight')
plt.show()