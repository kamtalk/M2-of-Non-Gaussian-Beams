# @title generate_figure_noise_filtering.py
# FINAL VERSION: Using medium-large fonts and no descriptive titles.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --- Import from library files ---
from m2_utils import setup_grid, plot_beam_profile, calculate_m2_spatial_fft
from beam_definitions import create_gaussian_beam
from analysis_models import model_add_modes

# --- Parameters ---
grid_size = 256; xy_max = 3.75; w0_base = 2.0  
noise_std = 0.05; lg_basis_order = 10 

# --- Analysis ---
x, y, xx, yy, rr, phi = setup_grid(grid_size, xy_max)
psi_ideal, _ = create_gaussian_beam(w0_base, xx, yy)
np.random.seed(42)
psi_noisy = psi_ideal + np.random.normal(0,noise_std,psi_ideal.shape)+1j*np.random.normal(0,noise_std,psi_ideal.shape)
m2x_noisy_trad, _ = calculate_m2_spatial_fft(psi_noisy, x, y)
addmodes_results = model_add_modes(psi_noisy, (x, y), w0_base, lg_basis_order, basis_type='lg')
modal_power_spectrum = np.abs(addmodes_results['coeffs'])**2 / np.sum(np.abs(addmodes_results['coeffs'])**2)

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 11))

# Panel A
ax = axs[0, 0]
plot_beam_profile(ax, x, y, psi_ideal, title="", show_phase=False)
ax.set_title("(a)", loc='left', fontsize=18, fontweight='bold')
ax.text(0.05, 0.95, f'$M^2 = 1.0$', transform=ax.transAxes, 
        ha='left', va='top', fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

# Panel B
ax = axs[0, 1]
plot_beam_profile(ax, x, y, psi_noisy, title="", show_phase=False)
ax.set_title("(b)", loc='left', fontsize=18, fontweight='bold')
ax.text(0.05, 0.95, f'Spatial/FFT $M^2_x \\approx {m2x_noisy_trad:.1f}$', 
        transform=ax.transAxes, ha='left', va='top', fontsize=15, color='red', bbox=dict(facecolor='white', alpha=0.8))

# Panel C
ax = axs[1, 0]
plot_beam_profile(ax, x, y, addmodes_results['psi_reconstructed'], title="", show_phase=False)
ax.set_title("(c)", loc='left', fontsize=18, fontweight='bold')
ax.text(0.05, 0.95, f'Coefficient $M^2_x \\approx {addmodes_results["m2_coeffs"][0]:.3f}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=15, color='blue', bbox=dict(facecolor='white', alpha=0.8))

# Panel D
ax = axs[1, 1]
indices = np.arange(len(modal_power_spectrum))
sorted_indices = np.argsort(modal_power_spectrum)[::-1]
ax.bar(indices, modal_power_spectrum[sorted_indices], color='royalblue')
ax.set_title('(d)', loc='left', fontsize=18, fontweight='bold')
ax.set_xlabel('LG Mode Index (Sorted by Power)', fontsize=16)
ax.set_ylabel('Normalized Power', fontsize=16)
ax.set_ylim(bottom=1e-4); ax.set_xlim(-1, 15)
try:
    lg00_idx = addmodes_results['mode_keys'].index((0,0))
    lg00_pos = np.where(sorted_indices == lg00_idx)[0][0]
    ax.get_children()[lg00_pos].set_color('crimson')
except (ValueError, IndexError): pass
ax.set_xticks([0]); ax.set_xticklabels(['$LG_{0,0}$'], fontsize=14)

for ax_img in [axs[0,0], axs[0,1], axs[1,0]]:
    ax_img.set_xlabel('x', fontsize=16); ax_img.set_ylabel('y', fontsize=16)
    ax_img.tick_params(axis='both', which='major', labelsize=14)
axs[1,1].tick_params(axis='y', which='major', labelsize=14)

plt.tight_layout()
plt.savefig("Figure_Noise_Filtering.png", dpi=300)
plt.show()