# @title generate_figure_mismatch_diagnostic.py
# FINAL VERSION: Using medium-large fonts and no descriptive titles.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --- Import from library files ---
from m2_utils import setup_grid, plot_beam_profile
from beam_definitions import create_aberrated_gaussian
from analysis_models import model_mult_poly, model_add_modes

# --- Parameters ---
grid_size = 256; xy_max = 15.0; w0_base = 2.0  
aberration_params_strong = {'P2': 2.0, 'P4': 1.0} 
lg_basis_order = 16; poly_order = 4      

# --- Analysis ---
x, y, xx, yy, rr, phi = setup_grid(grid_size, xy_max)
psi_data, _ = create_aberrated_gaussian(w0_base, xx, yy, rr, **aberration_params_strong)
pert_results = model_mult_poly(psi_data, (x, y), w0_base, poly_order)
addmodes_results = model_add_modes(psi_data, (x, y), w0_base, lg_basis_order, basis_type='lg')
modal_power_spectrum = np.abs(addmodes_results['coeffs'])**2 / np.sum(np.abs(addmodes_results['coeffs'])**2)

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 11))

# Panel A
ax = axs[0, 0]
plot_beam_profile(ax, x, y, psi_data, title="", show_phase=True)
ax.set_title("(a)", loc='left', fontsize=18, fontweight='bold')
ax.text(0.05, 0.95, f'$M^2_x \\approx 4.29$', transform=ax.transAxes, 
        ha='left', va='top', fontsize=15, bbox=dict(facecolor='white', alpha=0.7))

# Panel B
ax = axs[0, 1]
plot_beam_profile(ax, x, y, pert_results['psi_reconstructed'], title="", show_phase=True)
ax.set_title("(b)", loc='left', fontsize=18, fontweight='bold')
ax.text(0.05, 0.95, f'Model: MultPoly\n$R^2 = {pert_results["r_squared_intensity"]:.4f}$\nFitted $M^2_x \\approx {pert_results["m2_spatial"][0]:.3f}$', 
        transform=ax.transAxes, ha='left', va='top', fontsize=15, color='red', bbox=dict(facecolor='white', alpha=0.7))

# Panel C
ax = axs[1, 0]
plot_beam_profile(ax, x, y, addmodes_results['psi_reconstructed'], title="", show_phase=True)
ax.set_title("(c)", loc='left', fontsize=18, fontweight='bold')
ax.text(0.05, 0.95, f'Model: AddModes\n$R^2 = {addmodes_results["r_squared_intensity"]:.4f}$\nFitted $M^2_x \\approx {addmodes_results["m2_coeffs"][0]:.3f}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=15, color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Panel D
ax = axs[1, 1]
indices = np.arange(len(modal_power_spectrum))
sorted_indices = np.argsort(modal_power_spectrum)[::-1]
ax.bar(indices, modal_power_spectrum[sorted_indices], color='royalblue')
ax.set_title('(d)', loc='left', fontsize=18, fontweight='bold')
ax.set_xlabel('LG Mode Index (Sorted by Power)', fontsize=16, labelpad=15)
ax.set_ylabel('Normalized Power', fontsize=16)
ax.set_yscale('log'); ax.set_ylim(bottom=1e-5); ax.set_xlim(-1, 30)
top_mode_indices = sorted_indices[:4]
top_modes_labels = [f'$LG_{{{addmodes_results["mode_keys"][i][0]},{addmodes_results["mode_keys"][i][1]}}}$' for i in top_mode_indices]
ax.set_xticks(indices[:4])
ax.set_xticklabels(top_modes_labels, rotation=90, ha='center', fontsize=12)

for ax_img in [axs[0,0], axs[0,1], axs[1,0]]:
    ax_img.set_xlabel('x', fontsize=16); ax_img.set_ylabel('y', fontsize=16)
    ax_img.tick_params(axis='both', which='major', labelsize=14)
axs[1,1].tick_params(axis='y', which='major', labelsize=14)

plt.tight_layout()
plt.savefig("Figure_Mismatch_Diagnostic.png", dpi=300)
plt.show()