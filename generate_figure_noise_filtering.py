# @title generate_figure_noise_filtering.py
# This script generates the second new figure to address Reviewer 2, Comment 4.
# It visually demonstrates the noise-filtering capability of the AddModes model
# by showing the reconstruction of a clean Gaussian beam from a very noisy input.
# CORRECTED VERSION 3: Max zoom on the beam spot.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --- Import functions from your existing library files ---
try:
    from m2_utils import setup_grid, plot_beam_profile, calculate_m2_spatial_fft
    from beam_definitions import create_gaussian_beam
    from analysis_models import model_add_modes
except ImportError as e:
    print(f"Error importing from library files: {e}")
    print("Please ensure m2_utils.py, beam_definitions.py, and analysis_models.py are in the same directory.")
    exit()

# ==============================================================
# SECTION 1: CORE PARAMETERS
# ==============================================================
grid_size = 256
# --- THIS IS THE MODIFIED LINE ---
xy_max = 3.75  # Changed from 7.5 to 3.75 for maximum zoom
# --- END OF MODIFICATION ---
w0_base = 2.0  

# A significant noise level to make the effect visually clear
noise_std = 0.05 

# Analysis parameters
lg_basis_order = 10 # Matches the noise robustness plot

# ==============================================================
# SECTION 2: GENERATE BEAMS AND RUN ANALYSIS
# ==============================================================
print("1. Setting up grid and generating ideal and noisy beams...")
x, y, xx, yy, rr, phi = setup_grid(grid_size, xy_max)

# Generate the ideal, clean beam
psi_ideal, _ = create_gaussian_beam(w0_base, xx, yy)

# Create the noisy version for analysis
# Use a fixed seed for reproducibility of the noisy image
np.random.seed(42)
noise_real = np.random.normal(loc=0.0, scale=noise_std, size=psi_ideal.shape)
noise_imag = np.random.normal(loc=0.0, scale=noise_std, size=psi_ideal.shape)
psi_noisy = psi_ideal + (noise_real + 1j * noise_imag)

# Calculate the (incorrect) M2 of the noisy beam using the traditional method
m2x_noisy_trad, _ = calculate_m2_spatial_fft(psi_noisy, x, y)
print(f"Calculated M2 for noisy beam on truncated grid: {m2x_noisy_trad:.2f}")

# --- Run the Additive Modal Decomposition analysis on the noisy beam ---
print("2. Running AddModes model on the noisy beam...")
addmodes_results = model_add_modes(psi_noisy, (x, y), w0_base, lg_basis_order, basis_type='lg')
psi_reconstructed = addmodes_results['psi_reconstructed']
r2_addmodes = addmodes_results['r_squared_intensity']
m2x_addmodes = addmodes_results['m2_coeffs'][0]
coeffs_addmodes = addmodes_results['coeffs']
mode_keys_addmodes = addmodes_results['mode_keys']
modal_power_spectrum = np.abs(coeffs_addmodes)**2 / np.sum(np.abs(coeffs_addmodes)**2)

# ==============================================================
# SECTION 3: PLOTTING THE RESULTS
# ==============================================================
print("3. Generating the 2x2 plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(11, 10))
fig.suptitle('Framework as a Noise Filter: Recovering a Beam from Noisy Data', fontsize=16, fontweight='bold')

# --- Panel (a): Ideal Input Beam ---
ax = axs[0, 0]
plot_beam_profile(ax, x, y, psi_ideal, title="(a) Ideal TEM$_{00}$ Beam", show_phase=False)
ax.text(0.05, 0.95, f'$M^2 = 1.0$', transform=ax.transAxes, 
        ha='left', va='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

# --- Panel (b): Noisy Input Beam ---
ax = axs[0, 1]
plot_beam_profile(ax, x, y, psi_noisy, title="(b) Noisy Input Beam", show_phase=False)
ax.text(0.05, 0.95, f'Spatial/FFT $M^2_x \\approx {m2x_noisy_trad:.1f}$', 
        transform=ax.transAxes, ha='left', va='top', fontsize=11, color='red', bbox=dict(facecolor='white', alpha=0.8))

# --- Panel (c): Reconstructed Beam ---
ax = axs[1, 0]
plot_beam_profile(ax, x, y, psi_reconstructed, title="(c) Reconstructed Beam (from AddModes)", show_phase=False)
ax.text(0.05, 0.95, f'Coefficient $M^2_x \\approx {m2x_addmodes:.3f}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=11, color='blue', bbox=dict(facecolor='white', alpha=0.8))

# --- Panel (d): Modal Power Spectrum ---
ax = axs[1, 1]
indices = np.arange(len(modal_power_spectrum))
sorted_indices = np.argsort(modal_power_spectrum)[::-1] # Sort by power
ax.bar(indices, modal_power_spectrum[sorted_indices], color='royalblue')
ax.set_title('(d) Recovered Modal Spectrum')
ax.set_xlabel('LG Mode Index (Sorted by Power)')
ax.set_ylabel('Normalized Power')
ax.set_ylim(bottom=1e-4) # Set y-axis floor
ax.set_xlim(-1, 15) # Show the first 15 most powerful modes

# Find the LG(0,0) mode and highlight it
lg00_label = '$LG_{0,0}$'
try:
    lg00_index_in_keys = mode_keys_addmodes.index((0, 0))
    # Find where this mode ended up after sorting
    lg00_sorted_pos = np.where(sorted_indices == lg00_index_in_keys)[0][0]
    ax.get_children()[lg00_sorted_pos].set_color('crimson') # Highlight the bar
except (ValueError, IndexError):
    lg00_sorted_pos = -1 # LG00 not found or not in top modes shown

# Label the most powerful mode
ax.set_xticks([0])
ax.set_xticklabels([lg00_label])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Figure_Noise_Filtering.png", dpi=300)
plt.show()

print("Script finished. Figure saved as Figure_Noise_Filtering.png")