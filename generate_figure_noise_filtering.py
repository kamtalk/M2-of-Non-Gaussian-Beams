# @title generate_figure_noise_filtering.py
# This script generates the second new figure to address Reviewer 2, Comment 4.
# It visually demonstrates the noise-filtering capability of the AddModes model.
# CORRECTED VERSION 4: Increased font sizes for all text elements for better readability.

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
xy_max = 3.75  # Using the zoomed-in value
w0_base = 2.0  

# A significant noise level to make the effect visually clear
noise_std = 0.05 

# Analysis parameters
lg_basis_order = 10 

# ==============================================================
# SECTION 2: GENERATE BEAMS AND RUN ANALYSIS
# ==============================================================
print("1. Setting up grid and generating ideal and noisy beams...")
x, y, xx, yy, rr, phi = setup_grid(grid_size, xy_max)

# Generate the ideal, clean beam
psi_ideal, _ = create_gaussian_beam(w0_base, xx, yy)

# Create the noisy version for analysis
np.random.seed(42)
noise_real = np.random.normal(loc=0.0, scale=noise_std, size=psi_ideal.shape)
noise_imag = np.random.normal(loc=0.0, scale=noise_std, size=psi_ideal.shape)
psi_noisy = psi_ideal + (noise_real + 1j * noise_imag)

# Calculate the M2 of the noisy beam
m2x_noisy_trad, _ = calculate_m2_spatial_fft(psi_noisy, x, y)
print(f"Calculated M2 for noisy beam on truncated grid: {m2x_noisy_trad:.2f}")

# --- Run the Additive Modal Decomposition analysis ---
print("2. Running AddModes model on the noisy beam...")
addmodes_results = model_add_modes(psi_noisy, (x, y), w0_base, lg_basis_order, basis_type='lg')
psi_reconstructed = addmodes_results['psi_reconstructed']
m2x_addmodes = addmodes_results['m2_coeffs'][0]
coeffs_addmodes = addmodes_results['coeffs']
mode_keys_addmodes = addmodes_results['mode_keys']
modal_power_spectrum = np.abs(coeffs_addmodes)**2 / np.sum(np.abs(coeffs_addmodes)**2)

# ==============================================================
# SECTION 3: PLOTTING THE RESULTS (WITH LARGER FONTS)
# ==============================================================
print("3. Generating the 2x2 plot with larger fonts...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(12, 11)) # Slightly larger figure size
fig.suptitle('Framework as a Noise Filter: Recovering a Beam from Noisy Data', fontsize=28, fontweight='bold')

# --- Panel (a): Ideal Input Beam ---
ax = axs[0, 0]
plot_beam_profile(ax, x, y, psi_ideal, title="", show_phase=False) # Title set below
ax.set_title("(a) Ideal TEM$_{00}$ Beam", fontsize=24)
ax.text(0.05, 0.95, f'$M^2 = 1.0$', transform=ax.transAxes, 
        ha='left', va='top', fontsize=22, bbox=dict(facecolor='white', alpha=0.8))

# --- Panel (b): Noisy Input Beam ---
ax = axs[0, 1]
plot_beam_profile(ax, x, y, psi_noisy, title="", show_phase=False) # Title set below
ax.set_title("(b) Noisy Input Beam", fontsize=24)
ax.text(0.05, 0.95, f'Spatial/FFT $M^2_x \\approx {m2x_noisy_trad:.1f}$', 
        transform=ax.transAxes, ha='left', va='top', fontsize=22, color='red', bbox=dict(facecolor='white', alpha=0.8))

# --- Panel (c): Reconstructed Beam ---
ax = axs[1, 0]
plot_beam_profile(ax, x, y, psi_reconstructed, title="", show_phase=False) # Title set below
ax.set_title("(c) Reconstructed Beam (from AddModes)", fontsize=24)
ax.text(0.05, 0.95, f'Coefficient $M^2_x \\approx {m2x_addmodes:.3f}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=22, color='blue', bbox=dict(facecolor='white', alpha=0.8))

# --- Panel (d): Modal Power Spectrum ---
ax = axs[1, 1]
indices = np.arange(len(modal_power_spectrum))
sorted_indices = np.argsort(modal_power_spectrum)[::-1]
ax.bar(indices, modal_power_spectrum[sorted_indices], color='royalblue')
ax.set_title('(d) Recovered Modal Spectrum', fontsize=24)
ax.set_xlabel('LG Mode Index (Sorted by Power)', fontsize=20)
ax.set_ylabel('Normalized Power', fontsize=20)
ax.set_ylim(bottom=1e-4)
ax.set_xlim(-1, 15)

# Find and highlight the LG(0,0) mode
lg00_label = '$LG_{0,0}$'
try:
    lg00_index_in_keys = mode_keys_addmodes.index((0, 0))
    lg00_sorted_pos = np.where(sorted_indices == lg00_index_in_keys)[0][0]
    ax.get_children()[lg00_sorted_pos].set_color('crimson')
except (ValueError, IndexError):
    pass 
ax.set_xticks([0])
ax.set_xticklabels([lg00_label], fontsize=18)

# --- Apply global font size changes to ticks and labels for all image plots ---
for ax in [axs[0,0], axs[0,1], axs[1,0]]:
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
axs[1,1].tick_params(axis='y', which='major', labelsize=18) # Set y-ticks for the bar chart

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Figure_Noise_Filtering_LargeFont.png", dpi=300)
plt.show()

print("Script finished. Figure saved as Figure_Noise_Filtering_LargeFont.png")