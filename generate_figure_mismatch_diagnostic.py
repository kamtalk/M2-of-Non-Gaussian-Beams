# @title generate_figure_mismatch_diagnostic.py
# This script generates a new figure to address Reviewer 2, Comment 4.
# It visually demonstrates the framework's diagnostic capability by comparing
# a failed Perturbation Model fit with a successful AddModes fit for a
# strongly aberrated TEM00 beam.
# FINAL CORRECTED VERSION

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --- Import functions from your existing library files ---
try:
    from m2_utils import setup_grid, plot_beam_profile
    from beam_definitions import create_aberrated_gaussian
    from analysis_models import model_mult_poly, model_add_modes
except ImportError as e:
    print(f"Error importing from library files: {e}")
    print("Please ensure m2_utils.py, beam_definitions.py, and analysis_models.py are in the same directory.")
    exit()

# ==============================================================
# SECTION 1: CORE PARAMETERS
# ==============================================================
grid_size = 256
xy_max = 15.0 
w0_base = 2.0  

# Parameters for the strongly aberrated beam (matches Table 1)
aberration_params_strong = {'P2': 2.0, 'P4': 1.0} 

# Analysis parameters
lg_basis_order = 16 # Use a sufficient order for the AddModes fit
poly_order = 4      # Order for the failed MultPoly fit

# ==============================================================
# SECTION 2: GENERATE BEAM AND RUN ANALYSES
# ==============================================================
print("1. Setting up grid and generating the strongly aberrated beam...")
x, y, xx, yy, rr, phi = setup_grid(grid_size, xy_max)
psi_data, _ = create_aberrated_gaussian(w0_base, xx, yy, rr, **aberration_params_strong)

# --- Run the FAILED Perturbation Model analysis ---
print("2. Running the Perturbation Model (MultPoly)...")
pert_results = model_mult_poly(psi_data, (x, y), w0_base, poly_order)
psi_reconstructed_pert = pert_results['psi_reconstructed']
r2_pert = pert_results['r_squared_intensity']
m2x_pert, _ = pert_results['m2_spatial']
# From Table 1, True M2x is ~4.288

# --- Run the SUCCESSFUL Additive Modal Decomposition analysis ---
print("3. Running the Additive Modal Decomposition Model (AddModes)...")
addmodes_results = model_add_modes(psi_data, (x, y), w0_base, lg_basis_order, basis_type='lg')
psi_reconstructed_addmodes = addmodes_results['psi_reconstructed']
r2_addmodes = addmodes_results['r_squared_intensity']
m2x_addmodes = addmodes_results['m2_coeffs'][0] # Correct unpacking
coeffs_addmodes = addmodes_results['coeffs']
mode_keys_addmodes = addmodes_results['mode_keys']
modal_power_spectrum = np.abs(coeffs_addmodes)**2 / np.sum(np.abs(coeffs_addmodes)**2)

# ==============================================================
# SECTION 3: PLOTTING THE RESULTS
# ==============================================================
print("4. Generating the 2x2 plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(11, 10))
fig.suptitle('Framework Diagnostic: Characterizing a Strongly Aberrated Beam', fontsize=16, fontweight='bold')

# --- Panel (a): Original Input Beam ---
ax = axs[0, 0] # CORRECTED: Select the top-left subplot
plot_beam_profile(ax, x, y, psi_data, title="(a) Original Aberrated Beam", show_phase=True)
ax.text(0.05, 0.95, f'$M^2_x \\approx 4.29$', transform=ax.transAxes, 
        ha='left', va='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

# --- Panel (b): Failed Perturbation Model Reconstruction ---
ax = axs[0, 1] # CORRECTED: Select the top-right subplot
plot_beam_profile(ax, x, y, psi_reconstructed_pert, title="(b) Perturbation Model (Failed)", show_phase=True)
ax.text(0.05, 0.95, f'Model: MultPoly(Gauss Base)\n$R^2 = {r2_pert:.4f}$\nFitted $M^2_x \\approx {m2x_pert:.3f}$', 
        transform=ax.transAxes, ha='left', va='top', fontsize=11, color='red', bbox=dict(facecolor='white', alpha=0.7))

# --- Panel (c): Successful AddModes Reconstruction ---
ax = axs[1, 0] # CORRECTED: Select the bottom-left subplot
plot_beam_profile(ax, x, y, psi_reconstructed_addmodes, title="(c) AddModes Model (Successful)", show_phase=True)
ax.text(0.05, 0.95, f'Model: AddModes(LG, M={lg_basis_order})\n$R^2 = {r2_addmodes:.4f}$\nFitted $M^2_x \\approx {m2x_addmodes:.3f}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=11, color='blue', bbox=dict(facecolor='white', alpha=0.7))

# --- Panel (d): Modal Power Spectrum ---
ax = axs[1, 1] # CORRECTED: Select the bottom-right subplot
indices = np.arange(len(modal_power_spectrum))
sorted_indices = np.argsort(modal_power_spectrum)[::-1] # Sort by power
ax.bar(indices, modal_power_spectrum[sorted_indices], color='royalblue')
ax.set_title('(d) Modal Power Spectrum (from AddModes)')
ax.set_xlabel('LG Mode Index (Sorted by Power)', labelpad=15)
ax.set_ylabel('Normalized Power')
ax.set_yscale('log')
ax.set_ylim(bottom=1e-5)
ax.set_xlim(-1, 30) # Show the first 30 most powerful modes

# Correctly generate labels for the top 4 modes
top_mode_indices = sorted_indices[:4]
top_modes_labels = [f'$LG_{{{mode_keys_addmodes[i][0]},{mode_keys_addmodes[i][1]}}}$' for i in top_mode_indices]
ax.set_xticks(indices[:4])

# Apply robust rotation fix for labels
ax.set_xticklabels(top_modes_labels, rotation=90, ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Figure_Mismatch_Diagnostic.png", dpi=300)
plt.show()

print("Script finished. Figure saved as Figure_Mismatch_Diagnostic.png")