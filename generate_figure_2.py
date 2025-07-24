import numpy as np
import matplotlib.pyplot as plt # Essential import
from scipy.special import hermite # For Hermite polynomials in HG modes
from numpy.linalg import lstsq # For direct solver
import matplotlib.colors as mcolors
import math

# --- 1. Define Grid and Beam Parameters ---
grid_size = 256
xy_max = 15.0 # Full extent
w0 = 3.0
P2 = 2.0
P4 = 1.0

# --- 2. Create the Spatial Grid ---
x = np.linspace(-xy_max, xy_max, grid_size)
y = np.linspace(-xy_max, xy_max, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

# --- 3. Generate the Original Complex Field ---
gaussian_base = np.exp(-(R/w0)**2)
aberration_phase = P2 * (R/w0)**2 + P4 * (R/w0)**4
aberration_field = np.exp(1j * aberration_phase)
psi_data = gaussian_base * aberration_field
print("Original TEM00 (Ab Strong) beam generated.")

# --- 4. Function to Generate a Single HG Mode ---
def psi_nm_HG(x, y, n, m, w0):
    try:
        norm_val = math.factorial(n) * math.factorial(m)
        if norm_val <= 0: norm_factor = 0
        else: norm_factor = 1 / (w0 * np.sqrt(np.pi * 2**(n+m) * norm_val))
    except ValueError: norm_factor = 0
    if norm_factor == 0: return np.zeros_like(x, dtype=complex)
    xi = np.sqrt(2) * x / w0
    eta = np.sqrt(2) * y / w0
    Hn = hermite(n)(xi)
    Hm = hermite(m)(eta)
    field = norm_factor * Hn * Hm * np.exp(-(x**2 + y**2) / w0**2)
    return field

# --- 5. Define and Generate the HG Basis ---
max_mode_sum_fit = 6
hg_modes = []
mode_indices = []
print(f"Generating HG basis with n+m <= {max_mode_sum_fit}...")
for n in range(max_mode_sum_fit + 1):
    for m in range(max_mode_sum_fit + 1 - n):
        hg_modes.append(psi_nm_HG(X, Y, n, m, w0))
        mode_indices.append((n, m))
num_modes_fit = len(hg_modes)
print(f"Generated {num_modes_fit} HG modes for fitting (n+m <= {max_mode_sum_fit}).")

# --- 6. Prepare Data for lstsq Solver ---
Psi_matrix = np.array([mode.flatten() for mode in hg_modes]).T
psi_data_vector = psi_data.flatten()

# --- 7. Solve for Coefficients ---
print("Solving for coefficients using lstsq...")
coefficients, residuals, rank, s = lstsq(Psi_matrix, psi_data_vector, rcond=1e-8)
print(f"lstsq rank: {rank}/{num_modes_fit}")

# --- 8. Reconstruct the Complex Field ---
psi_reconstructed = (Psi_matrix @ coefficients).reshape(grid_size, grid_size)
original_intensity = np.abs(psi_data)**2
reconstructed_intensity = np.abs(psi_reconstructed)**2
ss_res = np.sum((original_intensity - reconstructed_intensity)**2)
ss_tot = np.sum((original_intensity - np.mean(original_intensity))**2)
r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
print(f"Intensity R^2: {r_squared:.4f}")

# --- 9. Calculate Modal Power Spectrum & M2 ---
modal_power = np.abs(coefficients)**2
total_power_fitted = np.sum(modal_power)
if total_power_fitted < 1e-12:
    normalized_modal_power = np.zeros_like(modal_power)
    m2x_coeffs_calc, m2y_coeffs_calc = np.nan, np.nan
else:
    normalized_modal_power = modal_power / total_power_fitted
    m2x_coeffs_calc = np.sum(modal_power * np.array([2*n + 1 for n,m in mode_indices])) / total_power_fitted
    m2y_coeffs_calc = np.sum(modal_power * np.array([2*m + 1 for n,m in mode_indices])) / total_power_fitted
print(f"Fitted M^2 (from HG coefficients): M2x={m2x_coeffs_calc:.3f}, M2y={m2y_coeffs_calc:.3f}")

# --- 10. Final Plotting (Zoom 3x on A, C; Zoom 2x (Half Scale) on B) ---

fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={'width_ratios': [1, 1.2]})
fig.subplots_adjust(wspace=0.3, hspace=0.4)

# Define the zoom extents
zoom_extent_3x = xy_max / 3.0 # For panels A and C -> +/- 5.0
zoom_extent_2x = xy_max / 2.0 # For panel B -> +/- 7.5

# Panel A: Original Intensity (Zoomed 3x)
ax1 = axes[0, 0]
im1 = ax1.imshow(original_intensity, cmap='viridis', extent=[-xy_max, xy_max, -xy_max, xy_max], origin='lower', interpolation='nearest')
ax1.set_title(r'A) Original Intensity $|\psi_{\text{data}}|^2$')
ax1.set_xlabel('x (a.u.)')
ax1.set_ylabel('y (a.u.)')
# --- Apply 3x zoom to ax1 ---
ax1.set_xlim(-zoom_extent_3x, zoom_extent_3x)
ax1.set_ylim(-zoom_extent_3x, zoom_extent_3x)
plt.colorbar(im1, ax=ax1, label='Intensity (a.u.)', fraction=0.046, pad=0.04)

# Panel B: Original Phase (Zoomed 2x - Half Scale) ### CHANGED ###
ax2 = axes[0, 1]
phase_cmap = mcolors.LinearSegmentedColormap.from_list("hsv_like", plt.cm.hsv(np.linspace(0, 1, 256)))
intensity_mask_threshold = 1e-4 * np.max(original_intensity)
original_phase_masked = np.where(original_intensity < intensity_mask_threshold, np.nan, np.angle(psi_data))
im2 = ax2.imshow(original_phase_masked, cmap=phase_cmap, extent=[-xy_max, xy_max, -xy_max, xy_max], origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
ax2.set_title(r'B) Original Phase $\arg(\psi_{\text{data}})$')
ax2.set_xlabel('x (a.u.)')
ax2.set_ylabel('y (a.u.)')
# --- Apply 2x zoom to ax2 --- ### CHANGED ###
ax2.set_xlim(-zoom_extent_2x, zoom_extent_2x)
ax2.set_ylim(-zoom_extent_2x, zoom_extent_2x)
# --- End Change --- ### CHANGED ###
cbar2 = plt.colorbar(im2, ax=ax2, label='Phase (rad)', ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi], fraction=0.046, pad=0.04)
cbar2.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

# Panel C: Reconstructed Intensity (Zoomed 3x)
ax3 = axes[1, 0]
im3 = ax3.imshow(reconstructed_intensity, cmap='viridis', extent=[-xy_max, xy_max, -xy_max, xy_max], origin='lower', interpolation='nearest')
ax3.set_title(f'C) Reconstructed Intensity $|\\psi_{{\\text{{recon}}}}|^2$ (HG Fit, R$^2$={r_squared:.4f})')
ax3.set_xlabel('x (a.u.)')
ax3.set_ylabel('y (a.u.)')
# --- Apply 3x zoom to ax3 ---
ax3.set_xlim(-zoom_extent_3x, zoom_extent_3x)
ax3.set_ylim(-zoom_extent_3x, zoom_extent_3x)
plt.colorbar(im3, ax=ax3, label='Intensity (a.u.)', fraction=0.046, pad=0.04)

# Panel D: HG Modal Power Spectrum (No Zoom - remains the same)
ax4 = axes[1, 1]
sorted_indices = sorted(range(num_modes_fit), key=lambda k: (mode_indices[k][0] + mode_indices[k][1], mode_indices[k][0], mode_indices[k][1]))
sorted_power = normalized_modal_power[sorted_indices]
sorted_mode_indices = [mode_indices[k] for k in sorted_indices]
x_pos = np.arange(num_modes_fit)
bars = ax4.bar(x_pos, sorted_power)
ax4.set_title(r'D) HG Modal Power Spectrum $|c_{nm}|^2$ (Normalized)')
ax4.set_ylabel('Normalized Power')
ax4.grid(axis='y', linestyle='--')

# --- Labeling Logic (Unchanged) ---
num_top_power_labels = 7
labeled_mode_indices = set()
fundamental_mode = (0,0)
if fundamental_mode in mode_indices:
    try:
        fundamental_original_idx = mode_indices.index(fundamental_mode)
        labeled_mode_indices.add(fundamental_mode)
    except ValueError: pass
top_indices_original = np.argsort(normalized_modal_power)[::-1]
power_labeled_count = 0
for original_idx in top_indices_original:
    mode_idx_tuple = mode_indices[original_idx]
    if mode_idx_tuple == fundamental_mode: continue
    if power_labeled_count >= num_top_power_labels: break
    if mode_idx_tuple not in labeled_mode_indices:
         labeled_mode_indices.add(mode_idx_tuple)
         power_labeled_count += 1
text_offset_above = 0.015
text_offset_inside = 0.05
min_height_for_inside_label = 0.08
for i in range(num_modes_fit):
    n, m = sorted_mode_indices[i]
    if (n, m) in labeled_mode_indices:
        label_text = f'({n},{m})'
        height = sorted_power[i]
        if height > min_height_for_inside_label:
            ax4.text(i, height - text_offset_inside, label_text, ha='center', va='top', rotation=90, fontsize=7, color='white', weight='bold')
        else:
            ax4.text(i, height + text_offset_above, label_text, ha='center', va='bottom', rotation=90, fontsize=7, color='black', weight='bold')

# --- Remove X-axis ticks and labels for spectrum ---
ax4.xaxis.set_major_locator(plt.NullLocator())
ax4.xaxis.set_major_formatter(plt.NullFormatter())
ax4.xaxis.set_minor_locator(plt.NullLocator())
ax4.xaxis.set_minor_formatter(plt.NullFormatter())
ax4.set_xlabel('Mode Index (Sorted by $n+m$; key modes labeled)')
# --- End Labeling/Axis Logic ---

# Save the figure
output_filename = 'aberrated_tem00_hg_fit_A3xC3xB2x.png' # Updated name
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Figure saved as {output_filename}")

plt.show()