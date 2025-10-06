# @title Frameworks_noise_robustness.py
# Revised version to include Monte Carlo analysis for statistical robustness,
# addressing Reviewer 2, Comment 3.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.fft import fft2, fftshift, ifftshift
import time
import math
import logging

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
# SECTION 1: CORE PARAMETERS
# ==============================================================
grid_size = 256
xy_max = 25.0 
w0_gaussian = 2.0  
lg_basis_order = 10 
rcond_solver = 1e-8 

# --- NEW: Monte Carlo simulation parameter ---
N_runs = 50  # Number of runs for each noise level to get statistics

log_noise_levels = np.logspace(-3, -1, 20) 
linear_noise_levels = np.linspace(2e-4, 8e-4, 4)
noise_levels_to_test = np.unique(np.concatenate(([0.0], linear_noise_levels, log_noise_levels)))

# ==============================================================
# SECTION 2: ESSENTIAL FUNCTIONS (Unchanged)
# ==============================================================

def setup_grid(grid_size, xy_max):
    x = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
    y = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    phi_coords = np.arctan2(yy, xx)
    return x, y, xx, yy, rr, phi_coords

def calculate_gaussian_field(w0, xx_grid, yy_grid):
    rr2 = xx_grid**2 + yy_grid**2
    psi_amp = np.exp(-rr2 / (w0**2))
    dx = (2 * xy_max) / (grid_size - 1)
    power = np.sum(psi_amp**2) * dx**2
    return (psi_amp / np.sqrt(power)).astype(np.complex128)

def calculate_m2_spatial_fft(psi, x_coords, y_coords):
    if psi is None or not np.all(np.isfinite(psi)): return np.nan, np.nan
    dx = x_coords[1] - x_coords[0]
    I = np.abs(psi)**2
    I_total = np.sum(I)
    if I_total < 1e-15: return np.nan, np.nan
    x_mean = np.sum(x_coords[np.newaxis, :] * I) / I_total
    y_mean = np.sum(y_coords[:, np.newaxis] * I) / I_total
    var_x = np.sum((x_coords[np.newaxis, :] - x_mean)**2 * I) / I_total
    var_y = np.sum((y_coords[:, np.newaxis] - y_mean)**2 * I) / I_total
    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)
    psi_fft = fftshift(fft2(ifftshift(psi)))
    I_fft = np.abs(psi_fft)**2
    I_fft_total = np.sum(I_fft)
    if I_fft_total < 1e-15: return np.nan, np.nan
    kx = 2 * np.pi * fftshift(np.fft.fftfreq(len(x_coords), d=dx))
    ky = 2 * np.pi * fftshift(np.fft.fftfreq(len(y_coords), d=dx))
    kx_mean = np.sum(kx[np.newaxis, :] * I_fft) / I_fft_total
    ky_mean = np.sum(ky[:, np.newaxis] * I_fft) / I_fft_total
    var_kx = np.sum((kx[np.newaxis, :] - kx_mean)**2 * I_fft) / I_fft_total
    var_ky = np.sum((ky[:, np.newaxis] - ky_mean)**2 * I_fft) / I_fft_total
    sigma_kx = np.sqrt(var_kx)
    sigma_ky = np.sqrt(var_ky)
    Mx2 = 2 * sigma_x * sigma_kx
    My2 = 2 * sigma_y * sigma_ky
    return Mx2, My2

def calculate_lg_field(p, l, w0, rr_grid, phi_grid):
    norm_factor = math.sqrt((2.0 * math.factorial(p)) / (np.pi * math.factorial(p + abs(l)))) / w0
    radial_term = (np.sqrt(2.0) * rr_grid / w0)**abs(l)
    laguerre_poly = genlaguerre(p, abs(l))(2.0 * rr_grid**2 / w0**2)
    gaussian_term = np.exp(-rr_grid**2 / w0**2)
    azimuthal_phase = np.exp(1j * l * phi_grid)
    lg_field = norm_factor * radial_term * laguerre_poly * gaussian_term * azimuthal_phase
    return lg_field.astype(np.complex128)

def generate_lg_basis_set(max_order_sum, w0, rr_grid, phi_grid):
    basis_modes = {}
    indices = [(p, l) for l in range(-max_order_sum, max_order_sum + 1) for p in range((max_order_sum - abs(l)) // 2 + 1) if 2*p + abs(l) <= max_order_sum]
    logger.info(f"Generating LG basis (2p+|l| <= {max_order_sum}) with {len(indices)} modes...")
    for p, l in indices:
        basis_modes[(p, l)] = calculate_lg_field(p, l, w0, rr_grid, phi_grid)
    return basis_modes

def calculate_m2_from_coeffs_lg(coeffs_complex, mode_keys):
    total_power = np.sum(np.abs(coeffs_complex)**2)
    if total_power < 1e-15: return np.nan
    m2_val = 0.0
    for i, (p, l) in enumerate(mode_keys):
        power_fraction = np.abs(coeffs_complex[i])**2 / total_power
        m2_val += power_fraction * (2 * p + abs(l) + 1)
    return m2_val

# ==============================================================
# SECTION 3: REVISED MAIN EXPERIMENT LOOP WITH MONTE CARLO
# ==============================================================

logger.info("Setting up grid and pre-calculating basis set...")
x_coords, y_coords, xx, yy, rr, phi = setup_grid(grid_size, xy_max)
psi_ideal_gaussian = calculate_gaussian_field(w0_gaussian, xx, yy)
basis_modes = generate_lg_basis_set(lg_basis_order, w0_gaussian, rr, phi)

mode_keys = list(basis_modes.keys())
psi_modes_flat = [basis_modes[k].flatten() for k in mode_keys]
Psi_matrix = np.stack(psi_modes_flat, axis=1)

# Lists to store the final statistical results (mean and std dev)
mean_traditional = []
std_traditional = []
mean_framework = []
std_framework = []

logger.info(f"Starting M² vs. Noise sweep for {len(noise_levels_to_test)} levels, with {N_runs} runs per level...")
for i, noise_std in enumerate(noise_levels_to_test):
    # Temporary lists to store results for the current noise level
    per_level_trad_m2s = []
    per_level_framework_m2s = []

    if noise_std == 0.0:
        # For zero noise, there's no randomness, so only run once
        psi_current = psi_ideal_gaussian
        Mx2_trad, My2_trad = calculate_m2_spatial_fft(psi_current, x_coords, y_coords)
        y_vector = psi_current.flatten()
        coeffs, _, _, _ = np.linalg.lstsq(Psi_matrix, y_vector, rcond=rcond_solver)
        m2_coeff = calculate_m2_from_coeffs_lg(coeffs, mode_keys)
        
        # Append the single deterministic result
        per_level_trad_m2s.append(Mx2_trad)
        per_level_framework_m2s.append(m2_coeff)
        
    else:
        # For non-zero noise, run the Monte Carlo simulation
        for run_num in range(N_runs):
            noise_real = np.random.normal(loc=0.0, scale=noise_std, size=psi_ideal_gaussian.shape)
            noise_imag = np.random.normal(loc=0.0, scale=noise_std, size=psi_ideal_gaussian.shape)
            psi_noisy = psi_ideal_gaussian + (noise_real + 1j * noise_imag)

            Mx2_trad, My2_trad = calculate_m2_spatial_fft(psi_noisy, x_coords, y_coords)
            per_level_trad_m2s.append(Mx2_trad) 

            y_vector = psi_noisy.flatten()
            coeffs, _, _, _ = np.linalg.lstsq(Psi_matrix, y_vector, rcond=rcond_solver)
            m2_coeff = calculate_m2_from_coeffs_lg(coeffs, mode_keys)
            per_level_framework_m2s.append(m2_coeff)

    # Calculate statistics for the current noise level
    mean_trad = np.nanmean(per_level_trad_m2s)
    std_trad = np.nanstd(per_level_trad_m2s)
    mean_fw = np.nanmean(per_level_framework_m2s)
    std_fw = np.nanstd(per_level_framework_m2s)

    # Append stats to the main results lists
    mean_traditional.append(mean_trad)
    std_traditional.append(std_trad)
    mean_framework.append(mean_fw)
    std_framework.append(std_fw)

    logger.info(f"Level {i+1}/{len(noise_levels_to_test)} (Noise Std={noise_std:.4f}): "
                f"Trad: {mean_trad:.2f} ± {std_trad:.2f}, "
                f"Framework: {mean_fw:.4f} ± {std_fw:.4f}")

# ==============================================================
# SECTION 4: REVISED PLOTTING WITH ERROR BARS
# ==============================================================

logger.info("Plotting results with error bars...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# --- PLOT THE DATA USING ERRORBAR ---
# This plots the mean value as a marker and the standard deviation as an error bar.
ax.errorbar(noise_levels_to_test, mean_traditional, yerr=std_traditional, 
            fmt='o', markersize=5, color='crimson', capsize=4, elinewidth=1.5,
            label='Traditional Method (Spatial/FFT)')
            
ax.errorbar(noise_levels_to_test, mean_framework, yerr=std_framework,
            fmt='s', markersize=5, color='royalblue', capsize=4, elinewidth=1.5,
            label=f'Framework (AddModes LG, M={lg_basis_order})')

# --- AXIS FORMATTING ---
ax.set_yscale('log')
ax.set_xscale('symlog', linthresh=1e-3) 
ax.set_xlim(left=-0.0001)

ax.set_xlabel('Noise Level (Standard Deviation of Complex Noise)', fontsize=12)
ax.set_ylabel(r'$M^2$ Factor (Log Scale)', fontsize=12)
ax.set_title(r'Framework Noise Robustness: $M^2$ vs. Noise Level (with Error Bars)', fontsize=14, fontweight='bold')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label=r'Ideal $M^2=1$')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("M2_vs_Noise_with_ErrorBars.png", dpi=300)
plt.show()

logger.info("Script finished.")