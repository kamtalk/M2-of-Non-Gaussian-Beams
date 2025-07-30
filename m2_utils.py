# /lib/m2_utils.py (CORRECTED)

import numpy as np
from scipy.fft import fft2, fftshift, ifftshift
import math
import logging

logger = logging.getLogger(__name__)
WAVELENGTH = 1.0 # Global constant for M2 calculation

def calculate_m2_spatial(psi, grid_params):
    """
    Calculates M2 using the spatial/FFT moment-based method (ISO 11146).
    This is the "measured" M2 for reference beams and the "fitted" M2 for
    perturbation model reconstructions.
    """
    dx, dy = grid_params['dx'], grid_params['dy']
    x, y = grid_params['x'], grid_params['y']
    
    if psi is None or not np.all(np.isfinite(psi)): return np.nan, np.nan
    
    I = np.abs(psi)**2
    I_total = np.longdouble(np.sum(I)) * np.longdouble(dx * dy)
    if I_total < 1e-15: return np.nan, np.nan
    
    # Spatial moments
    x_mean = np.longdouble(np.sum(x[np.newaxis, :] * I)) * np.longdouble(dx*dy) / I_total
    y_mean = np.longdouble(np.sum(y[:, np.newaxis] * I)) * np.longdouble(dx*dy) / I_total
    var_x = np.longdouble(np.sum((x[np.newaxis, :] - x_mean)**2 * I)) * np.longdouble(dx*dy) / I_total
    var_y = np.longdouble(np.sum((y[:, np.newaxis] - y_mean)**2 * I)) * np.longdouble(dx*dy) / I_total
    
    # Frequency moments
    psi_fft = fftshift(fft2(ifftshift(psi)))
    I_fft = np.abs(psi_fft)**2
    I_fft_total = np.longdouble(np.sum(I_fft))
    if I_fft_total < 1e-15: return np.nan, np.nan
    
    kx = 2 * np.pi * fftshift(np.fft.fftfreq(len(x), d=dx))
    ky = 2 * np.pi * fftshift(np.fft.fftfreq(len(y), d=dy))
    
    kx_mean = np.longdouble(np.sum(kx[np.newaxis, :] * I_fft)) / I_fft_total
    ky_mean = np.longdouble(np.sum(ky[:, np.newaxis] * I_fft)) / I_fft_total
    var_kx = np.longdouble(np.sum((kx[np.newaxis, :] - kx_mean)**2 * I_fft)) / I_fft_total
    var_ky = np.longdouble(np.sum((ky[:, np.newaxis] - ky_mean)**2 * I_fft)) / I_fft_total

    # Final M2 calculation
    sigma_x, sigma_y = np.sqrt(max(0, var_x)), np.sqrt(max(0, var_y))
    sigma_kx, sigma_ky = np.sqrt(max(0, var_kx)), np.sqrt(max(0, var_ky))
    
    # <<< START CORRECTION >>>
    # The raw calculation is 4 * sigma * sigma_k. This must be normalized by 2*pi/lambda
    # to get the dimensionless M^2 factor as defined in Siegman's work.
    m2_normalization = (2.0 * np.pi / WAVELENGTH)
    Mx2_raw = (4 * np.pi / WAVELENGTH) * sigma_x * sigma_kx
    My2_raw = (4 * np.pi / WAVELENGTH) * sigma_y * sigma_ky

    return max(1.0, Mx2_raw / m2_normalization), max(1.0, My2_raw / m2_normalization)
    # <<< END CORRECTION >>>


def calculate_m2_from_coeffs_simple(coeffs, mode_keys, basis_type):
    """
    Calculates M2 from a simple weighted sum of individual mode M2 factors.
    Only valid for orthogonal HG and LG bases on beams with matching symmetry.
    """
    total_power = np.sum(np.abs(coeffs)**2)
    if total_power < 1e-15: return np.nan, np.nan
    
    if basis_type == 'LG':
        m2_val = 0.0
        for i, (p, l) in enumerate(mode_keys):
            m2_val += (np.abs(coeffs[i])**2 / total_power) * (2*p + abs(l) + 1)
        return m2_val, m2_val
    elif basis_type == 'HG':
        m2x, m2y = 0.0, 0.0
        for i, (n, m) in enumerate(mode_keys):
            power_fraction = np.abs(coeffs[i])**2 / total_power
            m2x += power_fraction * (2*n + 1)
            m2y += power_fraction * (2*m + 1)
        return m2x, m2y
    return np.nan, np.nan
    
def calculate_m2_from_coeffs_robust(coeffs, basis_matrix, grid_params):
    """
    Calculates M2 robustly from coefficients and spatial cross-moments.
    This is the method from the `script_ince_gaussian_fit_FINAL_V4.py` script,
    generalized for use by any analysis.
    """
    total_power = np.sum(np.abs(coeffs)**2)
    if total_power < 1e-15: return np.nan, np.nan
    
    coeffs_conj = np.conj(coeffs)
    coeff_matrix = np.outer(coeffs_conj, coeffs) # c_i^* * c_j
    
    # --- Spatial Moments ---
    # Pre-calculating integrals of the form integral(f(x,y) * psi_i^* * psi_j)
    basis_H_dxdy = basis_matrix.conj().T * grid_params['dx'] * grid_params['dy']
    M_x = basis_H_dxdy @ (grid_params['xx'].flatten()[:, np.newaxis] * basis_matrix)
    M_y = basis_H_dxdy @ (grid_params['yy'].flatten()[:, np.newaxis] * basis_matrix)
    M_x2 = basis_H_dxdy @ (grid_params['xx'].flatten()[:, np.newaxis]**2 * basis_matrix)
    M_y2 = basis_H_dxdy @ (grid_params['yy'].flatten()[:, np.newaxis]**2 * basis_matrix)
    
    # Calculating expectation values <f(x,y)> = sum(c_i^* c_j * M_f)
    x_mean = np.sum(coeff_matrix * M_x).real / total_power
    y_mean = np.sum(coeff_matrix * M_y).real / total_power
    x2_mean = np.sum(coeff_matrix * M_x2).real / total_power
    y2_mean = np.sum(coeff_matrix * M_y2).real / total_power
    
    var_x = x2_mean - x_mean**2
    var_y = y2_mean - y_mean**2

    # --- Frequency Moments via Spatial Derivatives (Parseval's Theorem) ---
    num_modes = basis_matrix.shape[1]
    basis_2d = basis_matrix.T.reshape(num_modes, grid_params['grid_size'], grid_params['grid_size'])
    
    grad_y, grad_x = np.gradient(basis_2d, grid_params['dy'], grid_params['dx'], axis=(1, 2))
    
    dpsidx_matrix = grad_x.reshape(num_modes, -1).T
    dpsidy_matrix = grad_y.reshape(num_modes, -1).T
    
    # M_dxdx = integral( (d_psi_i^*/dx) * (d_psi_j/dx) )
    M_dxdx = dpsidx_matrix.conj().T @ dpsidx_matrix * grid_params['dx'] * grid_params['dy']
    M_dydy = dpsidy_matrix.conj().T @ dpsidy_matrix * grid_params['dx'] * grid_params['dy']

    var_kx = np.sum(coeff_matrix * M_dxdx).real / total_power
    var_ky = np.sum(coeff_matrix * M_dydy).real / total_power

    # Final M2 Calculation
    m2_normalization = (2.0 * np.pi / WAVELENGTH)
    Mx2_raw = (4 * np.pi / WAVELENGTH) * np.sqrt(max(0, var_x) * max(0, var_kx))
    My2_raw = (4 * np.pi / WAVELENGTH) * np.sqrt(max(0, var_y) * max(0, var_ky))
    
    return max(1.0, Mx2_raw / m2_normalization), max(1.0, My2_raw / m2_normalization)

def calculate_r_squared(psi_ref, psi_fit):
    """Calculates the R^2 value for the intensity fit."""
    intensity_ref = np.abs(psi_ref)**2
    intensity_fit = np.abs(psi_fit)**2
    ss_res = np.sum((intensity_ref - intensity_fit)**2)
    ss_tot = np.sum((intensity_ref - np.mean(intensity_ref))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0