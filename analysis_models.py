# @file: analysis_models.py
# Contains the main analysis models: Perturbation and Additive Modal Decomposition.
# CORRECTED VERSION 2

import numpy as np
import math
from scipy.special import genlaguerre, hermite
from scipy.optimize import curve_fit

from m2_utils import calculate_m2_spatial_fft, setup_grid

# --- HELPER FUNCTIONS ---
def _calculate_r_squared(y_true, y_pred):
    """Calculates R^2 for complex fields based on intensity."""
    I_true = np.abs(y_true)**2
    I_pred = np.abs(y_pred)**2
    ss_res = np.sum((I_true - I_pred)**2)
    ss_tot = np.sum((I_true - np.mean(I_true))**2)
    return 1 - (ss_res / ss_tot)

def _generate_poly_basis(x, y, order):
    """Generates a basis of 2D monomials."""
    xx, yy = np.meshgrid(x, y)
    basis = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i == 0 and j == 0: continue
            basis.append((xx**i * yy**j).flatten())
    return np.stack(basis, axis=-1)

# --- PERTURBATION MODEL ---
def model_mult_poly(psi_data, grid_coords, w0_base, poly_order):
    """Implements the Multiplicative Polynomial Perturbation model."""
    x, y = grid_coords
    xx, yy = np.meshgrid(x, y)
    
    # Base function
    psi_base = np.exp(-(xx**2 + yy**2) / w0_base**2)
    
    # Define the model function for curve_fit
    def model_func(xy_flat, *params):
        # xy_flat is a placeholder, we use global xx, yy
        C = params[0] + 1j*params[1]
        coeffs_real = np.array(params[2:len(params)//2+1])
        coeffs_imag = np.array(params[len(params)//2+1:])
        
        poly_term_real = 1.0
        poly_term_imag = 0.0
        
        idx = 0
        for i in range(poly_order + 1):
            for j in range(poly_order + 1 - i):
                if i==0 and j==0: continue
                term = xx**i * yy**j
                poly_term_real += coeffs_real[idx] * term
                poly_term_imag += coeffs_imag[idx] * term
                idx += 1
                
        poly_term = poly_term_real + 1j*poly_term_imag
        
        psi_model = C * psi_base * poly_term
        return np.concatenate([psi_model.real.flatten(), psi_model.imag.flatten()])

    # Initial guess
    num_poly_coeffs = (poly_order+1)*(poly_order+2)//2 - 1
    p0 = [1.0, 0.0] + [0.0]*num_poly_coeffs + [0.0]*num_poly_coeffs

    # Flatten data for fitting
    y_data_flat = np.concatenate([psi_data.real.flatten(), psi_data.imag.flatten()])
    
    try:
        popt, _ = curve_fit(model_func, np.zeros_like(y_data_flat), y_data_flat, p0=p0, method='trf')
        y_fit_flat = model_func(None, *popt)
        psi_reconstructed = (y_fit_flat[:y_data_flat.size//2] + 1j*y_fit_flat[y_data_flat.size//2:]).reshape(psi_data.shape)
    except RuntimeError:
        psi_reconstructed = np.zeros_like(psi_data)

    r_squared = _calculate_r_squared(psi_data, psi_reconstructed)
    m2 = calculate_m2_spatial_fft(psi_reconstructed, x, y)
    
    return {'psi_reconstructed': psi_reconstructed, 'r_squared_intensity': r_squared, 'm2_spatial': m2}

# --- ADDITIVE MODAL DECOMPOSITION MODEL ---
def _calculate_lg_field(p, l, w0, rr, phi):
    norm_factor = math.sqrt((2.0 * math.factorial(p)) / (np.pi * math.factorial(p + abs(l)))) / w0
    radial_term = (np.sqrt(2.0) * rr / w0)**abs(l)
    laguerre_poly = genlaguerre(p, abs(l))(2.0 * rr**2 / w0**2)
    gaussian_term = np.exp(-rr**2 / w0**2)
    azimuthal_phase = np.exp(1j * l * phi)
    return norm_factor * radial_term * laguerre_poly * gaussian_term * azimuthal_phase

def model_add_modes(psi_data, grid_coords, w0_basis, max_order, basis_type='lg', rcond=1e-8):
    """Implements the Additive Modal Decomposition model."""
    x, y = grid_coords
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Directly compute rr and phi from the provided x and y coordinates
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    # --- END OF CORRECTION ---
    
    # Generate basis set
    basis_modes = {}
    if basis_type == 'lg':
        indices = [(p, l) for l in range(-max_order, max_order + 1) for p in range((max_order - abs(l)) // 2 + 1) if 2*p + abs(l) <= max_order]
        for p, l in indices:
            basis_modes[(p, l)] = _calculate_lg_field(p, l, w0_basis, rr, phi)
    else:
        # Placeholder for HG basis if needed
        raise NotImplementedError("Only LG basis is implemented in this example.")
        
    mode_keys = list(basis_modes.keys())
    psi_modes_flat = [basis_modes[k].flatten() for k in mode_keys]
    Psi_matrix = np.stack(psi_modes_flat, axis=1)
    
    # Solve for coefficients
    y_vector = psi_data.flatten()
    coeffs, _, _, _ = np.linalg.lstsq(Psi_matrix, y_vector, rcond=rcond)
    
    # Reconstruct field
    psi_reconstructed = (Psi_matrix @ coeffs).reshape(psi_data.shape)
    
    # Calculate outputs
    r_squared = _calculate_r_squared(psi_data, psi_reconstructed)
    
    total_power = np.sum(np.abs(coeffs)**2)
    m2_val = 0.0
    for i, (p, l) in enumerate(mode_keys):
        power_fraction = np.abs(coeffs[i])**2 / total_power
        m2_val += power_fraction * (2 * p + abs(l) + 1)
        
    return {
        'psi_reconstructed': psi_reconstructed,
        'r_squared_intensity': r_squared,
        'm2_coeffs': (m2_val, m2_val), # (Mx, My)
        'coeffs': coeffs,
        'mode_keys': mode_keys
    }