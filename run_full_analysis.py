# @title script_additive_lg_v12.9.py - Added Noisy Gaussian Test Case
# --- Includes generation and AddModes(LG) fit for a noisy Gaussian beam ---

import numpy as np
from scipy.special import pbdv, genlaguerre, jv, airy, lpmn
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftshift, ifftshift
import time
from functools import partial
import logging
import math
import numpy.linalg as la

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
# SECTION 1: PARAMETERS AND GRID DEFINITION
# ==============================================================
grid_size = 256; xy_max = 15.0
x = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
y = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
xx, yy = np.meshgrid(x, y); rr = np.sqrt(xx**2 + yy**2); rr2 = xx**2 + yy**2
phi_coords = np.arctan2(yy, xx)
wavelength = 1.0
unit_factor = 1.0
logger.info(f"Running simulation with grid size: {grid_size}x{grid_size}")

# --- Parameters for Additive Perturbation Tests ---
w0_base_additive_poly_gauss = 1.5
b_base_additive_poly_gauss = np.sqrt(2) / w0_base_additive_poly_gauss
beta_ag = {'40': 0.0001+0.0j, '04': 0.0001+0.0j, '22': 0.0002+0.0j}; beta_np = {'20': 0.0+0.1j, '02': 0.0+0.1j, 'rr2': -0.05+0.0j}
D_nl = 0.5 + 0.0j; w0_gauss_pert_nl = 0.8

# --- Parameters for other beams ---
tem00_w0 = 1.0; tem13_w0 = 1.0; airy_x0 = 1.0; airy_a = 0.1
bessel_l=1; bessel_w0=1.0; bessel_wg=5.0; sg_N = 10; sg_w0 = 5.0
sg_defocus_coeff_weak = 0.05; sg_defocus_coeff_strong = 0.5
tem00_ab_P2_weak = 0.5; tem00_ab_P4_weak = 0.2; tem00_ab_P2_strong = 2.0; tem00_ab_P4_strong = 1.0
ig_p = 3; ig_m = 1; ig_epsilon = 2.0; ig_w0 = 2.0

# --- Parameters for Noisy Gaussian --- # ADDED
noisy_gauss_w0 = 1.0 # Waist of the underlying Gaussian (match TEM00 Ideal if desired)
noise_level_std_dev = 0.05 # Standard deviation of complex noise (per component)
noise_seed = 42 # Seed for reproducibility
noisy_gauss_lg_max_order = 10  # Max index sum for LG basis (M=10 => 121 modes)

# --- Global Shapes (Used by Perturbation Fits) ---
global_shape_x4 = xx**4; global_shape_y4 = yy**4; global_shape_x2y2 = xx**2 * yy**2
global_shape_x2 = xx**2; global_shape_y2 = yy**2; global_shape_rr2 = rr2

# ==============================================================
# SECTION 2: UTILITY AND BEAM GENERATION FUNCTIONS
# ==============================================================
# ... (All functions unchanged from v12.8) ...
def calculate_m2_refined(psi, x_coords, y_coords, wavelength):
    psi = np.asarray(psi, dtype=np.complex128); dx = x_coords[1]-x_coords[0]; dy = y_coords[1]-y_coords[0]
    if psi is None or not np.all(np.isfinite(psi)): logger.error("ERROR in M2: Input psi is None, NaN or Inf."); return np.nan,np.nan,np.nan,np.nan
    I = np.abs(psi)**2; I_total = np.longdouble(np.sum(I.astype(np.longdouble))) * np.longdouble(dx*dy)
    if I_total < np.finfo(np.longdouble).eps: logger.debug("M2 calc skipped: Zero total intensity."); return np.nan,np.nan,np.nan,np.nan
    I_norm = (I.astype(np.longdouble)/I_total).astype(np.float64)
    x_mean = np.longdouble(np.sum(x_coords[np.newaxis,:].astype(np.longdouble)*I_norm.astype(np.longdouble))) * np.longdouble(dx*dy)
    y_mean = np.longdouble(np.sum(y_coords[:,np.newaxis].astype(np.longdouble)*I_norm.astype(np.longdouble))) * np.longdouble(dy*dx)
    var_x = np.longdouble(np.sum(((x_coords[np.newaxis,:].astype(np.longdouble)-x_mean)**2)*I_norm.astype(np.longdouble))) * np.longdouble(dx*dy)
    var_y = np.longdouble(np.sum(((y_coords[:,np.newaxis].astype(np.longdouble)-y_mean)**2)*I_norm.astype(np.longdouble))) * np.longdouble(dy*dx)
    var_x_f64 = np.float64(max(0,var_x)); var_y_f64 = np.float64(max(0,var_y));
    sigma_x = np.sqrt(var_x_f64); sigma_y = np.sqrt(var_y_f64)
    try:
        psi_shifted_for_fft=ifftshift(psi); psi_fft_unshifted=fft2(psi_shifted_for_fft)
        if not np.all(np.isfinite(psi_fft_unshifted)): logger.error("ERROR in M2: FFT result contains NaN/Inf."); return np.nan, np.nan, sigma_x, sigma_y
        psi_fft=fftshift(psi_fft_unshifted)
    except Exception as e: logger.error(f"ERROR in M2 during FFT: {e}"); return np.nan, np.nan, sigma_x, sigma_y
    I_fft = np.abs(psi_fft)**2; I_fft_total = np.longdouble(np.sum(I_fft.astype(np.longdouble)))
    if I_fft_total < np.finfo(np.longdouble).eps: logger.debug("M2 calc skipped: Zero FFT intensity."); return np.nan,np.nan,sigma_x,sigma_y
    I_fft_norm = (I_fft.astype(np.longdouble)/I_fft_total).astype(np.float64)
    kx = 2*np.pi*fftshift(np.fft.fftfreq(psi.shape[1], dx)).astype(np.float64); ky = 2*np.pi*fftshift(np.fft.fftfreq(psi.shape[0], dy)).astype(np.float64)
    kx_mean=np.longdouble(np.sum(kx[np.newaxis,:].astype(np.longdouble)*I_fft_norm.astype(np.longdouble)))
    ky_mean=np.longdouble(np.sum(ky[:,np.newaxis].astype(np.longdouble)*I_fft_norm.astype(np.longdouble)))
    var_kx=np.longdouble(np.sum(((kx[np.newaxis,:].astype(np.longdouble)-kx_mean)**2)*I_fft_norm.astype(np.longdouble)))
    var_ky=np.longdouble(np.sum(((ky[:,np.newaxis].astype(np.longdouble)-ky_mean)**2)*I_fft_norm.astype(np.longdouble)))
    var_kx_f64=np.float64(max(0,var_kx)); var_ky_f64=np.float64(max(0,var_ky))
    sigma_kx = np.sqrt(var_kx_f64); sigma_ky = np.sqrt(var_ky_f64)
    Mx2_raw=np.nan; My2_raw=np.nan
    if sigma_x > 1e-15 and sigma_kx > 1e-15: Mx2_raw = (4 * np.pi / wavelength) * sigma_x * sigma_kx
    if sigma_y > 1e-15 and sigma_ky > 1e-15: My2_raw = (4 * np.pi / wavelength) * sigma_y * sigma_ky
    m2_normalization = 2.0 * np.pi / wavelength
    Mx2_norm = Mx2_raw / m2_normalization if not np.isnan(Mx2_raw) else np.nan
    My2_norm = My2_raw / m2_normalization if not np.isnan(My2_raw) else np.nan
    Mx2_final = Mx2_norm if np.isnan(Mx2_norm) else max(Mx2_norm, 1.0 - 1e-9)
    My2_final = My2_norm if np.isnan(My2_norm) else max(My2_norm, 1.0 - 1e-9)
    return Mx2_final, My2_final, sigma_x, sigma_y
def calculate_pcf_field(nu_x, nu_y, bx, by, xx_grid, yy_grid):
    x_in = (bx * xx_grid).astype(np.float64); y_in = (by * yy_grid).astype(np.float64); v_in_x = float(nu_x); v_in_y = float(nu_y)
    try: dv_x_grid, _ = pbdv(v_in_x, x_in); dv_y_grid, _ = pbdv(v_in_y, y_in)
    except Exception as e: logger.error(f"Error pbdv nu_x={nu_x}, nu_y={nu_y}: {e}"); return np.full_like(xx_grid, np.nan, dtype=np.complex128)
    pcf_field = dv_x_grid * dv_y_grid; pcf_field[~np.isfinite(pcf_field)] = 0.0; return pcf_field.astype(np.complex128)
def calculate_lg_field(p, l, w0, rr_grid, phi_grid):
    if p < 0: logger.error(f"Error: LG mode p must be non-negative (p={p}, l={l})."); return np.zeros_like(rr_grid, dtype=np.complex128)
    w0_safe = max(abs(w0), 1e-12); norm_factor = 1.0
    try: norm_factor = math.sqrt((2.0 * math.factorial(p)) / (np.pi * math.factorial(p + abs(l)))) / w0_safe
    except ValueError: logger.warning(f"Could not compute factorial for LG norm (p={p}, l={l}). Using norm=1.0.")
    except Exception as e: logger.error(f"Error calculating LG norm factor (p={p}, l={l}): {e}"); return np.zeros_like(rr_grid, dtype=np.complex128)
    radial_power_term = (np.sqrt(2.0) * rr_grid / w0_safe)**abs(l); laguerre_arg = 2.0 * rr_grid**2 / w0_safe**2
    try: laguerre_poly = genlaguerre(p, abs(l))(laguerre_arg)
    except Exception as e: logger.error(f"Error genlaguerre p={p}, l={abs(l)}, max_arg={np.max(laguerre_arg):.2e}: {e}"); return np.zeros_like(rr_grid, dtype=np.complex128)
    gaussian_term = np.exp(-rr_grid**2 / w0_safe**2); azimuthal_phase = np.exp(1j * l * phi_grid)
    lg_field = norm_factor * radial_power_term * laguerre_poly * gaussian_term * azimuthal_phase; lg_field[~np.isfinite(lg_field)] = 0.0; return lg_field.astype(np.complex128)
def calculate_super_gaussian_field(N, w0, amplitude, phase, xx_grid, yy_grid):
    _rr = np.sqrt(xx_grid**2 + yy_grid**2); _w0_safe = max(abs(w0), 1e-9); exponent = ((_rr / _w0_safe)**(2*N)); exponent_clipped = np.clip(exponent, 0, 700)
    psi_amp = np.exp(-exponent_clipped); psi = amplitude * np.exp(1j*phase) * psi_amp; return psi.astype(np.complex128)
def calculate_gaussian_field(w0, amplitude, phase, xx_grid, yy_grid):
    _rr2 = xx_grid**2 + yy_grid**2; _w0_safe = max(abs(w0), 1e-9)
    psi_amp = np.exp(-_rr2 / (_w0_safe**2)); psi = amplitude * np.exp(1j*phase) * psi_amp; return psi.astype(np.complex128)
def calculate_bessel_gauss_field(l, w0, wg, xx_grid, yy_grid):
    rr_g = np.sqrt(xx_grid**2 + yy_grid**2); phi_g = np.arctan2(yy_grid, xx_grid)
    w0_safe = max(abs(w0), 1e-9); wg_safe = max(abs(wg), 1e-9); k_rho = 2.4048 / w0_safe
    try: bessel_part = jv(l, k_rho * rr_g)
    except Exception as e: logger.error(f"Error calculating Bessel function jv(l={l}): {e}"); return np.zeros_like(xx_grid, dtype=np.complex128)
    gauss_part = np.exp(-rr_g**2 / wg_safe**2); phase_part = np.exp(1j * l * phi_g)
    psi = bessel_part * gauss_part * phase_part; psi[~np.isfinite(psi)] = 0.0; return psi.astype(np.complex128)
def calculate_airy_field(x0, a, xx_grid, yy_grid):
    sx = (xx_grid / x0); sy = (yy_grid / x0); airy_arg = sx - (a**2 / 4.0)
    try: ai, _, _, _ = airy(airy_arg)
    except Exception as e: logger.error(f"Error calculating Airy function: {e}"); return np.zeros_like(xx_grid, dtype=np.complex128)
    psi = ai * np.exp(a * sx - (a**3 / 12.0)) * np.exp(-sy**2)
    psi[~np.isfinite(psi)] = 0.0; return psi.astype(np.complex128)
def calculate_ince_gaussian_field(p, m, epsilon, w0, xx_grid, yy_grid):
    logger.warning("Using simplified placeholder for Ince-Gaussian beam generation.")
    w0_safe = max(w0, 1e-9); f = w0_safe * np.sqrt(epsilon / 2.0)
    term1 = np.sqrt((xx_grid + f)**2 + yy_grid**2); term2 = np.sqrt((xx_grid - f)**2 + yy_grid**2)
    xi_arg = (term1 + term2) / (2 * f); eta_arg = (term1 - term2) / (2 * f)
    xi = np.arccosh(np.clip(xi_arg, 1.0 + 1e-15, None)); eta = np.arccos(np.clip(eta_arg, -1.0 + 1e-15, 1.0 - 1e-15)) # Clip slightly inside domain
    gauss_env = np.exp(-(xx_grid**2 + yy_grid**2) / w0_safe**2)
    try:
        Pmn_vals, Pmn_derivs = lpmn(m, p, np.cos(eta)); legendre_eta = Pmn_vals[m, p, :, :]
        Pmn_xi_vals, _ = lpmn(m, p, np.tanh(xi)); legendre_xi = Pmn_xi_vals[m,p,:,:]
    except Exception as e: logger.error(f"Error calculating Legendre functions for IG placeholder: {e}"); return np.zeros_like(xx_grid, dtype=np.complex128)
    psi = gauss_env * legendre_xi * legendre_eta; psi[~np.isfinite(psi)] = 0.0
    power = np.sum(np.abs(psi)**2);
    if power > 1e-12: psi = psi / np.sqrt(power)
    return psi.astype(np.complex128)
def generate_basis_set(basis_type, max_order_sum, w0, xx_grid, yy_grid):
    basis_modes = {}; rr_grid = np.sqrt(xx_grid**2 + yy_grid**2); phi_grid = np.arctan2(yy_grid, xx_grid)
    dx = xx_grid[0,1] - xx_grid[0,0]; dy = yy_grid[1,0] - yy_grid[0,0]; bx = by = np.sqrt(2) / max(w0, 1e-9)
    logger.info(f"Generating {basis_type} basis (w0={w0:.3f}) up to order sum {max_order_sum}...")
    start_time = time.time(); count = 0; skipped_count = 0
    if basis_type.upper() in ['HG', 'PCF']:
        # For HG/PCF, max_order_sum is typically n+m
        indices = [(n, m) for n in range(max_order_sum + 1) for m in range(max_order_sum + 1 - n)]; logger.info(f"HG indices to generate (n+m <= {max_order_sum}): {len(indices)} modes")
        for n, m in indices:
            mode_field = calculate_pcf_field(n, m, bx, by, xx_grid, yy_grid); power = np.sum(np.abs(mode_field)**2) * dx * dy
            if np.all(np.isfinite(mode_field)) and power > 1e-12: basis_modes[(n, m)] = mode_field; count += 1
            else: logger.debug(f"Skipping HG mode ({n},{m}) due to non-finite values or low power ({power:.2e})."); skipped_count += 1
    elif basis_type.upper() == 'LG':
         # For LG, max_order_sum is typically 2p + |l|
         # We generate up to p_max, l_max such that 2p + |l| <= max_order_sum
        indices = [(p, l) for l in range(-max_order_sum, max_order_sum + 1) for p in range((max_order_sum - abs(l)) // 2 + 1) if 2*p + abs(l) <= max_order_sum];
        logger.info(f"LG indices to generate (2p+|l| <= {max_order_sum}): {len(indices)} modes")
        for p, l in indices:
             mode_field = calculate_lg_field(p, l, w0, rr_grid, phi_grid); power = np.sum(np.abs(mode_field)**2) * dx * dy
             if np.all(np.isfinite(mode_field)) and power > 1e-12 : basis_modes[(p, l)] = mode_field; count += 1
             else: logger.debug(f"Skipping LG mode (p={p},l={l}) due to non-finite values or low power ({power:.2e})."); skipped_count += 1
    else: raise ValueError(f"Unknown basis_type: {basis_type}. Choose 'HG'/'PCF' or 'LG'.")
    end_time = time.time(); logger.info(f"Generated {count} valid {basis_type} modes (skipped {skipped_count}) in {end_time - start_time:.2f} seconds.")
    if count == 0: logger.error(f"CRITICAL: No valid {basis_type} modes generated for order {max_order_sum}, w0={w0:.3f}")
    return basis_modes
def calculate_m2_from_coeffs_lg(coeffs_complex, mode_keys):
    if coeffs_complex is None or len(coeffs_complex) != len(mode_keys): logger.error("Invalid coefficients or key mismatch for LG M2 calculation."); return np.nan
    total_power = np.sum(np.abs(coeffs_complex)**2)
    if total_power < 1e-15: logger.warning("Total power from coefficients is near zero. Cannot calculate LG M2."); return np.nan
    coeffs_norm = coeffs_complex / np.sqrt(total_power); m2_val = 0.0
    # M^2 for LG is Sum[ |c_pl|^2 * (2p + |l| + 1) ]
    for i, (p, l) in enumerate(mode_keys): m2_val += np.abs(coeffs_norm[i])**2 * (2*p + abs(l) + 1)
    m2_final = max(m2_val, 1.0 - 1e-9); return m2_val # Return raw value before max(1.0)
def calculate_m2_from_coeffs_hg(coeffs_complex, mode_keys):
    if coeffs_complex is None or len(coeffs_complex) != len(mode_keys): logger.error("Invalid coefficients or key mismatch for HG M2 calculation."); return np.nan, np.nan
    total_power = np.sum(np.abs(coeffs_complex)**2)
    if total_power < 1e-15: logger.warning("Total power from coefficients is near zero. Cannot calculate HG M2."); return np.nan, np.nan
    coeffs_norm = coeffs_complex / np.sqrt(total_power); m2x_val = 0.0; m2y_val = 0.0
     # M^2_x for HG is Sum[ |c_nm|^2 * (2n + 1) ], M^2_y is Sum[ |c_nm|^2 * (2m + 1) ]
    for i, (n, m) in enumerate(mode_keys): power_fraction = np.abs(coeffs_norm[i])**2; m2x_val += power_fraction * (2*n + 1); m2y_val += power_fraction * (2*m + 1)
    # Don't apply max(1.0) here, return raw calculation
    # m2x_final = max(m2x_val, 1.0 - 1e-9); m2y_final = max(m2y_val, 1.0 - 1e-9); return m2x_final, m2y_final
    return m2x_val, m2y_val

# ==============================================================
# SECTION 3: FITTING MODELS (PERTURBATION + ADDMODES WRAPPERS)
# ==============================================================
# ... (All functions unchanged from v12.8) ...
def hybrid_pcf_fixedbase_mult_poly_complex_fit(coords, C_re, C_im, a40r, a40i, a04r, a04i, a22r, a22i, a20r, a20i, a02r, a02i, a_rr2r, a_rr2i, psi_base_fixed_in, shape_x4_in, shape_y4_in, shape_x2y2_in, shape_x2_in, shape_y2_in, shape_rr2_in):
    _psi_relative_pert = ( (a40r + 1j*a40i) * shape_x4_in + (a04r + 1j*a04i) * shape_y4_in + (a22r + 1j*a22i) * shape_x2y2_in + (a20r + 1j*a20i) * shape_x2_in + (a02r + 1j*a02i) * shape_y2_in + (a_rr2r + 1j*a_rr2i) * shape_rr2_in )
    _psi_relative = 1.0 + _psi_relative_pert; _psi_hybrid_flat = psi_base_fixed_in * _psi_relative; _psi_model_output_flat = (C_re + 1j*C_im) * _psi_hybrid_flat; return _psi_model_output_flat
def hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit(coords, *params, **precalc_args):
    complex_output_flat=hybrid_pcf_fixedbase_mult_poly_complex_fit(coords,*params,**precalc_args);
    if complex_output_is_bad(complex_output_flat, grid_size): return np.full(coords.shape[1]*2, 1e30)
    return np.concatenate([np.real(complex_output_flat), np.imag(complex_output_flat)])
def hybrid_fixedbase_add_poly_complex_fit(coords, C_re, C_im, beta40r, beta40i, beta04r, beta04i, beta22r, beta22i, beta20r, beta20i, beta02r, beta02i, betarr2r, betarr2i, psi_base_fixed_in, shape_x4_in, shape_y4_in, shape_x2y2_in, shape_x2_in, shape_y2_in, shape_rr2_in):
    _psi_poly_pert = ( (beta40r + 1j*beta40i) * shape_x4_in + (beta04r + 1j*beta04i) * shape_y4_in + (beta22r + 1j*beta22i) * shape_x2y2_in + (beta20r + 1j*beta20i) * shape_x2_in + (beta02r + 1j*beta02i) * shape_y2_in + (betarr2r + 1j*betarr2i) * shape_rr2_in )
    _psi_base_scaled = (C_re + 1j*C_im) * psi_base_fixed_in; _psi_model_output_flat = _psi_base_scaled + _psi_poly_pert; return _psi_model_output_flat
def hybrid_fixedbase_add_gauss_complex_fit(coords, C_re, C_im, D_re, D_im, psi_base_fixed_in, psi_gauss_pert_fixed_shape_in):
    _psi_base_scaled = (C_re + 1j*C_im) * psi_base_fixed_in; _psi_gauss_pert_scaled = (D_re + 1j*D_im) * psi_gauss_pert_fixed_shape_in
    _psi_model_output_flat = _psi_base_scaled + _psi_gauss_pert_scaled; return _psi_model_output_flat
def complex_output_is_bad(complex_output_flat, grid_size):
    expected_shape = grid_size*grid_size
    if complex_output_flat is None or not isinstance(complex_output_flat, np.ndarray) or complex_output_flat.shape is None or complex_output_flat.shape[0]!=expected_shape: return True
    if not np.all(np.isfinite(complex_output_flat)): return True; return False
def hybrid_fixedbase_add_poly_complex_fit_wrapper_for_real_fit(coords, *params, **precalc_args):
    complex_output_flat=hybrid_fixedbase_add_poly_complex_fit(coords,*params,**precalc_args);
    if complex_output_is_bad(complex_output_flat, grid_size): return np.full(coords.shape[1]*2, 1e30)
    return np.concatenate([np.real(complex_output_flat), np.imag(complex_output_flat)])
def hybrid_fixedbase_add_gauss_complex_fit_wrapper_for_real_fit(coords, *params, **precalc_args):
    complex_output_flat=hybrid_fixedbase_add_gauss_complex_fit(coords,*params,**precalc_args);
    if complex_output_is_bad(complex_output_flat, grid_size): return np.full(coords.shape[1]*2, 1e30)
    return np.concatenate([np.real(complex_output_flat), np.imag(complex_output_flat)])


# ==============================================================
# SECTION 4: RECONSTRUCTION FUNCTIONS
# ==============================================================
# (Functions unchanged from v12.8)
def hybrid_fixedbase_mult_poly_complex_recon(params, fixed_params, x_coords, y_coords):
    C_re,C_im,a40r,a40i,a04r,a04i,a22r,a22i,a20r,a20i,a02r,a02i,a_rr2r,a_rr2i=params
    if 'psi_base_fixed_rc' not in fixed_params: raise KeyError("'psi_base_fixed_rc' missing")
    _psi_base_rc=np.asarray(fixed_params['psi_base_fixed_rc']).reshape(grid_size, grid_size);
    if not np.all(np.isfinite(_psi_base_rc)): logger.warning("Warn: NaN/Inf in base MultPoly Recon"); return np.full((grid_size,grid_size), np.nan)
    shape_x4_rc = fixed_params.get('shape_x4_in_rc', global_shape_x4); shape_y4_rc = fixed_params.get('shape_y4_in_rc', global_shape_y4); shape_x2y2_rc = fixed_params.get('shape_x2y2_in_rc', global_shape_x2y2); shape_x2_rc = fixed_params.get('shape_x2_in_rc', global_shape_x2); shape_y2_rc = fixed_params.get('shape_y2_in_rc', global_shape_y2); shape_rr2_rc = fixed_params.get('shape_rr2_in_rc', global_shape_rr2)
    _psi_relative_pert = ((a40r+1j*a40i)*shape_x4_rc + (a04r+1j*a04i)*shape_y4_rc + (a22r+1j*a22i)*shape_x2y2_rc + (a20r+1j*a20i)*shape_x2_rc + (a02r+1j*a02i)*shape_y2_rc + (a_rr2r+1j*a_rr2i)*shape_rr2_rc)
    _psi_relative_rc = 1.0 + _psi_relative_pert; _psi_reconstructed = (C_re + 1j*C_im) * _psi_base_rc * _psi_relative_rc
    _psi_reconstructed[np.isinf(_psi_reconstructed)]=0.0; _psi_reconstructed[np.isnan(_psi_reconstructed)]=0.0; return _psi_reconstructed.astype(np.complex128)
def hybrid_fixedbase_add_poly_complex_recon(params, fixed_params, x_coords, y_coords):
    C_re,C_im,beta40r,beta40i,beta04r,beta04i,beta22r,beta22i,beta20r,beta20i,beta02r,beta02i,betarr2r,betarr2i = params
    if 'psi_base_fixed_rc' not in fixed_params: raise KeyError("'psi_base_fixed_rc' missing")
    _psi_base_rc=np.asarray(fixed_params['psi_base_fixed_rc']).reshape(grid_size, grid_size);
    if not np.all(np.isfinite(_psi_base_rc)): logger.warning("Warn: NaN/Inf in base AddPoly Recon"); return np.full((grid_size,grid_size), np.nan)
    shape_x4_rc = fixed_params.get('shape_x4_in_rc', global_shape_x4); shape_y4_rc = fixed_params.get('shape_y4_in_rc', global_shape_y4); shape_x2y2_rc = fixed_params.get('shape_x2y2_in_rc', global_shape_x2y2); shape_x2_rc = fixed_params.get('shape_x2_in_rc', global_shape_x2); shape_y2_rc = fixed_params.get('shape_y2_in_rc', global_shape_y2); shape_rr2_rc = fixed_params.get('shape_rr2_in_rc', global_shape_rr2)
    _psi_poly_pert = ((beta40r+1j*beta40i)*shape_x4_rc + (beta04r+1j*beta04i)*shape_y4_rc + (beta22r+1j*beta22i)*shape_x2y2_rc + (beta20r+1j*beta20i)*shape_x2_rc + (beta02r+1j*beta02i)*shape_y2_rc + (betarr2r+1j*betarr2i)*shape_rr2_rc)
    _psi_base_scaled = (C_re + 1j*C_im) * _psi_base_rc; _psi_reconstructed = _psi_base_scaled + _psi_poly_pert
    _psi_reconstructed[np.isinf(_psi_reconstructed)]=0.0; _psi_reconstructed[np.isnan(_psi_reconstructed)]=0.0; return _psi_reconstructed.astype(np.complex128)
def hybrid_fixedbase_add_gauss_complex_recon(params, fixed_params, x_coords, y_coords):
    C_re, C_im, D_re, D_im = params
    if 'psi_base_fixed_rc' not in fixed_params or 'psi_gauss_pert_fixed_shape_rc' not in fixed_params: raise KeyError("Missing base or gauss shape")
    _psi_base_rc=np.asarray(fixed_params['psi_base_fixed_rc']).reshape(grid_size, grid_size); _psi_gauss_pert_fixed_shape_rc=np.asarray(fixed_params['psi_gauss_pert_fixed_shape_rc']).reshape(grid_size, grid_size)
    if not np.all(np.isfinite(_psi_base_rc)): logger.warning("Warn: NaN/Inf in base AddGauss Recon"); return np.full((grid_size,grid_size), np.nan)
    if not np.all(np.isfinite(_psi_gauss_pert_fixed_shape_rc)): logger.warning("Warn: NaN/Inf in Gauss shape AddGauss Recon"); return np.full((grid_size,grid_size), np.nan)
    _psi_base_scaled = (C_re + 1j*C_im) * _psi_base_rc; _psi_gauss_pert_scaled = (D_re + 1j*D_im) * _psi_gauss_pert_fixed_shape_rc
    _psi_reconstructed = _psi_base_scaled + _psi_gauss_pert_scaled; _psi_reconstructed[np.isinf(_psi_reconstructed)]=0.0; _psi_reconstructed[np.isnan(_psi_reconstructed)]=0.0;
    return _psi_reconstructed.astype(np.complex128)
def hybrid_add_modes_complex_recon(params, fixed_params, x_coords, y_coords):
    if 'psi_modes_rc' not in fixed_params: raise KeyError("'psi_modes_rc' missing in fixed_params for reconstruction")
    psi_modes_rc=fixed_params['psi_modes_rc'];
    if not isinstance(psi_modes_rc, list) or not all(isinstance(m, np.ndarray) and m.ndim == 2 for m in psi_modes_rc): logger.error("psi_modes_rc should be a list of 2D numpy arrays."); return np.full((grid_size,grid_size), np.nan)
    num_modes=len(psi_modes_rc)
    if len(params) != 2*num_modes: logger.warning(f"Param count mismatch recon AddModes: Expected {2*num_modes}, Got {len(params)}"); return np.full((grid_size,grid_size), np.nan)
    _psi_reconstructed = np.zeros((grid_size,grid_size), dtype=np.complex128)
    for i in range(num_modes):
        C_re=params[2*i]; C_im=params[2*i+1]; mode_field=np.asarray(psi_modes_rc[i]).reshape(grid_size,grid_size)
        if not np.all(np.isfinite(mode_field)): logger.warning(f"Warn: NaN/Inf in basis mode {i} during recon AddModes"); return np.full((grid_size,grid_size), np.nan)
        _psi_reconstructed += (C_re + 1j*C_im) * mode_field
    _psi_reconstructed[np.isinf(_psi_reconstructed)]=0.0; _psi_reconstructed[np.isnan(_psi_reconstructed)]=0.0;
    return _psi_reconstructed.astype(np.complex128)


# ==============================================================
# SECTION 5: FITTING EXECUTION FUNCTIONS (curve_fit & direct)
# ==============================================================
# ... (execute_fit function unchanged from v12.8) ...
def execute_fit(beam_name, model_name, psi_reference, measured_data_dict,
                fit_model_func, p0, bounds, param_names,
                precalc_args_dict, recon_func, recon_fixed_params):
    logger.info(f"--- Starting curve_fit: {beam_name} --- Model: {model_name} ---")
    psi_ref=np.asarray(psi_reference,dtype=np.complex128); results_dict={'R2':np.nan, 'sigx_f':np.nan,'sigy_f':np.nan,'Mx2_f':np.nan,'My2_f':np.nan,'params':None,'fit_status':'Init Failed'}
    results_dict.update(measured_data_dict); intensity_ref_unnorm=np.abs(psi_ref)**2; intensity_max_ref=np.max(intensity_ref_unnorm)
    if not np.all(np.isfinite(psi_ref)): logger.error(f"ERROR: Input psi_reference NaN/Inf for {beam_name}. Skip."); results_dict.update({'fit_status':'Skipped(Invalid Input Psi)'}); results_dict['model_type'] = model_name; return results_dict, False
    if intensity_max_ref < np.finfo(float).eps: logger.warning(f"Ref beam '{beam_name}' zero intensity. Skip."); results_dict['fit_status']='Skipped(Zero Intensity)'; results_dict['model_type'] = model_name; return results_dict, False
    logger.info(f"Using Measured {beam_name}: M2x={measured_data_dict.get('Mx2_m',np.nan):.4f}, M2y={measured_data_dict.get('My2_m',np.nan):.4f}, SigX={measured_data_dict.get('sigx_m',np.nan):.4f}, SigY={measured_data_dict.get('sigy_m',np.nan):.4f}")
    logger.info(f"Preparing for complex fit..."); coords=np.vstack((xx.flatten(),yy.flatten())); psi_ref_flat=psi_ref.flatten(); ydata=np.concatenate([np.real(psi_ref_flat), np.imag(psi_ref_flat)])
    start_time=time.time(); popt=None; fit_ok=False; logger.info(f"Starting curve_fit with {len(p0)} parameters...")
    try:
        precalc_args_for_partial = precalc_args_dict if precalc_args_dict is not None else {}; func_to_fit=partial(fit_model_func, **precalc_args_for_partial)
        if len(p0) != len(param_names): raise ValueError(f"p0 length ({len(p0)}) != param_names length ({len(param_names)})")
        if len(bounds[0]) != len(p0) or len(bounds[1]) != len(p0): raise ValueError(f"Bounds length mismatch with p0 ({len(p0)})")
        max_eval = 100000; fit_tolerance = 1e-9
        popt, pcov = curve_fit(func_to_fit, xdata=coords, ydata=ydata, p0=p0, bounds=bounds, max_nfev=max_eval, method='trf', ftol=fit_tolerance, xtol=fit_tolerance, gtol=fit_tolerance)
        fit_ok=True; results_dict['params']=popt; results_dict['fit_status']='Success'
        on_bounds = np.sum(np.isclose(popt, bounds[0], atol=1e-6)) + np.sum(np.isclose(popt, bounds[1], atol=1e-6))
        if on_bounds > 0: logger.warning(f"{on_bounds} parameter(s) converged at the bounds."); results_dict['fit_status'] = 'Success (At Bounds)'
    except (ValueError, RuntimeError, Exception) as e: results_dict['fit_status']=f"FitError: {type(e).__name__}"; logger.error(f"ERROR during curve_fit for {beam_name} ({model_name}): {e}", exc_info=False); fit_ok=False
    end_time=time.time(); logger.info(f"curve_fit finished in {end_time-start_time:.2f} seconds. Status: {results_dict['fit_status']}")
    if fit_ok and popt is not None:
        logger.info(f"Optimal parameters ({beam_name} - {model_name}):"); max_params_to_print = 10
        if len(popt) == len(param_names):
            param_log_count = 0
            for i, pname in enumerate(param_names):
                if param_log_count < max_params_to_print: logger.info(f"  {pname}: {popt[i]:.6e}"); param_log_count += 1
            if len(popt) > max_params_to_print: logger.info(f"  ... ({len(popt)-max_params_to_print} more parameters not shown)")
        else: logger.warning(f"  Fitted params (count mismatch {len(popt)} vs {len(param_names)}): {popt}")
        logger.info("Reconstructing fitted field..."); psi_fit = None
        try: psi_fit = recon_func(popt, recon_fixed_params, x, y)
        except Exception as e: logger.error(f"ERROR during reconstruction: {e}", exc_info=False); results_dict['fit_status'] = f"ReconError: {e}"
        if psi_fit is None or not np.all(np.isfinite(psi_fit)): logger.warning("WARNING: Fitted field None or NaN/Inf."); results_dict['R2'] = np.nan; results_dict['Mx2_f'],results_dict['My2_f'],results_dict['sigx_f'],results_dict['sigy_f'] = np.nan, np.nan, np.nan, np.nan
        else:
            logger.info("Calculating M2 from fitted field..."); Mx2_f, My2_f, sigx_f, sigy_f = calculate_m2_refined(psi_fit, x, y, wavelength); results_dict.update({'Mx2_f': Mx2_f, 'My2_f': My2_f, 'sigx_f': sigx_f, 'sigy_f': sigy_f})
            logger.info("Calculating R^2 (Intensity Match)...");
            try:
                intensity_fit_unnorm = np.abs(psi_fit)**2; intensity_fit_flat = intensity_fit_unnorm.flatten()
                if not np.all(np.isfinite(intensity_fit_flat)): raise ValueError("NaN/Inf in fitted intensity")
                intensity_data_flat_f64 = intensity_ref_unnorm.flatten().astype(np.float64); intensity_fit_flat_f64 = intensity_fit_flat.astype(np.float64)
                if len(intensity_data_flat_f64) != len(intensity_fit_flat_f64): raise ValueError("Data/fit intensity length mismatch.")
                mean_data = np.mean(intensity_data_flat_f64); ss_res = np.sum((intensity_data_flat_f64 - intensity_fit_flat_f64)**2); ss_tot = np.sum((intensity_data_flat_f64 - mean_data)**2)
                if ss_tot < np.finfo(float).eps: results_dict['R2'] = 1.0 if ss_res < np.finfo(float).eps else 0.0
                else: results_dict['R2'] = 1.0 - (ss_res / ss_tot)
            except Exception as e: logger.error(f"ERROR calculating R^2: {e}", exc_info=False); results_dict['R2']=np.nan
            r2_str = f"{results_dict['R2']:.7f}" if not np.isnan(results_dict['R2']) else "NaN"; logger.info(f"R^2 ({beam_name} - {model_name}) = {r2_str}")
    else: logger.warning("curve_fit failed. Fitted field/M2/R^2 results are N/A."); results_dict['Mx2_f'],results_dict['My2_f'],results_dict['sigx_f'],results_dict['sigy_f'] = np.nan, np.nan, np.nan, np.nan; results_dict['R2'] = np.nan
    logger.info(f"\n--- curve_fit Summary ({beam_name} - {model_name}) ---")
    sigm_str = f"{results_dict.get('sigx_m', np.nan):.4f}, {results_dict.get('sigy_m', np.nan):.4f}" if not np.isnan(results_dict.get('sigx_m', np.nan)) else "N/A"; sigf_str = f"{results_dict.get('sigx_f', np.nan):.4f}, {results_dict.get('sigy_f', np.nan):.4f}" if not np.isnan(results_dict.get('sigx_f', np.nan)) else "N/A"
    m2m_str = f"{results_dict.get('Mx2_m', np.nan):.4f}, {results_dict.get('My2_m', np.nan):.4f}" if not np.isnan(results_dict.get('Mx2_m', np.nan)) else "N/A"; m2f_str = f"{results_dict.get('Mx2_f', np.nan):.4f}, {results_dict.get('My2_f', np.nan):.4f}" if not np.isnan(results_dict.get('Mx2_f', np.nan)) else "N/A"
    r2_sum_str = f"{results_dict.get('R2', np.nan):.7f}" if not np.isnan(results_dict.get('R2', np.nan)) else "N/A"
    logger.info(f"  Measured SigX, SigY: {sigm_str}"); logger.info(f"  Fitted SigX, SigY  : {sigf_str}")
    logger.info(f"  Measured M2x, M2y : {m2m_str}"); logger.info(f"  Fitted M2x, M2y   : {m2f_str} (Spatial)")
    logger.info(f"  R^2 (Intensity)    : {r2_sum_str}"); logger.info(f"  Final curve_fit Stat: {results_dict['fit_status']}")
    logger.info("-" * (len(f"--- curve_fit Summary ({beam_name} - {model_name}) ---")))
    results_dict['model_type'] = model_name; return results_dict, fit_ok


# --- Direct Solver Function --- # (Unchanged from v12.8)
def solve_add_modes_directly(beam_name, model_name, basis_type,
                            psi_reference, measured_data_dict,
                            basis_modes_dict, recon_func, rcond_val=1e-12):
    logger.info(f"--- Starting Direct Solve: {beam_name} --- Model: {model_name} (Basis: {basis_type}, rcond={rcond_val:.1e}) ---")
    psi_ref=np.asarray(psi_reference,dtype=np.complex128); results_direct = {'fit_status': 'Init Failed', 'R2': np.nan,'Mx2_f_spatial': np.nan, 'My2_f_spatial': np.nan,'sigx_f': np.nan, 'sigy_f': np.nan,'Mx2_f_coeff': np.nan, 'My2_f_coeff': np.nan,'params': None, 'params_complex': None, 'rank': np.nan}
    results_direct.update(measured_data_dict)
    if not basis_modes_dict: logger.error("No basis modes provided for direct solve. Skipping."); results_direct['fit_status']='Skipped(No Basis)'; results_direct['model_type'] = model_name; return results_direct
    if not np.all(np.isfinite(psi_ref)): logger.error(f"Input psi NaN/Inf for direct solve. Skip."); results_direct['fit_status'] = 'Skipped(Invalid Input Psi)'; results_direct['model_type'] = model_name; return results_direct
    try:
        mode_keys = list(basis_modes_dict.keys()); psi_modes_list = [basis_modes_dict[k] for k in mode_keys]; psi_modes_in_flat = [m.flatten() for m in psi_modes_list]
        Psi = np.stack(psi_modes_in_flat, axis=1); y = psi_ref.flatten()
        if Psi.shape[0] != y.shape[0]: raise ValueError("Basis mode pixel count != data pixel count")
        logger.info(f"Solving linear system: Psi({Psi.shape}) * c = y({y.shape})")
        start_time = time.time();
        # *** SVD implementation ***
        U, s, Vh = la.svd(Psi, full_matrices=False)
        # Apply regularization threshold
        threshold = rcond_val * s[0] # rcond relative to the largest singular value
        s_inv = np.zeros_like(s) # Initialize inverse singular values
        s_inv[s >= threshold] = 1 / s[s >= threshold] # Only invert singular values above threshold
        Sigma_plus = np.diag(s_inv) # Pseudo-inverse of Sigma
        # Calculate coefficients
        coeffs_c = Vh.conj().T @ Sigma_plus @ U.conj().T @ y

        end_time = time.time();
        # Determine effective rank based on threshold
        effective_rank = np.sum(s >= threshold)
        results_direct['rank'] = effective_rank
        logger.info(f"Direct solve (SVD) finished in {end_time-start_time:.3f} sec. Effective Rank={effective_rank}/{Psi.shape[1]} (rcond={rcond_val:.1e}).")

        results_direct['fit_status'] = 'Success (Direct)'; results_direct['params_complex'] = coeffs_c
        params_for_recon = [];
        for coeff in coeffs_c: params_for_recon.extend([np.real(coeff), np.imag(coeff)])
        results_direct['params'] = np.array(params_for_recon)
        coeff_mags = np.abs(coeffs_c); logger.info(f"Direct coefficient magnitudes: min={np.min(coeff_mags):.2e}, max={np.max(coeff_mags):.2e}, mean={np.mean(coeff_mags):.2e}")
        logger.info(f"Calculating M2 from coefficients ({basis_type} basis)...")
        if basis_type.upper() == 'LG': m2_coeff_val = calculate_m2_from_coeffs_lg(coeffs_c, mode_keys); results_direct['Mx2_f_coeff'] = m2_coeff_val; results_direct['My2_f_coeff'] = m2_coeff_val
        elif basis_type.upper() in ['HG', 'PCF']: m2x_coeff_val, m2y_coeff_val = calculate_m2_from_coeffs_hg(coeffs_c, mode_keys); results_direct['Mx2_f_coeff'] = m2x_coeff_val; results_direct['My2_f_coeff'] = m2y_coeff_val
        else: logger.error(f"Unknown basis_type '{basis_type}' for M2-from-coeffs calculation."); results_direct['Mx2_f_coeff'] = np.nan; results_direct['My2_f_coeff'] = np.nan
        logger.info("Reconstructing field from direct solution...")
        recon_fixed_params = {'psi_modes_rc': psi_modes_list}
        # *** CORRECTED CALL ***
        psi_fit_direct = hybrid_add_modes_complex_recon(params_for_recon, recon_fixed_params, x, y)
        logger.info(f"Checking reconstructed field (Direct): min={np.min(np.abs(psi_fit_direct)):.2e}, max={np.max(np.abs(psi_fit_direct)):.2e}, has_nan={np.any(np.isnan(psi_fit_direct))}, has_inf={np.any(np.isinf(psi_fit_direct))}")
        if psi_fit_direct is None or not np.all(np.isfinite(psi_fit_direct)): logger.error("Direct reconstruction resulted in non-finite values. Skipping Spatial M2/R2 calculation."); results_direct['fit_status'] += '+ReconFail'
        else:
            logger.info("Calculating Spatial M2 from reconstructed field...")
            Mx2_f_d, My2_f_d, sigx_f_d, sigy_f_d = calculate_m2_refined(psi_fit_direct, x, y, wavelength)
            if not np.isnan(Mx2_f_d) and not np.isnan(My2_f_d): results_direct.update({'Mx2_f_spatial': Mx2_f_d, 'My2_f_spatial': My2_f_d, 'sigx_f': sigx_f_d, 'sigy_f': sigy_f_d})
            else: logger.warning("Spatial M2 calculation failed (returned NaN)."); results_direct['fit_status'] += '+SpatialM2Fail'
            logger.info("Calculating R^2 (Intensity Match)...");
            try:
                intensity_fit_direct = np.abs(psi_fit_direct)**2; intensity_ref_unnorm = np.abs(psi_ref)**2; intensity_data_flat_f64 = intensity_ref_unnorm.flatten().astype(np.float64); intensity_fit_flat_f64 = intensity_fit_direct.flatten().astype(np.float64)
                if not np.all(np.isfinite(intensity_fit_flat_f64)): raise ValueError("NaN/Inf in direct fitted intensity")
                mean_data = np.mean(intensity_data_flat_f64); ss_res_d = np.sum((intensity_data_flat_f64 - intensity_fit_flat_f64)**2); ss_tot_d = np.sum((intensity_data_flat_f64 - mean_data)**2)
                if ss_tot_d < np.finfo(float).eps: results_direct['R2'] = 1.0 if ss_res_d < np.finfo(float).eps else 0.0
                else: results_direct['R2'] = 1.0 - (ss_res_d / ss_tot_d)
            except Exception as e: logger.error(f"Error calculating R2 for direct fit: {e}"); results_direct['R2'] = np.nan
        logger.info(f"\n--- Direct Solve Summary ({beam_name} - {model_name} - rcond={rcond_val:.1e}) ---")
        sigm_str = f"{results_direct.get('sigx_m', np.nan):.4f}, {results_direct.get('sigy_m', np.nan):.4f}"; sigf_str = f"{results_direct.get('sigx_f', np.nan):.4f}, {results_direct.get('sigy_f', np.nan):.4f}"
        m2m_str = f"{results_direct.get('Mx2_m', np.nan):.4f}, {results_direct.get('My2_m', np.nan):.4f}"; m2f_sp_str = f"{results_direct.get('Mx2_f_spatial', np.nan):.4f}, {results_direct.get('My2_f_spatial', np.nan):.4f}"
        m2f_co_str = f"{results_direct.get('Mx2_f_coeff', np.nan):.4f}, {results_direct.get('My2_f_coeff', np.nan):.4f}"
        r2_sum_str = f"{results_direct.get('R2', np.nan):.7f}" if not np.isnan(results_direct.get('R2', np.nan)) else "N/A"; rank_str = f"{results_direct.get('rank', np.nan)}/{len(mode_keys)}" if not np.isnan(results_direct.get('rank', np.nan)) else "N/A"
        logger.info(f"  Measured SigX, SigY : {sigm_str}"); logger.info(f"  Fitted Sig (Spatial): {sigf_str}")
        logger.info(f"  Measured M2x, M2y  : {m2m_str}"); logger.info(f"  Fitted M2 (Spatial) : {m2f_sp_str}"); logger.info(f"  Fitted M2 (Coeffs)  : {m2f_co_str}")
        logger.info(f"  SVD Rank            : {rank_str}"); logger.info(f"  R^2 (Intensity)     : {r2_sum_str}"); logger.info(f"  Direct Solve Status : {results_direct['fit_status']}")
        logger.info("-" * (len(f"--- Direct Solve Summary ({beam_name} - {model_name} - rcond={rcond_val:.1e}) ---")))
    except la.LinAlgError as e: logger.error(f"Linear algebra error during direct solve: {e}", exc_info=True); results_direct['fit_status'] = f"LinAlgError: {e}"
    except Exception as e: logger.error(f"Error during direct solve attempt: {e}", exc_info=True); results_direct['fit_status'] = f"Error: {e}"
    results_direct['model_type'] = model_name; return results_direct


# ==============================================================
# SECTION 6: MAIN EXECUTION BLOCK
# ==============================================================
results_table = {}
measured_data_cache = {}

# --- 1. Generate Reference Beams & Calculate Measured Data ---
logger.info("--- Generating Reference Beams ---")
# Multi-Mode (MM)
mm_w0_base = 1.0; mm_b = np.sqrt(2)/mm_w0_base; modes_to_include_gen_mm = [(0,0), (1,0), (0,1)]; true_coeffs_complex_mm = {(0,0):1.0+0.0j,(1,0):0.5+0.0j,(0,1):0.5j};
psi_ref_multimode = np.zeros_like(xx,dtype=np.complex128);
for (n_mm,m_mm), coeff_mm in true_coeffs_complex_mm.items(): psi_ref_multimode += coeff_mm*calculate_pcf_field(n_mm,m_mm,mm_b,mm_b,xx,yy);
# Super-Gaussian (SG) Ideal
sg_N = 10; sg_w0 = 5.0; sg_amplitude = 1.0; sg_phase = 0.0;
psi_ref_sg_ideal = calculate_super_gaussian_field(sg_N, sg_w0, sg_amplitude, sg_phase, xx, yy);
# Super-Gaussian (SG) Defocus Weak
defocus_phase_weak = sg_defocus_coeff_weak * global_shape_rr2; psi_ref_sg_defocus_weak = psi_ref_sg_ideal * np.exp(1j*defocus_phase_weak)
# Super-Gaussian (SG) Defocus Strong
defocus_phase_strong = sg_defocus_coeff_strong * global_shape_rr2; psi_ref_sg_defocus_strong = psi_ref_sg_ideal * np.exp(1j*defocus_phase_strong)
# TEM00 Ideal
tem00_w0 = 1.0; psi_ref_tem00 = calculate_lg_field(0, 0, tem00_w0, rr, phi_coords);
# TEM00 Aberrated Weak
aberration_phase_weak = tem00_ab_P2_weak * (rr/tem00_w0)**2 + tem00_ab_P4_weak * (rr/tem00_w0)**4; psi_ref_tem00_ab_weak = psi_ref_tem00 * np.exp(1j*aberration_phase_weak)
# TEM00 Aberrated Strong
aberration_phase_strong = tem00_ab_P2_strong * (rr/tem00_w0)**2 + tem00_ab_P4_strong * (rr/tem00_w0)**4; psi_ref_tem00_ab_strong = psi_ref_tem00 * np.exp(1j*aberration_phase_strong)
# TEM13 Ideal
tem13_w0 = 1.0; tem13_b = np.sqrt(2)/tem13_w0; psi_ref_tem13 = calculate_pcf_field(1, 3, tem13_b, tem13_b, xx, yy);
# Airy Ideal
airy_x0 = 1.0; airy_a = 0.1; psi_ref_airy = calculate_airy_field(airy_x0, airy_a, xx, yy);
# Bessel Ideal
bessel_l=1; bessel_w0=1.0; bessel_wg=5.0; psi_ref_bessel = calculate_bessel_gauss_field(bessel_l, bessel_w0, bessel_wg, xx, yy);
# AGauss
psi_base_gauss_ag = calculate_gaussian_field(w0_base_additive_poly_gauss, 1.0, 0.0, xx, yy);
psi_pert_ag = ( beta_ag['40']*global_shape_x4 + beta_ag['04']*global_shape_y4 + beta_ag['22']*global_shape_x2y2 ); psi_ref_agauss = psi_base_gauss_ag + psi_pert_ag;
# NP
psi_base_gauss_np = calculate_gaussian_field(w0_base_additive_poly_gauss, 1.0, 0.0, xx, yy);
psi_pert_np = ( beta_np['20']*global_shape_x2 + beta_np['02']*global_shape_y2 + beta_np['rr2']*global_shape_rr2 ); psi_ref_np = psi_base_gauss_np + psi_pert_np;
# NL
psi_base_gauss_nl = calculate_gaussian_field(w0_base_additive_poly_gauss, 1.0, 0.0, xx, yy);
psi_pert_gauss_nl = calculate_gaussian_field(w0_gauss_pert_nl, D_nl, 0.0, xx, yy); psi_ref_nl = psi_base_gauss_nl + psi_pert_gauss_nl;
# Ince-Gaussian Beam
psi_ref_incegauss = calculate_ince_gaussian_field(ig_p, ig_m, ig_epsilon, ig_w0, xx, yy)

# --- Generate Noisy Gaussian Beam --- # ADDED
logger.info(f"--- Generating Noisy Gaussian Beam (w0={noisy_gauss_w0:.3f}, noise_std={noise_level_std_dev:.3f}, seed={noise_seed}) ---")
np.random.seed(noise_seed) # Set seed for reproducibility
# Generate the ideal underlying Gaussian
psi_ideal_gaussian_noisy = calculate_gaussian_field(noisy_gauss_w0, 1.0, 0.0, xx, yy)
# Generate complex Gaussian noise (zero mean, specified std dev)
noise_real = np.random.normal(loc=0.0, scale=noise_level_std_dev, size=(grid_size, grid_size))
noise_imag = np.random.normal(loc=0.0, scale=noise_level_std_dev, size=(grid_size, grid_size))
complex_noise = noise_real + 1j * noise_imag
# Add noise to the ideal beam
psi_ref_noisy_gaussian = psi_ideal_gaussian_noisy + complex_noise
logger.info("Noisy Gaussian beam generated.")

# Calculate and Cache ALL Measured M2
logger.info("--- Calculating Measured M2 Values ---")
beam_refs = { # MODIFIED
    "MultiMode": psi_ref_multimode, "SG(N=10) Ideal": psi_ref_sg_ideal,
    "SG(N=10) Defocus W": psi_ref_sg_defocus_weak, "SG(N=10) Defocus S": psi_ref_sg_defocus_strong,
    "TEM00 Ideal": psi_ref_tem00, "TEM00 Ab Weak": psi_ref_tem00_ab_weak, "TEM00 Ab Strong": psi_ref_tem00_ab_strong,
    "TEM13 Ideal": psi_ref_tem13, "Airy Ideal": psi_ref_airy, "Bessel Ideal": psi_ref_bessel,
    "InceGaussian": psi_ref_incegauss,
    "AGauss": psi_ref_agauss, "NP": psi_ref_np, "NL": psi_ref_nl,
    "Noisy Gaussian": psi_ref_noisy_gaussian # ADDED
}
for name, psi in beam_refs.items():
    if psi is None or not np.all(np.isfinite(psi)):
        logger.error(f"Reference beam '{name}' contains NaN/Inf. Cannot calculate measured M2.")
        measured_data_cache[name] = {'Mx2_m': np.nan, 'My2_m': np.nan, 'sigx_m': np.nan, 'sigy_m': np.nan}
    else:
        mx2, my2, sx, sy = calculate_m2_refined(psi, x, y, wavelength)
        measured_data_cache[name] = {'Mx2_m': mx2, 'My2_m': my2, 'sigx_m': sx, 'sigy_m': sy}
        logger.info(f"Measured {name}: M2x={mx2:.4f}, M2y={my2:.4f}, SigX={sx:.4f}, SigY={sy:.4f}")

# --- 2. Generate Basis Sets Needed ---
logger.info("--- Generating Basis Sets ---")
mm_hg_max_order = 1; basis_modes_mm_hg = generate_basis_set('HG', mm_hg_max_order, mm_w0_base, xx, yy); mm_num_modes_hg = len(basis_modes_mm_hg) if basis_modes_mm_hg else 0
sg_hg_max_order = 10; basis_modes_sg_hg = generate_basis_set('HG', sg_hg_max_order, sg_w0 * 0.9, xx, yy); sg_num_modes_hg = len(basis_modes_sg_hg) if basis_modes_sg_hg else 0
sg_lg_max_order = 10; basis_modes_sg_lg = generate_basis_set('LG', sg_lg_max_order, sg_w0 * 0.9, xx, yy); sg_num_modes_lg = len(basis_modes_sg_lg) if basis_modes_sg_lg else 0
tem00_ideal_lg_max_order = 2; basis_modes_tem00_ideal_lg = generate_basis_set('LG', tem00_ideal_lg_max_order, tem00_w0, xx, yy); tem00_num_modes_ideal_lg = len(basis_modes_tem00_ideal_lg) if basis_modes_tem00_ideal_lg else 0
tem00_ab_max_order = 6
basis_modes_tem00_lg = generate_basis_set('LG', tem00_ab_max_order, tem00_w0 * 0.9, xx, yy); tem00_num_modes_lg = len(basis_modes_tem00_lg) if basis_modes_tem00_lg else 0
basis_modes_tem00_hg = generate_basis_set('HG', tem00_ab_max_order, tem00_w0 * 0.9, xx, yy); tem00_num_modes_hg = len(basis_modes_tem00_hg) if basis_modes_tem00_hg else 0
ig_decomp_max_order = 10
basis_modes_ig_hg = generate_basis_set('HG', ig_decomp_max_order, ig_w0 * 0.9, xx, yy); ig_num_modes_hg = len(basis_modes_ig_hg) if basis_modes_ig_hg else 0
basis_modes_ig_lg = generate_basis_set('LG', ig_decomp_max_order, ig_w0 * 0.9, xx, yy); ig_num_modes_lg = len(basis_modes_ig_lg) if basis_modes_ig_lg else 0

# --- Generate Basis Set for Noisy Gaussian (LG Basis, Order 10) --- # ADDED
# noisy_gauss_lg_max_order is defined in Section 1 as 10
basis_modes_noisy_gauss_lg = generate_basis_set(
    'LG',
    noisy_gauss_lg_max_order, # Should be 10
    noisy_gauss_w0,           # Use the underlying Gaussian waist
    xx, yy
    )
noisy_gauss_lg_num_modes = len(basis_modes_noisy_gauss_lg) if basis_modes_noisy_gauss_lg else 0 # Get actual number of modes generated
logger.info(f"Noisy Gaussian LG basis generated with {noisy_gauss_lg_num_modes} modes (M={noisy_gauss_lg_max_order}).")


# --- 3. Execute Fits ---
logger.info("--- Executing Fits ---")
# --- Perturbation Model Fits (using curve_fit) ---
poly_pert_params = ['C_re','C_im','a40r','a40i','a04r','a04i','a22r','a22i','a20r','a20i','a02r','a02i','a_rr2r','a_rr2i']
poly_pert_p0 = [1.0,0.0] + [0.0]*12; poly_pert_bounds = ([-2,-2]+[-0.1]*12, [2,2]+[0.1]*12)
poly_precalc = {'shape_x4_in':global_shape_x4.flatten(), 'shape_y4_in':global_shape_y4.flatten(), 'shape_x2y2_in':global_shape_x2y2.flatten(), 'shape_x2_in':global_shape_x2.flatten(), 'shape_y2_in':global_shape_y2.flatten(), 'shape_rr2_in':global_shape_rr2.flatten()}
poly_recon = {}

# AGauss, NP, NL Fits
psi_base_gauss_ag = calculate_gaussian_field(w0_base_additive_poly_gauss, 1.0, 0.0, xx, yy)
ag_precalc = {'psi_base_fixed_in':psi_base_gauss_ag.flatten(), **poly_precalc}; ag_recon = {'psi_base_fixed_rc':psi_base_gauss_ag, **poly_recon}
resAG, ok = execute_fit("AGauss", "AddPoly(Gauss Base)", psi_ref_agauss, measured_data_cache["AGauss"], hybrid_fixedbase_add_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, ag_precalc, hybrid_fixedbase_add_poly_complex_recon, ag_recon); results_table[("AGauss", "AddPoly(Gauss Base)")] = resAG
psi_base_gauss_np = calculate_gaussian_field(w0_base_additive_poly_gauss, 1.0, 0.0, xx, yy)
np_precalc = {'psi_base_fixed_in':psi_base_gauss_np.flatten(), **poly_precalc}; np_recon = {'psi_base_fixed_rc':psi_base_gauss_np, **poly_recon}
resNP, ok = execute_fit("NP", "AddPoly(Gauss Base)", psi_ref_np, measured_data_cache["NP"], hybrid_fixedbase_add_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, np_precalc, hybrid_fixedbase_add_poly_complex_recon, np_recon); results_table[("NP", "AddPoly(Gauss Base)")] = resNP
nl_params = ['C_re','C_im','D_re','D_im']; p0_nl = [1.0, 0.0, 0.1, 0.0]; bounds_nl = ([-2,-2,-1,-1], [2,2,1,1])
psi_base_gauss_nl = calculate_gaussian_field(w0_base_additive_poly_gauss, 1.0, 0.0, xx, yy)
psi_pert_gauss_nl = calculate_gaussian_field(w0_gauss_pert_nl, 1.0, 0.0, xx, yy)
nl_precalc = {'psi_base_fixed_in': psi_base_gauss_nl.flatten(), 'psi_gauss_pert_fixed_shape_in': psi_pert_gauss_nl.flatten()}; nl_recon = {'psi_base_fixed_rc': psi_base_gauss_nl, 'psi_gauss_pert_fixed_shape_rc': psi_pert_gauss_nl}
resNL, ok = execute_fit("NL", "AddGauss(Gauss Base)", psi_ref_nl, measured_data_cache["NL"], hybrid_fixedbase_add_gauss_complex_fit_wrapper_for_real_fit, p0_nl, bounds_nl, nl_params, nl_precalc, hybrid_fixedbase_add_gauss_complex_recon, nl_recon); results_table[("NL", "AddGauss(Gauss Base)")] = resNL

# Ideal Beams (MultPoly) Fits
psi_base_tem00_fit = calculate_gaussian_field(tem00_w0, 1.0, 0.0, xx, yy)
tem00_precalc = {'psi_base_fixed_in':psi_base_tem00_fit.flatten(), **poly_precalc}; tem00_recon = {'psi_base_fixed_rc':psi_base_tem00_fit, **poly_recon}
resT00, ok = execute_fit("TEM00 Ideal", "MultPoly(Gauss Base)", psi_ref_tem00, measured_data_cache["TEM00 Ideal"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, tem00_precalc, hybrid_fixedbase_mult_poly_complex_recon, tem00_recon); results_table[("TEM00 Ideal", "MultPoly(Gauss Base)")] = resT00
psi_base_tem13 = calculate_pcf_field(1, 3, np.sqrt(2)/tem13_w0, np.sqrt(2)/tem13_w0, xx, yy)
tem13_precalc = {'psi_base_fixed_in':psi_base_tem13.flatten(), **poly_precalc}; tem13_recon = {'psi_base_fixed_rc':psi_base_tem13, **poly_recon}
resT13, ok = execute_fit("TEM13 Ideal", "MultPoly(HG13 Base)", psi_ref_tem13, measured_data_cache["TEM13 Ideal"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, tem13_precalc, hybrid_fixedbase_mult_poly_complex_recon, tem13_recon); results_table[("TEM13 Ideal", "MultPoly(HG13 Base)")] = resT13
psi_base_airy = calculate_airy_field(airy_x0, airy_a, xx, yy)
airy_precalc = {'psi_base_fixed_in':psi_base_airy.flatten(), **poly_precalc}; airy_recon = {'psi_base_fixed_rc':psi_base_airy, **poly_recon}
resAiry, ok = execute_fit("Airy Ideal", "MultPoly(Airy Base)", psi_ref_airy, measured_data_cache["Airy Ideal"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, airy_precalc, hybrid_fixedbase_mult_poly_complex_recon, airy_recon); results_table[("Airy Ideal", "MultPoly(Airy Base)")] = resAiry
psi_base_bessel = calculate_bessel_gauss_field(bessel_l, bessel_w0, bessel_wg, xx, yy)
bessel_precalc = {'psi_base_fixed_in':psi_base_bessel.flatten(), **poly_precalc}; bessel_recon = {'psi_base_fixed_rc':psi_base_bessel, **poly_recon}
resBessel, ok = execute_fit("Bessel Ideal", "MultPoly(Bessel Base)", psi_ref_bessel, measured_data_cache["Bessel Ideal"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, bessel_precalc, hybrid_fixedbase_mult_poly_complex_recon, bessel_recon); results_table[("Bessel Ideal", "MultPoly(Bessel Base)")] = resBessel
psi_base_sg = calculate_super_gaussian_field(sg_N, sg_w0, 1.0, 0.0, xx, yy)
sg_precalc = {'psi_base_fixed_in':psi_base_sg.flatten(), **poly_precalc}; sg_recon = {'psi_base_fixed_rc':psi_base_sg, **poly_recon}
resSGIdeal, ok = execute_fit("SG(N=10) Ideal", "MultPoly(SG Base)", psi_ref_sg_ideal, measured_data_cache["SG(N=10) Ideal"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, sg_precalc, hybrid_fixedbase_mult_poly_complex_recon, sg_recon); results_table[("SG(N=10) Ideal", "MultPoly(SG Base)")] = resSGIdeal

# SG Defocus Fits (Weak & Strong - MultPoly)
sg_defocus_precalc_w = {'psi_base_fixed_in':psi_base_sg.flatten(), **poly_precalc}; sg_defocus_recon_w = {'psi_base_fixed_rc':psi_base_sg, **poly_recon}
poly_pert_p0_defocus_w = list(poly_pert_p0); poly_pert_bounds_defocus_w = list(poly_pert_bounds); poly_pert_bounds_defocus_w[0][-1] = -0.2; poly_pert_bounds_defocus_w[1][-1] = 0.2
resSGDefocusW, ok = execute_fit("SG(N=10) Defocus W", "MultPoly(SG Base)", psi_ref_sg_defocus_weak, measured_data_cache["SG(N=10) Defocus W"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0_defocus_w, tuple(poly_pert_bounds_defocus_w), poly_pert_params, sg_defocus_precalc_w, hybrid_fixedbase_mult_poly_complex_recon, sg_defocus_recon_w); results_table[("SG(N=10) Defocus W", "MultPoly(SG Base)")] = resSGDefocusW
sg_defocus_precalc_s = {'psi_base_fixed_in':psi_base_sg.flatten(), **poly_precalc}; sg_defocus_recon_s = {'psi_base_fixed_rc':psi_base_sg, **poly_recon}
poly_pert_p0_defocus_s = list(poly_pert_p0); poly_pert_bounds_defocus_s = list(poly_pert_bounds); poly_pert_bounds_defocus_s[0][-1] = -1.0; poly_pert_bounds_defocus_s[1][-1] = 1.0
resSGDefocusS, ok = execute_fit("SG(N=10) Defocus S", "MultPoly(SG Base)", psi_ref_sg_defocus_strong, measured_data_cache["SG(N=10) Defocus S"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0_defocus_s, tuple(poly_pert_bounds_defocus_s), poly_pert_params, sg_defocus_precalc_s, hybrid_fixedbase_mult_poly_complex_recon, sg_defocus_recon_s); results_table[("SG(N=10) Defocus S", "MultPoly(SG Base)")] = resSGDefocusS

# Aberrated TEM00 Fits (MultPoly)
tem00_ab_w_precalc = {'psi_base_fixed_in':psi_base_tem00_fit.flatten(), **poly_precalc}; tem00_ab_w_recon = {'psi_base_fixed_rc':psi_base_tem00_fit, **poly_recon}
resT00AbW_MP, ok = execute_fit("TEM00 Ab Weak", "MultPoly(Gauss Base)", psi_ref_tem00_ab_weak, measured_data_cache["TEM00 Ab Weak"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, tem00_ab_w_precalc, hybrid_fixedbase_mult_poly_complex_recon, tem00_ab_w_recon); results_table[("TEM00 Ab Weak", "MultPoly(Gauss Base)")] = resT00AbW_MP
tem00_ab_s_precalc = {'psi_base_fixed_in':psi_base_tem00_fit.flatten(), **poly_precalc}; tem00_ab_s_recon = {'psi_base_fixed_rc':psi_base_tem00_fit, **poly_recon}
poly_pert_p0_ab_s = list(poly_pert_p0); poly_pert_bounds_ab_s = list(poly_pert_bounds)
poly_pert_bounds_ab_s[0][-1] = -np.pi*2; poly_pert_bounds_ab_s[1][-1] = np.pi*2; poly_pert_bounds_ab_s[0][3] = -np.pi*2; poly_pert_bounds_ab_s[1][3] = np.pi*2; poly_pert_bounds_ab_s[0][5] = -np.pi*2; poly_pert_bounds_ab_s[1][5] = np.pi*2; poly_pert_bounds_ab_s[0][7] = -np.pi*2; poly_pert_bounds_ab_s[1][7] = np.pi*2
resT00AbS_MP, ok = execute_fit("TEM00 Ab Strong", "MultPoly(Gauss Base)", psi_ref_tem00_ab_strong, measured_data_cache["TEM00 Ab Strong"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0_ab_s, tuple(poly_pert_bounds_ab_s), poly_pert_params, tem00_ab_s_precalc, hybrid_fixedbase_mult_poly_complex_recon, tem00_ab_s_recon); results_table[("TEM00 Ab Strong", "MultPoly(Gauss Base)")] = resT00AbS_MP

# Ince-Gaussian Fit (MultPoly HG Base)
psi_base_ig_hg = calculate_pcf_field(2, 1, np.sqrt(2)/ig_w0, np.sqrt(2)/ig_w0, xx, yy)
ig_hg_precalc = {'psi_base_fixed_in':psi_base_ig_hg.flatten(), **poly_precalc}; ig_hg_recon = {'psi_base_fixed_rc':psi_base_ig_hg, **poly_recon}
resIG_MP_HG, ok = execute_fit("InceGaussian", "MultPoly(HG21 Base)", psi_ref_incegauss, measured_data_cache["InceGaussian"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, ig_hg_precalc, hybrid_fixedbase_mult_poly_complex_recon, ig_hg_recon); results_table[("InceGaussian", "MultPoly(HG21 Base)")] = resIG_MP_HG

# Ince-Gaussian Fit (MultPoly LG Base)
psi_base_ig_lg = calculate_lg_field(1, 1, ig_w0, rr, phi_coords) # Example LG(p=1, l=1) has order 2p+|l| = 3
ig_lg_precalc = {'psi_base_fixed_in':psi_base_ig_lg.flatten(), **poly_precalc}; ig_lg_recon = {'psi_base_fixed_rc':psi_base_ig_lg, **poly_recon}
resIG_MP_LG, ok = execute_fit("InceGaussian", "MultPoly(LG11 Base)", psi_ref_incegauss, measured_data_cache["InceGaussian"], hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit, poly_pert_p0, poly_pert_bounds, poly_pert_params, ig_lg_precalc, hybrid_fixedbase_mult_poly_complex_recon, ig_lg_recon); results_table[("InceGaussian", "MultPoly(LG11 Base)")] = resIG_MP_LG


# --- Additive Modal Decomposition Fits (using direct solver) ---
default_rcond = 1e-8
# Multi-Mode (HG Basis)
mm_key = ("MultiMode", f"AddModes(HG, M={mm_hg_max_order})")
if basis_modes_mm_hg: resMM_ds = solve_add_modes_directly("MultiMode", f"AddModes(HG, M={mm_hg_max_order})", "HG", psi_ref_multimode, measured_data_cache["MultiMode"], basis_modes_mm_hg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[mm_key] = resMM_ds
else: results_table[mm_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(HG, M={mm_hg_max_order})", **measured_data_cache["MultiMode"]}
# Super-Gaussian (HG Basis - Order 10)
sg_hg_key = ("SG(N=10) Ideal", f"AddModes(HG, M={sg_hg_max_order})")
if basis_modes_sg_hg: resSG_HG_ds = solve_add_modes_directly("SG(N=10) Ideal", f"AddModes(HG, M={sg_hg_max_order})", "HG", psi_ref_sg_ideal, measured_data_cache["SG(N=10) Ideal"], basis_modes_sg_hg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[sg_hg_key] = resSG_HG_ds
else: results_table[sg_hg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(HG, M={sg_hg_max_order})", **measured_data_cache["SG(N=10) Ideal"]}
# Super-Gaussian (LG Basis - Order 10)
sg_lg_key = ("SG(N=10) Ideal", f"AddModes(LG, M={sg_lg_max_order})")
if basis_modes_sg_lg: resSG_LG_ds = solve_add_modes_directly("SG(N=10) Ideal", f"AddModes(LG, M={sg_lg_max_order})", "LG", psi_ref_sg_ideal, measured_data_cache["SG(N=10) Ideal"], basis_modes_sg_lg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[sg_lg_key] = resSG_LG_ds
else: results_table[sg_lg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(LG, M={sg_lg_max_order})", **measured_data_cache["SG(N=10) Ideal"]}
# TEM00 Ideal (LG Basis - Order 2)
tem00_lg_key = ("TEM00 Ideal", f"AddModes(LG, M={tem00_ideal_lg_max_order})")
if basis_modes_tem00_ideal_lg: resTEM00_LG_ds = solve_add_modes_directly("TEM00 Ideal", f"AddModes(LG, M={tem00_ideal_lg_max_order})", "LG", psi_ref_tem00, measured_data_cache["TEM00 Ideal"], basis_modes_tem00_ideal_lg, hybrid_add_modes_complex_recon, rcond_val=1e-12); results_table[tem00_lg_key] = resTEM00_LG_ds
else: results_table[tem00_lg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(LG, M={tem00_ideal_lg_max_order})", **measured_data_cache["TEM00 Ideal"]}
# Aberrated TEM00 Fits (AddModes HG - Order 6)
tem00_ab_w_hg_key = ("TEM00 Ab Weak", f"AddModes(HG, M={tem00_ab_max_order})")
if basis_modes_tem00_hg: resT00AbW_AM_HG = solve_add_modes_directly("TEM00 Ab Weak", f"AddModes(HG, M={tem00_ab_max_order})", "HG", psi_ref_tem00_ab_weak, measured_data_cache["TEM00 Ab Weak"], basis_modes_tem00_hg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[tem00_ab_w_hg_key] = resT00AbW_AM_HG
else: results_table[tem00_ab_w_hg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(HG, M={tem00_ab_max_order})", **measured_data_cache["TEM00 Ab Weak"]}
tem00_ab_s_hg_key = ("TEM00 Ab Strong", f"AddModes(HG, M={tem00_ab_max_order})")
if basis_modes_tem00_hg: resT00AbS_AM_HG = solve_add_modes_directly("TEM00 Ab Strong", f"AddModes(HG, M={tem00_ab_max_order})", "HG", psi_ref_tem00_ab_strong, measured_data_cache["TEM00 Ab Strong"], basis_modes_tem00_hg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[tem00_ab_s_hg_key] = resT00AbS_AM_HG
else: results_table[tem00_ab_s_hg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(HG, M={tem00_ab_max_order})", **measured_data_cache["TEM00 Ab Strong"]}
# Aberrated TEM00 Fits (AddModes LG - Order 6)
tem00_ab_w_lg_key = ("TEM00 Ab Weak", f"AddModes(LG, M={tem00_ab_max_order})")
if basis_modes_tem00_lg: resT00AbW_AM_LG = solve_add_modes_directly("TEM00 Ab Weak", f"AddModes(LG, M={tem00_ab_max_order})", "LG", psi_ref_tem00_ab_weak, measured_data_cache["TEM00 Ab Weak"], basis_modes_tem00_lg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[tem00_ab_w_lg_key] = resT00AbW_AM_LG
else: results_table[tem00_ab_w_lg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(LG, M={tem00_ab_max_order})", **measured_data_cache["TEM00 Ab Weak"]}
tem00_ab_s_lg_key = ("TEM00 Ab Strong", f"AddModes(LG, M={tem00_ab_max_order})")
if basis_modes_tem00_lg: resT00AbS_AM_LG = solve_add_modes_directly("TEM00 Ab Strong", f"AddModes(LG, M={tem00_ab_max_order})", "LG", psi_ref_tem00_ab_strong, measured_data_cache["TEM00 Ab Strong"], basis_modes_tem00_lg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[tem00_ab_s_lg_key] = resT00AbS_AM_LG
else: results_table[tem00_ab_s_lg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(LG, M={tem00_ab_max_order})", **measured_data_cache["TEM00 Ab Strong"]}
# Ince-Gaussian Fits (AddModes - Order 10)
ig_hg_key = ("InceGaussian", f"AddModes(HG, M={ig_decomp_max_order})")
if basis_modes_ig_hg: resIG_AM_HG = solve_add_modes_directly("InceGaussian", f"AddModes(HG, M={ig_decomp_max_order})", "HG", psi_ref_incegauss, measured_data_cache["InceGaussian"], basis_modes_ig_hg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[ig_hg_key] = resIG_AM_HG
else: results_table[ig_hg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(HG, M={ig_decomp_max_order})", **measured_data_cache["InceGaussian"]}
ig_lg_key = ("InceGaussian", f"AddModes(LG, M={ig_decomp_max_order})")
if basis_modes_ig_lg: resIG_AM_LG = solve_add_modes_directly("InceGaussian", f"AddModes(LG, M={ig_decomp_max_order})", "LG", psi_ref_incegauss, measured_data_cache["InceGaussian"], basis_modes_ig_lg, hybrid_add_modes_complex_recon, rcond_val=default_rcond); results_table[ig_lg_key] = resIG_AM_LG
else: results_table[ig_lg_key] = {'fit_status':'Skipped(No Basis)', 'model_type':f"AddModes(LG, M={ig_decomp_max_order})", **measured_data_cache["InceGaussian"]}

# --- Execute Fit for Noisy Gaussian (AddModes LG - Order 10) --- # ADDED
noisy_gauss_lg_key = ("Noisy Gaussian", f"AddModes(LG, M={noisy_gauss_lg_max_order})")
if basis_modes_noisy_gauss_lg: # Check if basis was generated successfully
    resNoisyGauss_AM_LG = solve_add_modes_directly(
        beam_name="Noisy Gaussian",
        model_name=f"AddModes(LG, M={noisy_gauss_lg_max_order})", # Use M=X format
        basis_type="LG",
        psi_reference=psi_ref_noisy_gaussian,
        measured_data_dict=measured_data_cache["Noisy Gaussian"],
        basis_modes_dict=basis_modes_noisy_gauss_lg,
        recon_func=hybrid_add_modes_complex_recon, # Make sure this function exists
        rcond_val=1e-8 # Use the default rcond for potentially ill-conditioned data
        )
    results_table[noisy_gauss_lg_key] = resNoisyGauss_AM_LG
else:
    # Handle case where basis generation failed
    results_table[noisy_gauss_lg_key] = {
        'fit_status':'Skipped(No Basis)',
        'model_type':f"AddModes(LG, M={noisy_gauss_lg_max_order})",
        **measured_data_cache["Noisy Gaussian"] # Include measured data
    }


# --- 4. Print Results Summary Table ---
logger.info(f"--- Results Summary Table (v12.9 - Max Order={noisy_gauss_lg_max_order} for LG/SG/IG/Noisy, M={tem00_ab_max_order} for AbTEM00) ---")
print("\n\n" + "="*80)
print(f"--- Results Summary Table (v12.9 - Max Order={noisy_gauss_lg_max_order} for LG/SG/IG/Noisy, M={tem00_ab_max_order} for AbTEM00) ---")
print("="*80)
h_beam = "Beam Type"; h_model = "Model Type"; h_r2 = "R^2"; h_sigm = "SigX, SigY (M)"; h_m2m = "M2x, M2y (M)"
h_sigf = "SigX, SigY (F)"; h_m2f_s = "M2x, M2y (F Spatial)"; h_m2f_c = "M2x, M2y (F Coeff)"; h_stat = "Status"
# Adjusted widths slightly for better alignment with potentially longer model names
w_beam = 20; w_model = 24; w_r2 = 9; w_sig = 16; w_m2 = 16; w_stat = 18; w_m2f = 20

print(f"{h_beam:<{w_beam}} | {h_model:<{w_model}} | {h_r2:<{w_r2}} | {h_sigm:<{w_sig}} | {h_m2m:<{w_m2}} | {h_sigf:<{w_sig}} | {h_m2f_s:<{w_m2f}} | {h_m2f_c:<{w_m2f}} | {h_stat:<{w_stat}}")
total_width = w_beam + w_model + w_r2 + w_sig*2 + w_m2 + w_m2f*2 + w_stat + 8 # Recalculate width
print("-" * total_width)

# MODIFIED ordered_keys to include new test case and use M=X format
ordered_keys = [
    # Perturbation - Mult
    ("TEM00 Ideal", "MultPoly(Gauss Base)"), ("TEM13 Ideal", "MultPoly(HG13 Base)"),
    ("Airy Ideal", "MultPoly(Airy Base)"), ("Bessel Ideal", "MultPoly(Bessel Base)"),
    ("SG(N=10) Ideal", "MultPoly(SG Base)"), ("SG(N=10) Defocus W", "MultPoly(SG Base)"),
    ("SG(N=10) Defocus S", "MultPoly(SG Base)"), ("TEM00 Ab Weak", "MultPoly(Gauss Base)"),
    ("TEM00 Ab Strong", "MultPoly(Gauss Base)"), ("InceGaussian", "MultPoly(HG21 Base)"),
    ("InceGaussian", "MultPoly(LG11 Base)"),
    # Perturbation - Add
    ("AGauss", "AddPoly(Gauss Base)"), ("NP", "AddPoly(Gauss Base)"), ("NL", "AddGauss(Gauss Base)"),
    # Decomposition - HG
    ("MultiMode", f"AddModes(HG, M={mm_hg_max_order})"),
    ("SG(N=10) Ideal", f"AddModes(HG, M={sg_hg_max_order})"),
    ("TEM00 Ab Weak", f"AddModes(HG, M={tem00_ab_max_order})"),
    ("TEM00 Ab Strong", f"AddModes(HG, M={tem00_ab_max_order})"),
    ("InceGaussian", f"AddModes(HG, M={ig_decomp_max_order})"),
    # Decomposition - LG
    ("SG(N=10) Ideal", f"AddModes(LG, M={sg_lg_max_order})"),
    ("TEM00 Ideal", f"AddModes(LG, M={tem00_ideal_lg_max_order})"),
    ("TEM00 Ab Weak", f"AddModes(LG, M={tem00_ab_max_order})"),
    ("TEM00 Ab Strong", f"AddModes(LG, M={tem00_ab_max_order})"),
    ("InceGaussian", f"AddModes(LG, M={ig_decomp_max_order})"),
    ("Noisy Gaussian", f"AddModes(LG, M={noisy_gauss_lg_max_order})"), # ADDED
]

for (beam_key, model_key) in ordered_keys:
    res = results_table.get((beam_key, model_key))
    if res is None:
        # Ensure model key is formatted correctly even if results are missing
        formatted_model_key = model_key
        # Special handling if model_key contains placeholders that weren't evaluated
        try:
            formatted_model_key = model_key.format(
                mm_hg_max_order=mm_hg_max_order,
                sg_hg_max_order=sg_hg_max_order, sg_lg_max_order=sg_lg_max_order,
                tem00_ideal_lg_max_order=tem00_ideal_lg_max_order,
                tem00_ab_max_order=tem00_ab_max_order,
                ig_decomp_max_order=ig_decomp_max_order,
                noisy_gauss_lg_max_order=noisy_gauss_lg_max_order
                # Add others if needed
            )
        except NameError: # In case some variables aren't defined when skip occurs early
             pass
        except KeyError: # In case format string uses an unexpected key
             pass

        print(f"{beam_key:<{w_beam}} | {formatted_model_key:<{w_model}} | {'N/A':<{w_r2}} | {'N/A':<{w_sig}} | {'N/A':<{w_m2}} | {'N/A':<{w_sig}} | {'N/A':<{w_m2f}} | {'N/A':<{w_m2f}} | {'Not Run/Error':<{w_stat}}"); continue

    r2_str = f"{res.get('R2', np.nan):.6f}" if not np.isnan(res.get('R2', np.nan)) else "NaN"
    sigm_str = f"{res.get('sigx_m', np.nan):.3f},{res.get('sigy_m', np.nan):.3f}"
    m2m_str = f"{res.get('Mx2_m', np.nan):.3f},{res.get('My2_m', np.nan):.3f}"
    sigf_str = f"{res.get('sigx_f', np.nan):.3f},{res.get('sigy_f', np.nan):.3f}"
    status_str = str(res.get('fit_status', 'Unknown')); status_str = status_str[:w_stat-3] + "..." if len(status_str) > w_stat else status_str
    model_type_indicator = res.get('model_type', '') # Get the actual model name from the results dict

    # Initialize M2 strings
    m2f_s_str = "---"
    m2f_c_str = "---"

    if "AddModes" in model_type_indicator:
        m2f_s_val_x = res.get('Mx2_f_spatial', np.nan)
        m2f_s_val_y = res.get('My2_f_spatial', np.nan)
        m2f_c_val_x = res.get('Mx2_f_coeff', np.nan)
        m2f_c_val_y = res.get('My2_f_coeff', np.nan)
        m2f_s_str = f"{m2f_s_val_x:.3f},{m2f_s_val_y:.3f}"
        m2f_c_str = f"{m2f_c_val_x:.3f},{m2f_c_val_y:.3f}"
    elif "MultPoly" in model_type_indicator or "AddPoly" in model_type_indicator or "AddGauss" in model_type_indicator:
        m2f_s_val_x = res.get('Mx2_f', np.nan) # Perturbation fits store M2 spatial here
        m2f_s_val_y = res.get('My2_f', np.nan)
        m2f_s_str = f"{m2f_s_val_x:.3f},{m2f_s_val_y:.3f}"
        # No coefficient M2 for perturbation models
        m2f_c_str = "---"

    # Replace nan with N/A for printing
    m2m_str = m2m_str.replace("nan", "N/A"); sigm_str = sigm_str.replace("nan", "N/A")
    m2f_s_str = m2f_s_str.replace("nan", "N/A"); m2f_c_str = m2f_c_str.replace("nan", "N/A")
    sigf_str = sigf_str.replace("nan", "N/A")

    print(f"{beam_key:<{w_beam}} | {model_type_indicator:<{w_model}} | {r2_str:<{w_r2}} | {sigm_str:<{w_sig}} | {m2m_str:<{w_m2}} | {sigf_str:<{w_sig}} | {m2f_s_str:<{w_m2f}} | {m2f_c_str:<{w_m2f}} | {status_str:<{w_stat}}")


print("-" * total_width)
print("="*80)
logger.info("\n*** Full Script Completed (v12.9 with Noisy Gaussian) ***")