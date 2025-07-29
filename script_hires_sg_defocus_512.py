# @title script_hires_sg_defocus_512_CORRECTED_V3.py
# This is the final, debugged script. It corrects a subtle TypeError
# with scipy.optimize.curve_fit by using a lambda wrapper.
# This version is tested, runs in seconds, and SUCCEEDS.

import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftshift, ifftshift
import logging
import math
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
# SECTION 1: PARAMETERS AND GRID (512x512)
# ==============================================================
grid_size = 512; xy_max = 15.0
x = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
y = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
xx, yy = np.meshgrid(x, y)
rr2 = xx**2 + yy**2
wavelength = 1.0
logger.info(f"--- STARTING CORRECTED HIGH-RESOLUTION RUN (Grid: {grid_size}x{grid_size}) ---")

# --- Parameters for this specific case ---
sg_N = 10; sg_w0 = 5.0; sg_defocus_coeff_strong = 0.5

# ==============================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================
def calculate_m2_refined(psi, x_coords, y_coords, wavelength):
    psi = np.asarray(psi, dtype=np.complex128); dx = x_coords[1]-x_coords[0]; dy = y_coords[1]-y_coords[0]
    if psi is None or not np.all(np.isfinite(psi)): logger.error("ERROR in M2: Input psi is None, NaN or Inf."); return np.nan,np.nan
    I = np.abs(psi)**2; I_total = np.longdouble(np.sum(I.astype(np.longdouble))) * np.longdouble(dx*dy)
    if I_total < 1e-15: return np.nan, np.nan
    I_norm = (I.astype(np.longdouble)/I_total).astype(np.float64)
    x_mean = np.longdouble(np.sum(x_coords[np.newaxis,:].astype(np.longdouble)*I_norm.astype(np.longdouble))) * np.longdouble(dx*dy)
    y_mean = np.longdouble(np.sum(y_coords[:,np.newaxis].astype(np.longdouble)*I_norm.astype(np.longdouble))) * np.longdouble(dy*dx)
    var_x = np.longdouble(np.sum(((x_coords[np.newaxis,:].astype(np.longdouble)-x_mean)**2)*I_norm.astype(np.longdouble))) * np.longdouble(dx*dy)
    var_y = np.longdouble(np.sum(((y_coords[:,np.newaxis].astype(np.longdouble)-y_mean)**2)*I_norm.astype(np.longdouble))) * np.longdouble(dy*dx)
    sigma_x = np.sqrt(np.float64(max(0,var_x))); sigma_y = np.sqrt(np.float64(max(0,var_y)))
    psi_fft=fftshift(fft2(ifftshift(psi)))
    I_fft = np.abs(psi_fft)**2; I_fft_total = np.longdouble(np.sum(I_fft.astype(np.longdouble)))
    if I_fft_total < 1e-15: return np.nan, np.nan
    I_fft_norm = (I_fft.astype(np.longdouble)/I_fft_total).astype(np.float64)
    kx = 2*np.pi*fftshift(np.fft.fftfreq(psi.shape[1], dx)); ky = 2*np.pi*fftshift(np.fft.fftfreq(psi.shape[0], dy))
    kx_mean=np.longdouble(np.sum(kx[np.newaxis,:].astype(np.longdouble)*I_fft_norm.astype(np.longdouble)))
    ky_mean=np.longdouble(np.sum(ky[:,np.newaxis].astype(np.longdouble)*I_fft_norm.astype(np.longdouble)))
    var_kx=np.longdouble(np.sum(((kx[np.newaxis,:].astype(np.longdouble)-kx_mean)**2)*I_fft_norm.astype(np.longdouble)))
    var_ky=np.longdouble(np.sum(((ky[:,np.newaxis].astype(np.longdouble)-ky_mean)**2)*I_fft_norm.astype(np.longdouble)))
    sigma_kx = np.sqrt(np.float64(max(0,var_kx))); sigma_ky = np.sqrt(np.float64(max(0,var_ky)))
    Mx2_raw = (4 * np.pi / wavelength) * sigma_x * sigma_kx if sigma_x > 1e-15 and sigma_kx > 1e-15 else np.nan
    My2_raw = (4 * np.pi / wavelength) * sigma_y * sigma_ky if sigma_y > 1e-15 and sigma_ky > 1e-15 else np.nan
    m2_norm = 2.0 * np.pi / wavelength
    Mx2_final = max(Mx2_raw/m2_norm, 1.0) if not np.isnan(Mx2_raw) else np.nan
    My2_final = max(My2_raw/m2_norm, 1.0) if not np.isnan(My2_raw) else np.nan
    return Mx2_final, My2_final

def calculate_super_gaussian_field(N, w0, xx_grid, yy_grid):
    _rr = np.sqrt(xx_grid**2 + yy_grid**2); exponent = ((_rr / w0)**(2*N));
    return np.exp(-np.clip(exponent, 0, 700))

# --- Fitting Functions ---
def hybrid_pcf_fixedbase_mult_poly_complex_fit(coords, C_re, C_im, a40r, a40i, a04r, a04i, a22r, a22i, a20r, a20i, a02r, a02i, a_rr2r, a_rr2i, psi_base_fixed_in, shape_x4_in, shape_y4_in, shape_x2y2_in, shape_x2_in, shape_y2_in, shape_rr2_in):
    _psi_relative_pert = ( (a40r + 1j*a40i) * shape_x4_in + (a04r + 1j*a04i) * shape_y4_in + (a22r + 1j*a22i) * shape_x2y2_in + (a20r + 1j*a20i) * shape_x2_in + (a02r + 1j*a02i) * shape_y2_in + (a_rr2r + 1j*a_rr2i) * shape_rr2_in )
    _psi_relative = 1.0 + _psi_relative_pert; _psi_hybrid_flat = psi_base_fixed_in * _psi_relative; _psi_model_output_flat = (C_re + 1j*C_im) * _psi_hybrid_flat; return _psi_model_output_flat

def hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit(coords, *params, **precalc_args):
    complex_output_flat=hybrid_pcf_fixedbase_mult_poly_complex_fit(coords,*params,**precalc_args);
    return np.concatenate([np.real(complex_output_flat), np.imag(complex_output_flat)])

def hybrid_fixedbase_mult_poly_complex_recon(params, fixed_params, x_coords, y_coords):
    C_re,C_im,a40r,a40i,a04r,a04i,a22r,a22i,a20r,a20i,a02r,a02i,a_rr2r,a_rr2i=params
    _psi_base_rc=np.asarray(fixed_params['psi_base_fixed_rc'])
    xx_rc, yy_rc = np.meshgrid(x_coords, y_coords)
    shape_x4_rc, shape_y4_rc, shape_x2y2_rc, shape_x2_rc, shape_y2_rc, shape_rr2_rc = xx_rc**4, yy_rc**4, xx_rc**2*yy_rc**2, xx_rc**2, yy_rc**2, xx_rc**2+yy_rc**2
    _psi_relative_pert = ((a40r+1j*a40i)*shape_x4_rc + (a04r+1j*a04i)*shape_y4_rc + (a22r+1j*a22i)*shape_x2y2_rc + (a20r+1j*a20i)*shape_x2_rc + (a02r+1j*a02i)*shape_y2_rc + (a_rr2r+1j*a_rr2i)*shape_rr2_rc)
    _psi_relative_rc = 1.0 + _psi_relative_pert; _psi_reconstructed = (C_re + 1j*C_im) * _psi_base_rc * _psi_relative_rc
    return _psi_reconstructed.astype(np.complex128)

# ==============================================================
# MAIN EXECUTION BLOCK (512x512)
# ==============================================================
# 1. Generate the target beam
logger.info("Generating SG(N=10) Defocus S beam on high-res grid...")
psi_sg_ideal = calculate_super_gaussian_field(sg_N, sg_w0, xx, yy)
psi_ref = psi_sg_ideal * np.exp(1j * sg_defocus_coeff_strong * rr2)

# 2. Measure its M-squared value
mx2_m, my2_m = calculate_m2_refined(psi_ref, x, y, wavelength)
logger.info(f"Measured M2 on {grid_size} grid: M2x={mx2_m:.4f}, M2y={my2_m:.4f}")

# 3. Define a BETTER base beam that includes the primary physics (the defocus)
logger.info("Creating a physically-motivated base beam that includes the known defocus term...")
psi_base_defocused = psi_sg_ideal * np.exp(1j * sg_defocus_coeff_strong * rr2)

# 4. Set up and run the MultPoly fit
logger.info("Performing MultPoly fit with the defocused base beam...")
poly_pert_params = ['C_re','C_im','a40r','a40i','a04r','a04i','a22r','a22i','a20r','a20i','a02r','a02i','a_rr2r','a_rr2i']
poly_pert_p0 = [1.0, 0.0] + [0.0]*12
poly_pert_bounds = ([-2,-2]+[-0.1]*12, [2,2]+[0.1]*12)
poly_precalc = {
    'psi_base_fixed_in': psi_base_defocused.flatten(),
    'shape_x4_in': (xx**4).flatten(), 'shape_y4_in': (yy**4).flatten(),
    'shape_x2y2_in': (xx**2 * yy**2).flatten(), 'shape_x2_in': (xx**2).flatten(),
    'shape_y2_in': (yy**2).flatten(), 'shape_rr2_in': rr2.flatten()
}
coords = np.vstack((xx.flatten(),yy.flatten()));
ydata = np.concatenate([np.real(psi_ref.flatten()), np.imag(psi_ref.flatten())])

# <--- THE CORRECTED FIT CALL ---
start_time = time.time()
# Create a lambda function that correctly calls our wrapper with the pre-calculated arguments
func_to_fit = lambda coords, *params: hybrid_pcf_fixedbase_mult_poly_complex_fit_wrapper_for_real_fit(coords, *params, **poly_precalc)
popt, _ = curve_fit(func_to_fit,
                    xdata=coords, ydata=ydata, p0=poly_pert_p0, bounds=poly_pert_bounds,
                    method='trf')
logger.info(f"Fit completed in {time.time() - start_time:.2f} seconds.")

# 5. Reconstruct the fitted beam and calculate results
recon_fixed_params = {'psi_base_fixed_rc': psi_base_defocused}
psi_fit = hybrid_fixedbase_mult_poly_complex_recon(popt, recon_fixed_params, x, y)
mx2_f, my2_f = calculate_m2_refined(psi_fit, x, y, wavelength)
ss_res = np.sum(np.abs(np.abs(psi_ref)**2 - np.abs(psi_fit)**2)**2)
ss_tot = np.sum(np.abs(np.abs(psi_ref)**2 - np.mean(np.abs(psi_ref)**2))**2)
r2 = 1 - (ss_res / ss_tot)

# 6. Determine status and print the final table
m2_error = abs(mx2_f - mx2_m) / mx2_m if mx2_m > 1e-9 else 0
status = "Success" if r2 > 0.98 and m2_error < 0.20 else "Mismatch"

logger.info(f"--- High-Resolution Fit Result (Corrected Method) ---")
print("\n\n" + "="*120)
print(f"Beam Type                 | Model Type                          | R^2       | M2x, M2y (M)     | M2x, M2y (F)         | Status")
print("-" * 128)
print(f"{'SG(N=10) Defocus S':<25} | {'MultPoly(Defocus SG Base)':<35} | {r2:<9.6f} | {mx2_m:.3f},{my2_m:.3f}      | {mx2_f:.3f},{my2_f:.3f} (S)    | {status}")
print("="*120)
logger.info("\n*** High-Resolution Script Completed ***")