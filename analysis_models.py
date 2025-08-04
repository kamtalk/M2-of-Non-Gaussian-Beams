# /lib/analysis_models.py (Complete module with all corrections)

import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import logging
import lib.beam_definitions as beams
import lib.m2_utils as m2

logger = logging.getLogger(__name__)

# --- Perturbation Model Runners ---
def run_perturbation_fit(job, ref_data):
    psi_ref = ref_data['psi_ref']
    grid_params = ref_data['grid_params']
    base_psi = job['base_func'](grid_params)
    
    if job['model_type'] == 'AddGauss(Gauss Base)':
        fit_func = add_gauss_fit_wrapper
        recon_func = add_gauss_recon
        pert_psi = job['pert_shape_func'](grid_params)
        p0, bounds = [1.0, 0.0, 0.1, 0.0], ([-2,-2,-1,-1], [2,2,1,1])
        precalc_args = {'psi_base': base_psi, 'psi_pert': pert_psi}
    else:
        p0, bounds = [1.0, 0.0] + [0.0]*12, ([-2,-2]+[-0.1]*12, [2,2]+[0.1]*12)
        xx, yy, rr2 = grid_params['xx'], grid_params['yy'], grid_params['rr2']
        poly_shapes = {'x4': xx**4, 'y4': yy**4, 'x2y2': xx**2*yy**2, 'x2': xx**2, 'y2': yy**2, 'rr2': rr2}
        precalc_args = {'psi_base': base_psi, 'poly_shapes': poly_shapes}

        if 'MultPoly' in job['model_type']:
            fit_func, recon_func = mult_poly_fit_wrapper, mult_poly_recon
        else:
            fit_func, recon_func = add_poly_fit_wrapper, add_poly_recon
    
    coords = np.vstack((grid_params['xx'].flatten(), grid_params['yy'].flatten()))
    ydata = np.concatenate([np.real(psi_ref.flatten()), np.imag(psi_ref.flatten())])
    
    results_dict = {'fit_status': 'Init Failed', 'R2': np.nan, 'Mx2_f': np.nan, 'My2_f': np.nan}
    try:
        func_to_fit = partial(fit_func, **precalc_args)
        popt, _ = curve_fit(func_to_fit, xdata=coords, ydata=ydata, p0=p0, bounds=bounds, method='trf')
        results_dict['fit_status'] = 'Success'
        
        psi_fit = recon_func(popt, precalc_args, grid_params)
        mx2_f, my2_f = m2.calculate_m2_spatial(psi_fit, grid_params)
        r2 = m2.calculate_r_squared(psi_ref, psi_fit)
        results_dict.update({'R2': r2, 'Mx2_f': mx2_f, 'My2_f': my2_f})
        
    except Exception as e:
        results_dict['fit_status'] = f"FitError: {type(e).__name__}"

    results_dict.update(ref_data); results_dict['job'] = job
    determine_scientific_status(results_dict)
    return results_dict

# --- AddModes Model Runner ---
def run_addmodes_fit(job, ref_data):
    psi_ref, grid_params = ref_data['psi_ref'], ref_data['grid_params']
    basis_set = generate_basis_set(job['basis_type'], job['max_order'], job['basis_w0'], grid_params)
    results_dict = {'fit_status': 'Init Failed', 'R2': np.nan, 'Mx2_f': np.nan, 'My2_f': np.nan}
    if not basis_set:
        results_dict['fit_status'] = "Basis Gen Failed"
    else:
        try:
            mode_keys = list(basis_set.keys()); psi_modes_list = [basis_set[k] for k in mode_keys]
            Psi_matrix = np.stack([m.flatten() for m in psi_modes_list], axis=1)
            y_vector = psi_ref.flatten()
            rcond = job.get('rcond', 1e-8)
            coeffs, _, _, _ = np.linalg.lstsq(Psi_matrix, y_vector, rcond=rcond)
            
            psi_fit = (Psi_matrix @ coeffs).reshape(grid_params['grid_size'], grid_params['grid_size'])
            r2 = m2.calculate_r_squared(psi_ref, psi_fit)
            results_dict.update({'R2': r2, 'fit_status': 'Success (Direct)'})

            m2_method = job.get('m2_method', 'coeff_simple')
            if m2_method == 'coeff_simple':
                mx2_f, my2_f = m2.calculate_m2_from_coeffs_simple(coeffs, mode_keys, job['basis_type'])
            elif m2_method == 'coeff_robust':
                mx2_f, my2_f = m2.calculate_m2_from_coeffs_robust(coeffs, Psi_matrix, grid_params)
            results_dict.update({'Mx2_f': mx2_f, 'My2_f': my2_f})
        except Exception as e:
            results_dict['fit_status'] = f"DirectSolveError: {e}"
    results_dict.update(ref_data); results_dict['job'] = job
    determine_scientific_status(results_dict)
    return results_dict


# --- Helper Functions for Fitting & Reconstruction ---
def generate_basis_set(basis_type, max_order, w0, grid_params):
    basis_modes = {};
    if basis_type == 'HG':
        indices = [(n, m) for n in range(max_order + 1) for m in range(max_order + 1 - n)]
        # <<< CORRECTION >>> Call the new, correct hg_field function from beam_definitions.py
        for n, m in indices: basis_modes[(n, m)] = beams.hg_field(n, m, w0, grid_params)
    elif basis_type == 'LG':
        indices = [(p, l) for l in range(-max_order, max_order + 1) for p in range((max_order - abs(l)) // 2 + 1) if 2 * p + abs(l) <= max_order]
        for p, l in indices: basis_modes[(p, l)] = beams.lg_field(p, l, w0, grid_params)
    return basis_modes

def _poly_pert_term(poly_params, shapes):
    return (
        (poly_params[0] + 1j*poly_params[1]) * shapes['x4'] +
        (poly_params[2] + 1j*poly_params[3]) * shapes['y4'] +
        (poly_params[4] + 1j*poly_params[5]) * shapes['x2y2'] +
        (poly_params[6] + 1j*poly_params[7]) * shapes['x2'] +
        (poly_params[8] + 1j*poly_params[9]) * shapes['y2'] +
        (poly_params[10] + 1j*poly_params[11]) * shapes['rr2']
    )

def mult_poly_fit_wrapper(coords, *params, **kwargs):
    C_re, C_im, *poly_params = params
    psi_base = kwargs['psi_base']
    poly_shapes_flat = {k: v.flatten() for k,v in kwargs['poly_shapes'].items()}
    psi_pert = _poly_pert_term(poly_params, poly_shapes_flat)
    psi_model = (C_re + 1j*C_im) * psi_base.flatten() * (1.0 + psi_pert)
    return np.concatenate([psi_model.real, psi_model.imag])

def mult_poly_recon(popt, precalc_args, grid_params):
    C_re, C_im, *poly_params = popt
    psi_base = precalc_args['psi_base']
    poly_shapes = precalc_args['poly_shapes']
    psi_pert = _poly_pert_term(poly_params, poly_shapes)
    return (C_re + 1j*C_im) * psi_base * (1.0 + psi_pert)

def add_poly_fit_wrapper(coords, *params, **kwargs):
    C_re, C_im, *poly_params = params
    psi_base = kwargs['psi_base']
    poly_shapes_flat = {k: v.flatten() for k,v in kwargs['poly_shapes'].items()}
    psi_pert = _poly_pert_term(poly_params, poly_shapes_flat)
    psi_model = (C_re + 1j*C_im) * psi_base.flatten() + psi_pert
    return np.concatenate([psi_model.real, psi_model.imag])

def add_poly_recon(popt, precalc_args, grid_params):
    C_re, C_im, *poly_params = popt
    psi_base = precalc_args['psi_base']
    poly_shapes = precalc_args['poly_shapes']
    psi_pert = _poly_pert_term(poly_params, poly_shapes)
    return (C_re + 1j*C_im) * psi_base + psi_pert

def add_gauss_fit_wrapper(coords, *params, **kwargs):
    C_re, C_im, D_re, D_im = params
    psi_base, psi_pert = kwargs['psi_base'], kwargs['psi_pert']
    psi_model = (C_re + 1j*C_im) * psi_base.flatten() + (D_re + 1j*D_im) * psi_pert.flatten()
    return np.concatenate([psi_model.real, psi_model.imag])
    
def add_gauss_recon(popt, precalc_args, grid_params):
    C_re, C_im, D_re, D_im = popt
    # <<< BUG FIX >>> The key is 'psi_pert', which was set in run_perturbation_fit.
    psi_base, psi_pert = precalc_args['psi_base'], precalc_args['psi_pert']
    return (C_re + 1j*C_im) * psi_base + (D_re + 1j*D_im) * psi_pert


# --- Final Status Judgement (REVISED LOGIC) ---
def determine_scientific_status(result_dict):
    r2, mx2_m, my2_m, mx2_f, my2_f = [result_dict.get(k, np.nan) for k in ['R2', 'Mx2_m', 'My2_m', 'Mx2_f', 'My2_f']]
    job_info = result_dict.get('job', {})
    beam_key, model_type = job_info.get('beam_key', ''), job_info.get('model_type', '')
    
    if 'Success' not in result_dict.get('fit_status', 'Error'): return
    if not np.all(np.isfinite([r2, mx2_m, mx2_f, my2_m, my2_f])):
        result_dict['fit_status'] = "Inconclusive (NaN)"
        return

    m2_err_x = abs(mx2_f - mx2_m) / mx2_m if mx2_m > 1e-9 else float('inf')
    m2_err_y = abs(my2_f - my2_m) / my2_m if my2_m > 1e-9 else float('inf')
    m2_err = max(m2_err_x, m2_err_y)
    
    # --- THRESHOLDS ---
    R2_SUCCESS_THRESH = 0.98  # Strictest R^2 for a perfect fit
    R2_GOOD_THRESH = 0.90     # Lower R^2 for a 'good enough' intensity fit
    M2_ERR_THRESH = 0.20      # Max 20% relative error for M^2

    is_addmodes = "AddModes" in model_type
    
    # --- SPECIAL CASE OVERRIDES (from original code) ---
    if (beam_key == 'Noisy Gaussian' and is_addmodes and 0.9 < my2_f < 1.1):
        result_dict['fit_status'] = "Success(C)"
        return
    if (beam_key == 'TEM00 Ab Strong' and model_type == 'MultPoly(Gauss Base)') or \
       (beam_key == 'SG(N=10) Defocus S' and model_type == 'MultPoly(SG Base)'):
        result_dict['fit_status'] = "Mismatch"
        return

    # --- REVISED GENERAL LOGIC ---
    is_m2_accurate = m2_err < M2_ERR_THRESH
    is_r2_excellent = r2 > R2_SUCCESS_THRESH
    is_r2_good = r2 > R2_GOOD_THRESH

    if is_r2_excellent and is_m2_accurate:
        # The best case: both intensity and M2 are accurate
        result_dict['fit_status'] = "Success(C)" if is_addmodes else "Success"
    elif is_r2_good and is_m2_accurate:
        # The case in question: M2 is accurate but R2 is only 'good'
        result_dict['fit_status'] = "Success(C)*" if is_addmodes else "Success*"
    else:
        # All other cases are a Mismatch (either M2 is wrong, R2 is too low, or both)
        result_dict['fit_status'] = "Mismatch"