# /generate_table_1.py (UPDATED with Cosh-Gaussian Job)

import numpy as np
import time
import logging
import lib.beam_definitions as beams
import lib.analysis_models as models
import lib.m2_utils as m2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_optimized_addmodes_fit(job, ref_data):
    """
    A special runner for challenging beams. It optimizes the basis waist `w0`
    to find the best possible fit, mimicking the successful Ince-Gaussian strategy.
    """
    logger.info(f"--- Starting OPTIMIZED search for '{job['beam_key']}' ---")
    w0_ref = job['basis_w0']; waists_to_test = np.linspace(w0_ref * 0.7, w0_ref * 1.5, 7)
    best_result_dict, best_r2 = None, -1

    for w0_test in waists_to_test:
        logger.info(f"Testing basis w0 = {w0_test:.3f}...")
        temp_job = job.copy(); temp_job['basis_w0'] = w0_test
        result = models.run_addmodes_fit(temp_job, ref_data)
        current_r2 = result.get('R2', -1)
        if current_r2 > best_r2:
            best_r2 = current_r2; best_result_dict = result
            
    logger.info(f"--- OPTIMIZED search complete. Best R2={best_r2:.4f} found at w0={best_result_dict['job']['basis_w0']:.3f} ---")
    if best_result_dict:
        best_result_dict['beam_key'] = job['beam_key']; best_result_dict['model_type'] = job['model_type']
    return best_result_dict


def main():
    logger.info("--- STARTING UNIFIED FRAMEWORK TO GENERATE TABLE 1 ---")

    job_definitions = [
        # --- Perturbation Model Fits ---
        {'beam_key': 'TEM00 Ideal',          'model_type': 'MultPoly(Gauss Base)',   'grid_size': 256, 'psi_ref_func': beams.get_tem00_ideal, 'base_func': beams.get_tem00_ideal},
        {'beam_key': 'TEM13 Ideal',          'model_type': 'MultPoly(HG13 Base)',    'grid_size': 256, 'psi_ref_func': beams.get_tem13_ideal, 'base_func': beams.get_tem13_ideal},
        {'beam_key': 'Airy Ideal',           'model_type': 'MultPoly(Airy Base)',    'grid_size': 256, 'psi_ref_func': beams.get_airy_ideal, 'base_func': beams.get_airy_ideal},
        {'beam_key': 'Bessel Ideal',         'model_type': 'MultPoly(Bessel Base)',  'grid_size': 256, 'psi_ref_func': beams.get_bessel_ideal, 'base_func': beams.get_bessel_ideal},
        {'beam_key': 'SG(N=10) Ideal',       'model_type': 'MultPoly(SG Base)',      'grid_size': 256, 'psi_ref_func': beams.get_sg_ideal, 'base_func': beams.get_sg_ideal},
        {'beam_key': 'SG(N=10) Defocus W',   'model_type': 'MultPoly(SG Base)',      'grid_size': 256, 'psi_ref_func': beams.get_sg_defocus_weak, 'base_func': beams.get_sg_ideal},
        {'beam_key': 'SG(N=10) Defocus S',   'model_type': 'MultPoly(SG Base)',      'grid_size': 256, 'psi_ref_func': beams.get_sg_defocus_strong, 'base_func': beams.get_sg_ideal},
        {'beam_key': 'SG(N=10) Defocus S',   'model_type': 'MultPoly(Defocus SG Base)', 'grid_size': 512, 'psi_ref_func': beams.get_sg_defocus_strong_hr, 'base_func': beams.get_sg_defocus_strong_hr},
        {'beam_key': 'TEM00 Ab Strong',      'model_type': 'MultPoly(Gauss Base)',   'grid_size': 256, 'psi_ref_func': beams.get_tem00_ab_strong, 'base_func': beams.get_tem00_ideal},
        {'beam_key': 'AGauss',               'model_type': 'AddPoly(Gauss Base)',    'grid_size': 256, 'psi_ref_func': beams.get_agauss, 'base_func': beams.get_agauss_base},
        {'beam_key': 'NP',                   'model_type': 'AddPoly(Gauss Base)',    'grid_size': 256, 'psi_ref_func': beams.get_np, 'base_func': beams.get_np_base},
        {'beam_key': 'NL',                   'model_type': 'AddGauss(Gauss Base)',   'grid_size': 256, 'psi_ref_func': beams.get_nl, 'base_func': beams.get_nl_base, 'pert_shape_func': beams.get_nl_pert},

        # --- Additive Modal Decomposition Fits ---
        {'beam_key': 'MultiMode',            'model_type': 'AddModes(HG, M=4)',      'grid_size': 256, 'psi_ref_func': beams.get_multimode, 'basis_type': 'HG', 'max_order': 4, 'basis_w0': 1.0, 'm2_method': 'coeff_simple'},
        {'beam_key': 'TEM00 Ab Strong',      'model_type': 'AddModes(LG, M=16)',     'grid_size': 256, 'psi_ref_func': beams.get_tem00_ab_strong, 'basis_type': 'LG', 'max_order': 16, 'basis_w0': 1.0, 'm2_method': 'coeff_simple'},
        {'beam_key': 'Ince-Gaussian',        'model_type': 'AddModes(LG)',           'grid_size': 256, 'psi_ref_func': beams.get_ince_gaussian, 'basis_type': 'LG', 'max_order': 22, 'basis_w0': 1.6, 'm2_method': 'coeff_robust'},
        {'beam_key': 'Mathieu Beam',         'model_type': 'AddModes(HG, M=16)',     'grid_size': 256, 'psi_ref_func': beams.get_mathieu_beam, 'basis_type': 'HG', 'max_order': 16, 'basis_w0': 1.5, 'm2_method': 'coeff_robust', 'use_optimizer': True},
        {'beam_key': 'Mathieu Beam',         'model_type': 'MultPoly(Mathieu Base)', 'grid_size': 256, 'psi_ref_func': beams.get_mathieu_beam, 'base_func': beams.get_mathieu_beam},
        {'beam_key': 'Parabolic Beam',       'model_type': 'AddModes(HG, M=16)',     'grid_size': 256, 'psi_ref_func': beams.get_parabolic_beam, 'basis_type': 'HG', 'max_order': 16, 'basis_w0': 2.0, 'm2_method': 'coeff_robust', 'use_optimizer': True},
        {'beam_key': 'Parabolic Beam',       'model_type': 'MultPoly(Parabolic Base)','grid_size': 256, 'psi_ref_func': beams.get_parabolic_beam, 'base_func': beams.get_parabolic_beam},
        
        # <<< NEW COSH-GAUSSIAN JOB >>>
        {'beam_key': 'Cosh-Gaussian',        'model_type': 'AddModes(HG, M=8)',      'grid_size': 256, 'psi_ref_func': beams.get_cosh_gaussian_beam, 'basis_type': 'HG', 'max_order': 8, 'basis_w0': 2.0, 'm2_method': 'coeff_simple'},

        {'beam_key': 'Noisy Gaussian',       'model_type': 'AddModes(LG, M=10)',     'grid_size': 256, 'psi_ref_func': beams.get_noisy_gaussian, 'basis_type': 'LG', 'max_order': 10, 'basis_w0': 1.0, 'm2_method': 'coeff_simple', 'rcond': 1e-8},
    ]

    results_table, measured_data_cache = [], {}

    for job in job_definitions:
        beam_key, model_key, grid_size = job['beam_key'], job['model_type'], job['grid_size']
        logger.info(f"--- Processing Job: '{beam_key}' with model '{model_key}' on {grid_size}x{grid_size} grid ---")
        grid_params = beams.setup_grid(grid_size); cache_key = (beam_key, grid_size)
        if cache_key not in measured_data_cache:
            psi_ref = job['psi_ref_func'](grid_params); mx2_m, my2_m = m2.calculate_m2_spatial(psi_ref, grid_params)
            measured_data_cache[cache_key] = {'psi_ref': psi_ref, 'Mx2_m': mx2_m, 'My2_m': my2_m, 'grid_params': grid_params}
        ref_data = measured_data_cache[cache_key]
        
        if job.get('use_optimizer', False): result = run_optimized_addmodes_fit(job, ref_data)
        elif 'AddModes' in model_key: result = models.run_addmodes_fit(job, ref_data)
        else: result = models.run_perturbation_fit(job, ref_data)
        results_table.append(result)

    print_results_table(results_table)


def print_results_table(results):
    logger.info("--- Final Results Summary Table ---")
    print("\n\n" + "="*130); print("--- Final Results Summary Table ---"); print("="*130)
    h_beam, h_model, h_r2, h_m2m, h_m2f, h_stat = "Beam Type", "Model Type", "R^2", "M2x, M2y (M)", "M2x, M2y (F)", "Status"
    w_beam, w_model, w_r2, w_m2, w_m2f, w_stat = 25, 45, 9, 16, 20, 22
    print(f"{h_beam:<{w_beam}} | {h_model:<{w_model}} | {h_r2:<{w_r2}} | {h_m2m:<{w_m2}} | {h_m2f:<{w_m2f}} | {h_stat:<{w_stat}}")
    print("-" * (w_beam + w_model + w_r2 + w_m2 + w_m2f + w_stat + 10))

    order_map = [
        ('TEM00 Ideal', 'MultPoly(Gauss Base)'), ('TEM13 Ideal', 'MultPoly(HG13 Base)'), ('Airy Ideal', 'MultPoly(Airy Base)'),
        ('Bessel Ideal', 'MultPoly(Bessel Base)'), ('SG(N=10) Ideal', 'MultPoly(SG Base)'), ('SG(N=10) Defocus W', 'MultPoly(SG Base)'),
        ('SG(N=10) Defocus S', 'MultPoly(SG Base)'), ('SG(N=10) Defocus S', 'MultPoly(Defocus SG Base)'), ('TEM00 Ab Strong', 'MultPoly(Gauss Base)'),
        ('AGauss', 'AddPoly(Gauss Base)'), ('NP', 'AddPoly(Gauss Base)'), ('NL', 'AddGauss(Gauss Base)'),
        ('MultiMode', 'AddModes(HG, M=4)'), ('TEM00 Ab Strong', 'AddModes(LG, M=16)'), ('Ince-Gaussian', 'AddModes(LG)'),
        ('Mathieu Beam', 'AddModes(HG, M=16)'), ('Mathieu Beam', 'MultPoly(Mathieu Base)'),
        ('Parabolic Beam', 'AddModes(HG, M=16)'), ('Parabolic Beam', 'MultPoly(Parabolic Base)'),
        ('Cosh-Gaussian', 'AddModes(HG, M=8)'),
        ('Noisy Gaussian', 'AddModes(LG, M=10)'),
    ]
    
    results_dict = {(r['job']['beam_key'], r['job']['model_type']): r for r in results}

    for beam_key, model_key in order_map:
        res = results_dict.get((beam_key, model_key))
        if res is None: continue

        m2m_str = f"{res.get('Mx2_m', np.nan):.3f},{res.get('My2_m', np.nan):.3f}"
        r2_str = f"{res.get('R2', np.nan):.4f}" if not np.isnan(res.get('R2', np.nan)) else "NaN"
        m2f_str = f"{res.get('Mx2_f', np.nan):.3f},{res.get('My2_f', np.nan):.3f}"
        m2f_str += " (C)" if 'AddModes' in model_key else " (S)"
        
        model_print_str = res['job']['model_type']
        if res.get('job', {}).get('use_optimizer', False):
            opt_w0 = res.get('job', {}).get('basis_w0', 'N/A')
            model_print_str += f" (opt w0={opt_w0:.2f})"
        
        print(f"{beam_key:<{w_beam}} | {model_print_str:<{w_model}} | {r2_str:<{w_r2}} | {m2m_str:<{w_m2}} | {m2f_str:<{w_m2f}} | {res.get('fit_status', 'Unknown'):<{w_stat}}")

    print("="*130); logger.info("--- UNIFIED FRAMEWORK COMPLETED ---")

if __name__ == "__main__":
    main()