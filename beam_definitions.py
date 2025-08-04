# /lib/beam_definitions.py (Complete and Final)

import numpy as np
from scipy.special import genlaguerre, jv, airy, hermite
import math
import logging

logger = logging.getLogger(__name__)

# --- Global Parameters for Beam Definitions ---
W0_ADDITIVE_POLY_GAUSS = 1.5
BETA_AG = {'40': 0.0001+0.0j, '04': 0.0001+0.0j, '22': 0.0002+0.0j}
BETA_NP = {'20': 0.0+0.1j, '02': 0.0+0.1j, 'rr2': -0.05+0.0j}
D_NL = 0.5 + 0.0j
W0_GAUSS_PERT_NL = 0.8
TEM00_W0 = 1.0
TEM13_W0 = 1.0
AIRY_X0 = 1.0
AIRY_A = 0.1
BESSEL_L = 1
BESSEL_W0 = 1.0
BESSEL_WG = 5.0
SG_N = 10
SG_W0 = 5.0
SG_DEFOCUS_COEFF_WEAK = 0.05
SG_DEFOCUS_COEFF_STRONG = 0.5
TEM00_AB_P2_STRONG = 2.0
TEM00_AB_P4_STRONG = 1.0
IG_P = 3
IG_M = 1
IG_EPSILON = 2.0
IG_W0 = 2.0
NOISE_LEVEL_STD_DEV = 0.05
NOISE_SEED = 42
MATHIEU_W0 = 1.5
MATHIEU_ELLIPTICITY = 0.6
PARABOLIC_W0 = 2.0
COSH_GAUSS_W0 = 2.0
COSH_GAUSS_OMEGA = 0.5 # Decentering parameter
RING_W0 = 1.5 # Waist parameter for the new ring beam

# --- Grid Setup ---
def setup_grid(grid_size=256, xy_max=15.0):
    x = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
    y = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    rr2 = xx**2 + yy**2
    phi = np.arctan2(yy, xx)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return {'grid_size': grid_size, 'xy_max': xy_max, 'x': x, 'y': y, 'xx': xx, 'yy': yy, 'rr': rr, 'rr2': rr2, 'phi': phi, 'dx': dx, 'dy': dy}

# --- Core Field Calculation Functions ---
def hg_field(n, m, w0, grid_params):
    """Generates a normalized Hermite-Gaussian mode."""
    try:
        norm_factor = 1.0 / math.sqrt(w0**2 * (2**(n+m-1)) * np.pi * math.factorial(n) * math.factorial(m))
    except (ValueError, OverflowError):
        norm_factor = 1.0
        
    w0_safe = max(abs(w0), 1e-12)
    x_norm = np.sqrt(2) * grid_params['xx'] / w0_safe
    y_norm = np.sqrt(2) * grid_params['yy'] / w0_safe
    Hn = hermite(n)(x_norm)
    Hm = hermite(m)(y_norm)
    gauss_env = np.exp(-grid_params['rr2'] / w0_safe**2)
    
    field = norm_factor * Hn * Hm * gauss_env
    field[~np.isfinite(field)] = 0.0
    return field.astype(np.complex128)

def lg_field(p, l, w0, grid_params):
    w0_safe = max(abs(w0), 1e-12)
    try:
        norm_factor = math.sqrt((2.0 * math.factorial(p)) / (np.pi * math.factorial(p + abs(l)))) / w0_safe
    except (ValueError, OverflowError):
        norm_factor = 1.0
    
    radial_term = (np.sqrt(2.0) * grid_params['rr'] / w0_safe)**abs(l)
    laguerre_poly = genlaguerre(p, abs(l))(2.0 * grid_params['rr2'] / w0_safe**2)
    gaussian_term = np.exp(-grid_params['rr2'] / w0_safe**2)
    azimuthal_phase = np.exp(1j * l * grid_params['phi'])
    
    field = norm_factor * radial_term * laguerre_poly * gaussian_term * azimuthal_phase
    field[~np.isfinite(field)] = 0.0
    return field.astype(np.complex128)

def gaussian_field(w0, grid_params):
    return np.exp(-grid_params['rr2'] / w0**2).astype(np.complex128)

def super_gaussian_field(N, w0, grid_params):
    exponent = (grid_params['rr'] / w0)**(2 * N)
    return np.exp(-np.clip(exponent, 0, 700)).astype(np.complex128)

# --- Reference Beam "Getter" Functions ---
def get_tem00_ideal(grid_params): return lg_field(0, 0, TEM00_W0, grid_params)
def get_tem13_ideal(grid_params): return hg_field(1, 3, TEM13_W0, grid_params)
def get_airy_ideal(grid_params):
    sx = grid_params['xx'] / AIRY_X0; sy = grid_params['yy'] / AIRY_X0
    ai, _, _, _ = airy(sx - (AIRY_A**2 / 4.0)); field = ai * np.exp(AIRY_A * sx - (AIRY_A**3 / 12.0)) * np.exp(-sy**2)
    field[~np.isfinite(field)] = 0.0; return field.astype(np.complex128)
def get_bessel_ideal(grid_params):
    k_rho = 2.4048 / BESSEL_W0; bessel_part = jv(BESSEL_L, k_rho * grid_params['rr'])
    gauss_part = np.exp(-grid_params['rr2'] / BESSEL_WG**2); phase_part = np.exp(1j * BESSEL_L * grid_params['phi'])
    field = bessel_part * gauss_part * phase_part; field[~np.isfinite(field)] = 0.0; return field.astype(np.complex128)
def get_sg_ideal(grid_params): return super_gaussian_field(SG_N, SG_W0, grid_params)
def get_sg_defocus_weak(grid_params): return get_sg_ideal(grid_params) * np.exp(1j * SG_DEFOCUS_COEFF_WEAK * grid_params['rr2'])
def get_sg_defocus_strong(grid_params): return get_sg_ideal(grid_params) * np.exp(1j * SG_DEFOCUS_COEFF_STRONG * grid_params['rr2'])
def get_sg_defocus_strong_hr(grid_params): return get_sg_defocus_strong(grid_params)
def get_tem00_ab_strong(grid_params): return get_tem00_ideal(grid_params) * np.exp(1j * (TEM00_AB_P2_STRONG * (grid_params['rr']/TEM00_W0)**2 + TEM00_AB_P4_STRONG * (grid_params['rr']/TEM00_W0)**4))
def get_agauss_base(grid_params): return gaussian_field(W0_ADDITIVE_POLY_GAUSS, grid_params)
def get_agauss(grid_params): return get_agauss_base(grid_params) + (BETA_AG['40']*grid_params['xx']**4 + BETA_AG['04']*grid_params['yy']**4 + BETA_AG['22']*grid_params['xx']**2*grid_params['yy']**2)
def get_np_base(grid_params): return gaussian_field(W0_ADDITIVE_POLY_GAUSS, grid_params)
def get_np(grid_params): return get_np_base(grid_params) + (BETA_NP['20']*grid_params['xx']**2 + BETA_NP['02']*grid_params['yy']**2 + BETA_NP['rr2']*grid_params['rr2'])
def get_nl_base(grid_params): return gaussian_field(W0_ADDITIVE_POLY_GAUSS, grid_params)
def get_nl_pert(grid_params): return D_NL * gaussian_field(W0_GAUSS_PERT_NL, grid_params)
def get_nl(grid_params): return get_nl_base(grid_params) + get_nl_pert(grid_params)
def get_multimode(grid_params):
    coeffs = {(0,0):1.0+0.0j, (1,0):0.5+0.0j, (0,1):0.5j}; psi = np.zeros_like(grid_params['xx'], dtype=np.complex128)
    for (n,m), c in coeffs.items(): psi += c * hg_field(n, m, 1.0, grid_params)
    return psi
def get_ince_gaussian(grid_params):
    f = IG_W0 * np.sqrt(IG_EPSILON / 2.0); term1 = np.sqrt((grid_params['xx'] + f)**2 + grid_params['yy']**2); term2 = np.sqrt((grid_params['xx'] - f)**2 + grid_params['yy']**2)
    xi_arg = (term1 + term2)/(2*f); eta_arg = (term1 - term2)/(2*f)
    xi = np.arccosh(np.clip(xi_arg, 1.0+1e-15, None)); eta = np.arccos(np.clip(eta_arg, -1.0+1e-15, 1.0-1e-15))
    gauss_env = np.exp(-grid_params['rr2'] / IG_W0**2)
    # Placeholder for actual Ince polynomial calculation
    psi = gauss_env * (np.cos(3*eta)*np.cosh(xi) + np.sin(eta)*np.sinh(3*xi))
    psi[~np.isfinite(psi)] = 0.0
    power = np.sum(np.abs(psi)**2);
    if power > 1e-12: psi /= np.sqrt(power)
    return psi.astype(np.complex128)
def get_noisy_gaussian(grid_params):
    np.random.seed(NOISE_SEED); psi_ideal = get_tem00_ideal(grid_params)
    noise_real = np.random.normal(0.0, NOISE_LEVEL_STD_DEV, psi_ideal.shape); noise_imag = np.random.normal(0.0, NOISE_LEVEL_STD_DEV, psi_ideal.shape)
    return psi_ideal + (noise_real + 1j * noise_imag)
def get_mathieu_beam(grid_params):
    logger.info(f"Generating placeholder Mathieu beam (Elliptical HG11)")
    xx, yy, w0, e = grid_params['xx'], grid_params['yy'], MATHIEU_W0, MATHIEU_ELLIPTICITY
    # Simplified elliptical transformation
    x_prime = xx * np.cos(np.pi/4) + yy * np.sin(np.pi/4)
    y_prime = -xx * np.sin(np.pi/4) + yy * np.cos(np.pi/4)
    x_prime_scaled = x_prime * (1 + e)
    y_prime_scaled = y_prime * (1 - e)
    psi = hg_field(1, 1, w0, {'xx': x_prime_scaled, 'yy': y_prime_scaled, 'rr2': x_prime_scaled**2 + y_prime_scaled**2})
    power = np.sum(np.abs(psi)**2);
    if power > 1e-12: psi /= np.sqrt(power)
    return psi.astype(np.complex128)
def get_parabolic_beam(grid_params):
    logger.info(f"Generating placeholder Parabolic beam (w0={PARABOLIC_W0})")
    xx, yy, w0 = grid_params['xx'], grid_params['yy'], PARABOLIC_W0
    u = np.sqrt(grid_params['rr'] + yy); v = np.sqrt(grid_params['rr'] - yy)
    amp = (u**3 - 3*u*v**2) * np.exp(-(u**2 + v**2) / w0**2)
    phase = np.exp(1j * 0.2 * (u**2 - v**2))
    psi = amp * phase
    power = np.sum(np.abs(psi)**2);
    if power > 1e-12: psi /= np.sqrt(power)
    return psi.astype(np.complex128)
def get_cosh_gaussian_beam(grid_params):
    logger.info(f"Generating Cosh-Gaussian beam (w0={COSH_GAUSS_W0}, Omega={COSH_GAUSS_OMEGA})")
    xx, yy, w0, Omega = grid_params['xx'], grid_params['yy'], COSH_GAUSS_W0, COSH_GAUSS_OMEGA
    psi = np.exp(-(xx**2 + yy**2) / w0**2) * np.cosh(Omega * xx) * np.cosh(Omega * yy)
    power = np.sum(np.abs(psi)**2)
    if power > 1e-12: psi /= np.sqrt(power)
    return psi.astype(np.complex128)

def get_ring_beam(grid_params):
    logger.info(f"Generating Ring Beam (LG01, w0={RING_W0})")
    return lg_field(0, 1, RING_W0, grid_params)