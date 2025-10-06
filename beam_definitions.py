# @file: beam_definitions.py
# Contains functions for generating various beam types.

import numpy as np
from scipy.special import genlaguerre
import math

def create_gaussian_beam(w0, xx, yy):
    """Creates a normalized fundamental Gaussian beam."""
    rr2 = xx**2 + yy**2
    psi_amp = np.exp(-rr2 / (w0**2))
    
    # Normalize power to 1
    dx = (xx[0, 1] - xx[0, 0])
    power = np.sum(psi_amp**2) * dx**2
    psi_normalized = (psi_amp / np.sqrt(power)).astype(np.complex128)
    return psi_normalized, {} # Return empty dict for compatibility

def create_aberrated_gaussian(w0, xx, yy, rr, P2=0.0, P4=0.0):
    """Creates a Gaussian beam with defocus (P2) and spherical (P4) aberration."""
    psi_gauss, _ = create_gaussian_beam(w0, xx, yy)
    
    # Phase aberration term
    phase_term = P2 * (rr / w0)**2 + P4 * (rr / w0)**4
    psi_aberrated = psi_gauss * np.exp(1j * phase_term)
    
    params = {'P2': P2, 'P4': P4}
    return psi_aberrated, params

def create_lg_beam(p, l, w0, rr, phi):
    """Creates a single normalized Laguerre-Gaussian beam."""
    norm_factor = math.sqrt((2.0 * math.factorial(p)) / (np.pi * math.factorial(p + abs(l)))) / w0
    radial_term = (np.sqrt(2.0) * rr / w0)**abs(l)
    laguerre_poly = genlaguerre(p, abs(l))(2.0 * rr**2 / w0**2)
    gaussian_term = np.exp(-rr**2 / w0**2)
    azimuthal_phase = np.exp(1j * l * phi)
    
    lg_field = norm_factor * radial_term * laguerre_poly * gaussian_term * azimuthal_phase
    params = {'p': p, 'l': l}
    return lg_field.astype(np.complex128), params