# @file: m2_utils.py
# Contains core utility functions for grid setup, M2 calculation, and plotting.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
import matplotlib.colors as colors

def setup_grid(grid_size, xy_max):
    """Sets up the computational grid."""
    x = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
    y = np.linspace(-xy_max, xy_max, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    return x, y, xx, yy, rr, phi

def calculate_m2_spatial_fft(psi, x_coords, y_coords):
    """Calculates M2 using the standard Spatial/FFT method."""
    if psi is None or not np.all(np.isfinite(psi)): return np.nan, np.nan
    dx = x_coords[1] - x_coords[0]
    I = np.abs(psi)**2
    I_total = np.sum(I)
    if I_total < 1e-15: return np.nan, np.nan
    
    x_mean = np.sum(x_coords[np.newaxis, :] * I) / I_total
    y_mean = np.sum(y_coords[:, np.newaxis] * I) / I_total
    var_x = np.sum((x_coords[np.newaxis, :] - x_mean)**2 * I) / I_total
    var_y = np.sum((y_coords[:, np.newaxis] - y_mean)**2 * I) / I_total
    if var_x < 0 or var_y < 0: return np.nan, np.nan
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
    if var_kx < 0 or var_ky < 0: return np.nan, np.nan
    sigma_kx = np.sqrt(var_kx)
    sigma_ky = np.sqrt(var_ky)
    
    Mx2 = 2 * sigma_x * sigma_kx
    My2 = 2 * sigma_y * sigma_ky
    return Mx2, My2

def plot_beam_profile(ax, x, y, psi, title, show_phase=True):
    """A helper function to plot beam intensity and phase on a given axis."""
    intensity = np.abs(psi)**2
    phase = np.angle(psi)
    
    # Plot Intensity
    im_int = ax.imshow(intensity, extent=[x.min(), x.max(), y.min(), y.max()],
                       cmap='inferno', origin='lower')
    
    if show_phase:
        # Plot Phase Contours
        contour_levels = np.linspace(-np.pi, np.pi, 9)
        ax.contour(x, y, phase, levels=contour_levels, colors='white', linewidths=0.5, alpha=0.7)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')