# @title Frameworks_noise_robustness.py
# FINAL VERSION 2: Optimized with aggressive font sizes for single-column manuscript layout.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.fft import fft2, fftshift, ifftshift
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
N_runs = 50 

log_noise_levels = np.logspace(-3, -1, 20) 
linear_noise_levels = np.linspace(2e-4, 8e-4, 4)
noise_levels_to_test = np.unique(np.concatenate(([0.0], linear_noise_levels, log_noise_levels)))

# ==============================================================
# SECTION 2: ESSENTIAL FUNCTIONS (Condensed for brevity, no changes in logic)
# ==============================================================
def setup_grid(gs, xm):
    x=np.linspace(-xm,xm,gs);y=np.linspace(-xm,xm,gs);xx,yy=np.meshgrid(x,y)
    return x,y,xx,yy,np.sqrt(xx**2+yy**2),np.arctan2(yy,xx)
def calculate_gaussian_field(w0,xx,yy):
    p=np.sum(np.exp(-2*(xx**2+yy**2)/w0**2))*((2*xy_max)/(grid_size-1))**2
    return (np.exp(-(xx**2+yy**2)/w0**2)/np.sqrt(p)).astype(np.complex128)
def calculate_m2_spatial_fft(psi,x,y):
    dx=x[1]-x[0];I=np.abs(psi)**2;It=np.sum(I)
    if It < 1e-15: return np.nan, np.nan
    xm=np.sum(x[None,:]*I)/It;ym=np.sum(y[:,None]*I)/It;vx=np.sum((x[None,:]-xm)**2*I)/It;vy=np.sum((y[:,None]-ym)**2*I)/It
    sx,sy=np.sqrt(vx),np.sqrt(vy);pf=fftshift(fft2(ifftshift(psi)));If=np.abs(pf)**2;Ift=np.sum(If)
    kx,ky=2*np.pi*fftshift(np.fft.fftfreq(len(x),d=dx)),2*np.pi*fftshift(np.fft.fftfreq(len(y),d=dx))
    kxm=np.sum(kx[None,:]*If)/Ift;kym=np.sum(ky[:,None]*If)/Ift;vkx=np.sum((kx[None,:]-kxm)**2*If)/Ift;vky=np.sum((ky[:,None]-kym)**2*If)/Ift
    skx,sky=np.sqrt(vkx),np.sqrt(vky);return 2*sx*skx,2*sy*sky
def calculate_lg_field(p,l,w0,rr,phi):
    nf=math.sqrt(2.0*math.factorial(p)/(np.pi*math.factorial(p+abs(l))))/w0
    return (nf*(np.sqrt(2.0)*rr/w0)**abs(l)*genlaguerre(p,abs(l))(2.0*rr**2/w0**2)*np.exp(-rr**2/w0**2)*np.exp(1j*l*phi)).astype(np.complex128)
def generate_lg_basis_set(mo,w0,rr,phi):
    bm={};inds=[(p,l) for l in range(-mo,mo+1) for p in range((mo-abs(l))//2+1) if 2*p+abs(l)<=mo]
    for p,l in inds: bm[(p,l)]=calculate_lg_field(p,l,w0,rr,phi)
    return bm
def calculate_m2_from_coeffs_lg(c,k):
    tp=np.sum(np.abs(c)**2); return np.sum([np.abs(c[i])**2/tp*(2*p+abs(l)+1) for i,(p,l) in enumerate(k)]) if tp>1e-15 else np.nan

# ==============================================================
# SECTION 3: MAIN EXPERIMENT LOOP
# ==============================================================
logger.info("Setting up grid and pre-calculating basis set...")
x_coords,y_coords,xx,yy,rr,phi=setup_grid(grid_size,xy_max)
psi_ideal_gaussian=calculate_gaussian_field(w0_gaussian,xx,yy)
basis_modes=generate_lg_basis_set(lg_basis_order,w0_gaussian,rr,phi)
mode_keys=list(basis_modes.keys()); Psi_matrix=np.stack([basis_modes[k].flatten() for k in mode_keys],axis=1)
mean_traditional,std_traditional,mean_framework,std_framework=[],[],[],[]

logger.info(f"Starting M² vs. Noise sweep for {len(noise_levels_to_test)} levels, with {N_runs} runs per level...")
for i, noise_std in enumerate(noise_levels_to_test):
    per_level_trad_m2s,per_level_framework_m2s=[],[]
    num_runs_this_level = 1 if noise_std == 0.0 else N_runs
    for run_num in range(num_runs_this_level):
        psi_current=psi_ideal_gaussian if noise_std==0.0 else psi_ideal_gaussian+np.random.normal(0,noise_std,psi_ideal_gaussian.shape)+1j*np.random.normal(0,noise_std,psi_ideal_gaussian.shape)
        Mx2_trad,_=calculate_m2_spatial_fft(psi_current,x_coords,y_coords)
        per_level_trad_m2s.append(Mx2_trad)
        y_vector=psi_current.flatten()
        coeffs,_,_,_ = np.linalg.lstsq(Psi_matrix, y_vector, rcond=rcond_solver)
        m2_coeff=calculate_m2_from_coeffs_lg(coeffs,mode_keys)
        per_level_framework_m2s.append(m2_coeff)
    mean_traditional.append(np.nanmean(per_level_trad_m2s)); std_traditional.append(np.nanstd(per_level_trad_m2s))
    mean_framework.append(np.nanmean(per_level_framework_m2s)); std_framework.append(np.nanstd(per_level_framework_m2s))
    logger.info(f"Level {i+1}/{len(noise_levels_to_test)} (Noise Std={noise_std:.4f}): Trad: {mean_traditional[-1]:.2f} ± {std_traditional[-1]:.2f}, Framework: {mean_framework[-1]:.4f} ± {std_framework[-1]:.4f}")

# ==============================================================
# SECTION 4: PLOTTING (WITH AGGRESSIVE FONTS FOR SINGLE COLUMN)
# ==============================================================
logger.info("Plotting results with larger fonts for single-column format...")
plt.style.use('seaborn-v0_8-whitegrid')
# Use a more standard figure size that fits well in a column
fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(noise_levels_to_test,mean_traditional,yerr=std_traditional,fmt='o',markersize=6,color='crimson',capsize=4,elinewidth=1.5,label='Traditional Method')
ax.errorbar(noise_levels_to_test,mean_framework,yerr=std_framework,fmt='s',markersize=6,color='royalblue',capsize=4,elinewidth=1.5,label='Framework Method')

ax.set_yscale('log'); ax.set_xscale('symlog',linthresh=1e-3); ax.set_xlim(left=-0.0001)
ax.set_xlabel('Noise Level (Std. Dev.)',fontsize=22)
ax.set_ylabel(r'$M^2$ Factor (Log Scale)',fontsize=22)
ax.tick_params(axis='both',which='major',labelsize=20)
ax.axhline(y=1.0,color='gray',linestyle=':',linewidth=2.5,label=r'Ideal $M^2=1$')
ax.legend(fontsize=20)

plt.tight_layout()
plt.savefig("M2_vs_Noise_with_ErrorBars.png", dpi=300)
plt.show()

logger.info("Script finished.")