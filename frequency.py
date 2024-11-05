import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator
import matplotlib.pyplot as plt


# Constants and model parameters for Brown Dwarfs
A_bd_ln = -4.08  # Amplitude for Brown Dwarfs (natural log scale)
mean_bd = 1.43   # Mean of the log-normal distribution for M dwarfs
sigma_bd = 1.21  # Standard deviation of the log-normal distribution
A_bd = np.exp(A_bd_ln)  # Convert amplitude to linear scale

# Constants and model parameters for Giant Planets
A_pl_ln = -5.28  # Amplitude for Giant Planets (natural log scale)
mu_ln = 1.31     # Mean in natural log scale
sigma_pl_ln = 0.52  # Standard deviation in natural log scale

# Conversion constant from natural log to log_10
constant = 2.302585092994046

# Convert parameters for Giant Planets to log_10 scale
A_pl = np.exp(A_pl_ln)  # Linear amplitude for Giant Planets
mu_pl = mu_ln / constant  # Mean in log_10 scale
sigma_natural = np.exp(sigma_pl_ln)  # Standard deviation in natural scale
sigma_pl = sigma_natural / constant  # Convert to log_10 scale

# Display model parameters in Streamlit
st.header("Two Component Model for Exoplanet and Brown Dwarf Companions (Meyer et al. 2024)")
st.write(r"$A_{pl}$:", A_pl)
st.write(r"$\sigma_{pl}$:", sigma_pl)
st.write(r"$\mu_{pl}$:", mu_pl)
st.write(r"$A_{bd}$:", A_bd)
st.write(r"$\sigma_{bd}$:", sigma_bd)
st.write(r"$\mu_{bd}$:", mean_bd)

# User input for mass parameters
host_mass = st.number_input("Host Mass ($\mathrm{M_{\odot}}$)", 0.01, None)
Jup_min = st.slider("Companion Minimum Mass ($\mathrm{M_{Jup}}$)", min_value=0.01, max_value=50.0, value=1.0)
Jup_max = st.slider("Companion Maximum Mass ($\mathrm{M_{Jup}}$)", min_value=0.01, max_value=100.0, value=10.0)

# Mass ratio calculations
q_Jupiter = 0.001 / host_mass
d_q = np.logspace(-5, 3, 500)  # Mass ratios from 0.0001 to 1 on a logarithmic scale
a1 = d_q

# Brown Dwarf model distribution (mass ratio)
alpha_bd = -0.3
a2_bd = d_q ** alpha_bd
#area_bd = np.trapz(a2_bd, d_q)  # Compute area under the curve for normalization

# Giant Planet model distribution (mass ratio)
alpha_gp = 1.39
a2_gp = d_q ** -alpha_gp

# Create plots for both distributions side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plotting normalized mass ratio distributions
ax1.plot(a1, a2_bd, color='blue', label='Brown Dwarf Model', linewidth=3)
ax1.plot(a1, a2_gp, color='orange', label='Giant Planet Model', linewidth=3)

ax1.set_xscale('log')
ax1.set_yscale('log')

# Calculate mass ratio limits
mass_ratio_min = Jup_min * q_Jupiter
mass_ratio_max = Jup_max * q_Jupiter

# Add vertical lines for mass ratio limits
ax1.axvline(x=mass_ratio_min, color='green', linestyle='--', label=f'Min Mass Ratio = {mass_ratio_min:.2f}')
ax1.axvline(x=mass_ratio_max, color='purple', linestyle='--', label=f'Max Mass Ratio = {mass_ratio_max:.2f}')

# Configure mass ratio distribution plot
ax1.set_xlabel('Mass Ratio q', fontsize=20, labelpad=10.4)
ax1.set_ylabel('Normalized Probability Density', fontsize=20, labelpad=10.4)
ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_xlim(np.log10(0.0001), np.log10(1))  # Adjust x-limits
#ax1.set_ylim(0, 1.1)
ax1.legend(loc='upper right', fontsize=14)
ax1.set_title("Mass Ratio Distributions", fontsize=18)


# User input for orbital separations using sliders
a_min = st.slider("Orbital Minimum Separation (AU)", min_value=0.01, max_value=100.0, value=1.0)
a_max = st.slider("Orbital Maximum Separation (AU)", min_value=0.01, max_value=100.0, value=10.0)

# Define orbital separation distributions for Brown Dwarfs and Giant Planets
# Update the orbital distribution functions to include amplitude
def orbital_dist_bd(a):
    return A_bd * (np.exp(-(np.log10(a) - mean_bd) ** 2 / (2 * sigma_bd ** 2))) / (np.sqrt(2 * np.pi) * sigma_bd)

def orbital_dist_pl(a):
    return A_pl * (np.exp(-(np.log10(a) - mu_pl) ** 2 / (2 * sigma_pl ** 2))) / (np.sqrt(2 * np.pi) * sigma_pl)

# Ensure you have a fine range for plotting
a_values = np.logspace(-4, 3, 500)  # Use logspace for better resolution
ax2.plot(a_values, orbital_dist_bd(a_values), label='Brown Dwarf Distribution', color='C2')
ax2.plot(a_values, orbital_dist_pl(a_values), label='Giant Planet Distribution', color='C3')

# Add vertical lines for min and max separations
ax2.axvline(x=a_min, color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
ax2.axvline(x=a_max, color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')

# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(loc='upper right')
ax2.set_title("Semi-Major Axis Distributions", fontsize=18)

# Display the plots
st.pyplot(fig)

# Frequency calculations using np.trapz
f_bd = (np.trapz([orbital_dist_bd(a) for a in np.linspace(a_min, a_max)], dx=0.01) *
        np.trapz([d_q_i * Jup_min * q_Jupiter for d_q_i in d_q], dx=0.01))
f_pl = (np.trapz([orbital_dist_pl(a) for a in np.linspace(a_min, a_max)], dx=0.01) *
        np.trapz([d_q_i * Jup_max * q_Jupiter for d_q_i in d_q], dx=0.01))

# Display results
st.text(f"Frequency of Planets: {f_pl}")
st.text(f"Frequency of Brown Dwarfs: {f_bd}")

#
## Written by Michael R. Meyer, University of Michigan
## Updated June 30, 2024 
## New Update October 28, 2024 
#
## The parameters of the two component model from Meyer et al. (2024)
#
## Brown Dwarf Companions 
## beta is the power-law exponent for the brown dwarf binary companion mass ratio distribution dN/dq ~ q^(beta) Meyer et al. 2024
#
#beta = -0.3
#
## 68 % confidence intervals (hereafter CI) for beta [-0.34, 0.30]
#
## mean_bd = 1.70 is the mean of the log-10-normal in log-AU log(50) Raghavan et al. 2010 (physical separation) for FGK stars
## mean_bd = 1.43 for M dwarfs log-10(27) Winters et al. (2019) - physical separation not projected. Meyer et al. 2024 
## mean_bd = 2.72 for A stars log-10(525) De Rosa et al. (2014) - physical separation not projected. Meyer et al. 2024
#
#mean_bd = 1.43
#
## square-root of the variance for companion orbital density log-10-normal distribution. R10 sigma_bd = 1.68 
## sigma_bd = 1.21 M dwarf Winters et al. (2019) aghavan et al. 20
## sigmba_bd = 0.79 A stars De Rosa et al. (2014)
#
#sigma_bd = 1.21
#
## amplitude of the BD new fit 
#A_bd_ln = -4.08 # with 68 % CI [-0.47,0.7]
#
#A_bd = np.exp(A_bd_ln)
#
## power-law planet mass function values in q for M-FGK-A stars Meyer et al. 2024   
## Note power-law is fit as (- alpha) so alpha here is positive, but multiplied by (-1) in mass function below. 
#
#alpha_pl = 1.39 # 68 % CI [-0.13,0.12] 
#
## In this code, we transform mu and sigma from natural log to log_10 in the orbital separation distribution.
## sigma is log_base10 of log-normal while fits are in natural ln. 
## sigma_logbase10 = sigma_naturallog / constant where constant = ln(10) = 2.302585092994046  
## Meyer et al. (2024) 6-25-24
##
#A_pl_ln = -5.28 # 68 % CI [-0.83,0.93] 
#mu_ln = 1.31  # 68 % CI [-0.22,0.18] 
#sigma_pl_ln = 0.52 # 68 % CI [-0.07,0.06]
#
#constant = 2.302585092994046
#
#A_pl = np.exp(A_pl_ln)
#mu_pl = (mu_ln)/constant
#sigma_natural = np.exp(sigma_pl_ln)
#sigma_pl = sigma_natural/constant
#print(A_pl, sigma_pl, mu_pl, A_bd)
#
#st.header("Two Component Model for Exoplanet and Brown Dwarf Companions (Meyer et al. 2024)")
#st.write(r"$A_{pl}$:", A_pl)
#st.write(r"$\sigma_{pl}$:", sigma_pl)
#st.write(r"$\mu_{pl}$:", mu_pl)
#st.write(r"$A_{bd}$:", A_bd)
#st.write(r"$\sigma_{bd}$:", sigma_bd)
#st.write(r"$\mu_{bd}$:", mean_bd)
#
#
## User input parameter host_mass in Solar masses, companion mass range of interest in Jupiter masses (min, max) 
## and orbital range of interest in AU (min, max) 
#
#host_mass = st.number_input("Host Mass ($\mathrm{M_{\odot}}$)", 0.01, None, step=None, format=None)
#Jup_min = st.number_input("Companion Minimum Mass ($\mathrm{M_{Jup}}$)", 0.01, None, step=None, format=None)
#Jup_max = st.number_input("Companion Maximum Mass ($\mathrm{M_{Jup}}$)", 0.01, None, step=None, format=None)
#q_Jupiter = 0.001/host_mass # Msun
#
#a_min = st.number_input("Orbital Minimum Separation (AU)", 0.01, None, step=None, format=None)
#a_max = st.number_input("Orbital Maximum Separation (AU)", 0.01, None, step=None, format=None)
#
## Defining the functions for mass and orbital separation distibutions for both brown dwarf and planets 
#
#def mass_fctn_bd(q):
#    return q**(beta)#dq
#def orbital_dist_bd(a):
#    return (A_bd*np.exp((-(np.log10(a)-(mean_bd))**2.)/(2.*sigma_bd**2.)))/(np.sqrt(2.0*np.pi)*sigma_bd)#da
#def mass_fctn_pl(q):
#    return q**(-alpha_pl)#dm
#def orbital_dist_pl(a):
#     return (A_pl*np.exp((-(np.log10(a)-(mu_pl))**2.)/(2.*sigma_pl**2.)))/(2.0*np.pi*sigma_pl*a)#da
#
## mass function ranges are given in Jupiter masses, but passed in q (which depends on host star mass)
## orbital distributions are given in AU. 
## st.text( np.log10(Jup_min*q_Jupiter))
#f_bd = (integrate.quad(mass_fctn_bd, (Jup_min*q_Jupiter), (Jup_max*q_Jupiter))[0]*integrate.quad(orbital_dist_bd,a_min,a_max)[0])
#
#f_pl = (integrate.quad(mass_fctn_pl, (Jup_min*q_Jupiter), (Jup_max*q_Jupiter))[0]*integrate.quad(orbital_dist_pl,a_min,a_max)[0])
#
##Output is the mean number of companions per star for the brown dwarf component (f_ bd) and planet part (f_pl).
#
#print(f_bd,f_pl)
#st.text("Frequency of Planets: {}".format(f_pl))
#st.text("Frequency of Brown Dwarfs: {}".format(f_bd))
#
