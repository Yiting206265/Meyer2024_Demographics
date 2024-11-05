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
mu_natural = 1.31     # Mean in natural log scale
sigma_pl_ln = 0.52  # Standard deviation in natural log scale

# Conversion constant from natural log to log_10
constant = 2.302585092994046

# Convert parameters for Giant Planets to log_10 scale
A_pl = np.exp(A_pl_ln)  # Linear amplitude for Giant Planets
mu_pl = mu_natural / constant  # Mean in log_10 scale
sigma_natural = np.exp(sigma_pl_ln)  # Standard deviation in natural scale
sigma_pl = sigma_natural / constant # Convert to log_10 scale

# Display model parameters in Streamlit
st.header("Two Component Model for Exoplanet and Brown Dwarf Companions (Meyer et al. 2024)")
st.write(r"$A_{pl}$:", A_pl)
st.write(r"$\sigma_{pl}$:", sigma_pl)
st.write(r"$\mu_{pl}$:", mu_pl)
st.write(r"$A_{bd}$:", A_bd)
st.write(r"$\sigma_{bd}$:", sigma_bd)
st.write(r"$\mu_{bd}$:", mean_bd)

# User input for mass parameters
host_mass = st.number_input("Host Mass ($\mathrm{M_{\odot}}$)", 0.01, value=1.0)
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

# Plotting normalized mass ratio distributions
ax1.plot(a1, a2_bd, color='r', label='Brown Dwarf Model')
ax1.plot(a1, a2_gp, color='blue', label='Giant Planet Model')

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
ax1.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_xlim(np.log10(0.0001), np.log10(1))  # Adjust x-limits
#ax1.set_ylim(0, 1.1)
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title("Mass Ratio Distributions", fontsize=18)

# User input for orbital separations using sliders
a_min = st.slider("Orbital Minimum Separation (AU)", min_value=0.01, max_value=100.0, value=1.0)
a_max = st.slider("Orbital Maximum Separation (AU)", min_value=0.01, max_value=1000.0, value=10.0)

s_m =  np.log(10**1.21)    # Winters
mu_m = np.log(10**mean_bd)                # Winters (x1.35 DM91)
# ~ s_m = np.log(10**.47)        # Janson
# ~ mu_m = np.log(10**.78)    # Janson
# ~ s_m = np.log(10**0.97)            # Susemiehl
# ~ mu_m = np.log(10**1.68)        #  Susemiehl
s_fgk = np.log(10**1.68)
mu_fgk = np.log(50)
s_a = np.log(10**0.92)
mu_a = np.log(522)            # De Rosa (x1.35 DM91)

# Define orbital separation distributions for Brown Dwarfs and Giant Planets
def orbital_dist_bd(a):
    return (np.exp(-(np.log(a) - mu_m) ** 2 / (2 * s_m ** 2))) #/ (np.sqrt(2 * np.pi) * s_m*a)

def orbital_dist_pl(a):
    return (np.exp(-(np.log(a) - mu_natural) ** 2 / (2 * sigma_pl_ln ** 2))) #/ (np.sqrt(2 * np.pi) * sigma_pl_ln*a)

# Create a log-scaled range for plotting
a_values = np.logspace(-6, 6, 500, base=2.71828)  # Values from 0.01 to 1000, log-scaled

# Plotting the normalized distributions
ax2.plot(np.log(a_values), orbital_dist_bd(a_values), label='Brown Dwarf Model', color='r')
ax2.plot(np.log(a_values), orbital_dist_pl(a_values), label='Giant Planet Model', color='blue')

# Add vertical lines for min and max separations
ax2.axvline(x=np.log(a_min), color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
ax2.axvline(x=np.log(a_max), color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')

# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (ln AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(loc='upper left', fontsize=12)
ax2.set_title("Semi-Major Axis Distributions", fontsize=18)

# Display the plots
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
mass_ratio_values = np.linspace(Jup_min * q_Jupiter, Jup_max * q_Jupiter, 500)
a_values_m = np.linspace(a_min,a_max, 500)

# Calculate frequency for Brown Dwarfs using integration with `np.trapz`
f_bd = A_bd * np.trapz([orbital_dist_bd(a)/(np.sqrt(2 * np.pi)*s_m*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** alpha_bd for d_q_i in mass_ratio_values], mass_ratio_values)

# Re-run integration over corrected distribution ranges
f_pl = A_pl * np.trapz([orbital_dist_pl(a)/(np.sqrt(2*np.pi)*sigma_pl_ln*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)

# Display results in Streamlit
st.text(f"Frequency of Planets: {f_pl}")
st.text(f"Frequency of Brown Dwarfs: {f_bd}")

## Sub-Jupiter Model
#st.subheader("Sub-Jupiters (< 1 MJ)")
## Constants and parameters for Sub-Jupiter model
#A_sub_jupiter = A_pl   # Amplitude for Sub-Jupiter model (adjust as needed)
#transition_radius = 10.0  # AU, transition point from super-Jupiter to log-flat
#log_flat_value = 0.00235  # Normalized log-flat value
#
#
#def sub_jupiter_distribution(a):
#    if a <= transition_radius:
#        return A_sub_jupiter * np.exp(-(np.log10(a) - mu_pl) ** 2 / (2 * sigma_pl ** 2)) / (np.sqrt(2 * np.pi) * sigma_pl)
#    else:
#        return log_flat_value
#
## Prepare data for Sub-Jupiter model plot
#a_values_sub_jupiter = np.linspace(0.01, 100, 500)  # Define a range for semi-major axes
#sub_jupiter_values = [sub_jupiter_distribution(a) for a in a_values_sub_jupiter]
#
## Create a separate plot for the Sub-Jupiter model
#fig_sub_jupiter, ax_sub_jupiter = plt.subplots(figsize=(10, 6))
#ax_sub_jupiter.plot(a_values_sub_jupiter, sub_jupiter_values, label='Sub-Jupiter Model', color='orange')
#ax_sub_jupiter.set_xscale('log')
#ax_sub_jupiter.set_yscale('log')
#
## Configure the plot
#ax_sub_jupiter.set_xlabel('Semi-Major Axis (AU)', fontsize=16)
#ax_sub_jupiter.set_ylabel('Probability Density', fontsize=16)
#ax_sub_jupiter.set_title("Sub-Jupiter Model Distribution", fontsize=18)
#ax_sub_jupiter.legend()
#
## Display the Sub-Jupiter model plot
#st.pyplot(fig_sub_jupiter)
#
## Final yield calculation for Sub-Jupiter model
#yield_sub_jupiter = np.trapz(sub_jupiter_values, a_values_sub_jupiter)
#st.text(f"Freuqency of Sub-Jupiters: {yield_sub_jupiter:.5f}")
#
