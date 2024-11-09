import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator
import matplotlib.pyplot as plt

# Constants and model parameters for Brown Dwarfs
A_bd_ln = -3.78
mean_bd = 1.43      #log10
sigma_bd = 1.21     #log10
A_bd = np.exp(A_bd_ln)
alpha_bd = -0.36

# Constants and model parameters for Giant Planets
alpha_gp = 1.43
A_pl_ln = -5.52
mu_natural = 1.32   #ln
sigma_pl_ln = 0.53  #ln
A_pl = np.exp(A_pl_ln)

# Conversion constant from natural log to log_10
constant = 2.302585092994046
mu_pl = mu_natural / constant     #log10
sigma_pl = sigma_pl_ln / constant #log10

# Display model parameters in Streamlit
st.header("Two Component Model for Exoplanet and Brown Dwarf Companions (Meyer et al. 2024)")
st_type = st.radio("Choose the Stellar Spectral Type:", ("M Dwarfs", "FGK", "A Stars"))

# User input for mass parameters
host_mass = st.number_input("Host Mass ($\mathrm{M_{\odot}}$)", 0.01, value=1.0)
Jup_min = st.slider("Companion Minimum Mass ($\mathrm{M_{Jup}}$)", min_value=0.01, max_value=50.0, value=1.0)
Jup_max = st.slider("Companion Maximum Mass ($\mathrm{M_{Jup}}$)", min_value=0.01, max_value=100.0, value=10.0)

# Mass ratio calculations
q_Jupiter = 0.001/host_mass
d_q = np.logspace(-5, 3, 500)  # Mass ratios from 0.0001 to 1 on a logarithmic scale

# Brown Dwarf model distribution (mass ratio)
a2_bd = d_q ** -alpha_bd

# Giant Planet model distribution (mass ratio)
a2_gp = d_q ** -alpha_gp

# Create plots for both distributions side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

# Plotting normalized mass ratio distributions
ax1.plot(d_q, a2_bd, color='r', label='Brown Dwarf Model')
ax1.plot(d_q, a2_gp, color='blue', label='Giant Planet Model')

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
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title("Mass Ratio Distributions", fontsize=18)

# User input for orbital separations using sliders
a_min = st.slider("Orbital Minimum Separation (AU)", min_value=0.01, max_value=100.0, value=1.0)
a_max = st.slider("Orbital Maximum Separation (AU)", min_value=0.01, max_value=2000.0, value=10.0)

if st_type == "M Dwarfs":
    s_m =  np.log(10**1.21)    # Winters
    mu_m = np.log(10**mean_bd)
elif st_type == "FGK":
    s_m = np.log(10**1.68)
    mu_m = np.log(50)
elif st_type == "A Stars":
    s_m = np.log(10**0.92)
    mu_m = np.log(522)            # De Rosa (x1.35 DM91)

# Define orbital separation distributions for Brown Dwarfs and Giant Planets
def orbital_dist_bd(a):
    return (np.exp(-(np.log(a) - mu_m) ** 2 / (2 * s_m ** 2))) #/ (np.sqrt(2 * np.pi) * s_m*a)

def orbital_dist_pl(a):
    return (np.exp(-(np.log(a) - mu_natural) ** 2 / (2 * sigma_pl_ln ** 2))) #/ (np.sqrt(2 * np.pi) * sigma_pl_ln*a)

# Create a log-scaled range for plotting
a_values = np.logspace(-6, 10, 500, base=2.71828)

# Plotting the normalized distributions
ax2.plot(np.log(a_values), orbital_dist_bd(a_values), label='Brown Dwarf Model', color='r')
ax2.plot(np.log(a_values), orbital_dist_pl(a_values), label='Giant Planet Model', color='blue')

# Add vertical lines for min and max separations
ax2.axvline(x=np.log(a_min), color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
ax2.axvline(x=np.log(a_max), color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')

ax2.set_ylim(0,1.5)
# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (ln AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_title("Semi-Major Axis Distributions", fontsize=18)

# Display the plots
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
mass_ratio_values = np.linspace(Jup_min * q_Jupiter, Jup_max * q_Jupiter, 500)
a_values_m = np.linspace(a_min,a_max, 500)

# Calculate frequency for Brown Dwarfs using integration with `np.trapz`
f_bd = A_bd * np.trapz([orbital_dist_bd(a)/(np.sqrt(2 * np.pi)*s_m*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_bd for d_q_i in mass_ratio_values], mass_ratio_values)

# Re-run integration over corrected distribution ranges
f_pl = A_pl * np.trapz([orbital_dist_pl(a)/(np.sqrt(2*np.pi)*sigma_pl_ln*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)

# Display results in Streamlit
st.write(f"Frequency of Planets:", f_pl)
st.write(f"Frequency of Brown Dwarfs:", f_bd)

# Sub-Jupiter Model
st.subheader("Sub-Jupiters (< 1 MJ)")
st.write(f"For planets less than 1 Jupiter mass, we made a customized sub-Jupiter model where the orbital distribution within 10 AU doubles the giant planet model (Fulton 2021), but log-flat with a constant density beyond 10 AU.")

# Create plots for both distributions side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

# Plotting mass ratio distributions
ax1.plot(d_q, a2_gp, color='blue', label='Giant Planet Model')

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
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title("Mass Ratio Distributions", fontsize=18)


# Define orbital separation distributions for sub-Jupiters
def orbital_dist_subJupiter(a):
    if a <= 10:
        return 2*(np.exp(-(np.log(a) - mu_natural) ** 2/(2 * 2*sigma_pl_ln ** 2)))#/(np.sqrt(2*np.pi)*sigma_pl_ln*a)
    else:
        return 0.8430271150978883

# Create a log-scaled range for plotting
a_values = np.logspace(-6, 10, 500, base=2.71828)  # Values from 0.01 to 1000, log-scaled
sub_jupiter_values = [orbital_dist_subJupiter(a) for a in a_values]
# Plotting the distributions
ax2.plot(np.log(a_values), sub_jupiter_values, label='Sub-Jupiter Model', color='orange')

# Add vertical lines for min and max separations
ax2.axvline(x=np.log(a_min), color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
ax2.axvline(x=np.log(a_max), color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')

# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (ln AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_title("Semi-Major Axis Distributions", fontsize=18)
ax2.set_ylim(0,3)
# Display the plots
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
mass_ratio_values = np.linspace(Jup_min * q_Jupiter, Jup_max * q_Jupiter, 500)
if a_min<=10 and a_max <=10:
    a_values_m = np.linspace(a_min,a_max, 500)
    f_subJ = A_pl * np.trapz([orbital_dist_subJupiter(a)/(np.sqrt(2*np.pi)*2*sigma_pl_ln*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
elif a_min<=10 and a_max>10:
    a_values_m1 = np.linspace(a_min,10, 500)
    f_subJ1 = A_pl * np.trapz([orbital_dist_subJupiter(a)/(np.sqrt(2*np.pi)*2*sigma_pl_ln*a) for a in a_values_m1], a_values_m1) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
    a_values_m2 = np.linspace(10,a_max, 500)
    f_subJ2 = A_pl * np.trapz([0.8430271150978883/(a*np.log(a_max/10)) for a in a_values_m2], a_values_m2) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
    f_subJ = f_subJ1 + f_subJ2
elif a_min>10 and a_max >10:
    a_values_m = np.linspace(a_min,a_max, 500)
    f_subJ = A_pl * np.trapz([0.8430271150978883/(a*np.log(a_max/a_min)) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
       
# Re-run integration over corrected distribution ranges


# Display results in Streamlit
st.write(f"Frequency of Sub-Jupiters:", f_subJ)
