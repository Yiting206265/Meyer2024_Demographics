import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#not using
## Constants and model parameters for Brown Dwarfs
#A_bd_ln = -3.78
#mean_bd = 1.43      #log10
#sigma_bd = 1.21     #log10
#A_bd = np.exp(A_bd_ln)
#alpha_bd = -0.36
#
## Constants and model parameters for Giant Planets
#alpha_gp = 1.43
#A_pl_ln = -5.52
#mu_natural = 1.32   #ln
#sigma_pl_ln = 0.53  #ln
#A_pl = np.exp(A_pl_ln)

# Start of code

# Display title
st.title("Composite Model for Exoplanet and Brown Dwarf Companions")
st.caption("Based on Meyer et al. (2025)")

##############################################################################
#Section 1 - Intro Paragraph
##############################################################################


st.write("""Welcome to the on-line tool based on Meyer et al. (submitted) meant to provide estimates of the expectation values of the mean number of gas giant planets per star and the mean number of brown dwarfs per star generated from our model. The model assumes that the companion mass ratio of gas giants and brown dwarf companions does not vary with orbital separation. However, it explicitly treats brown dwarf companions as an extension of stellar mass companions drawn from the same orbital separations as a function of host star mass. 

In the paper we fit the orbital distribution of gas giants and find that a log-normal function provides a good fit, with a peak near 3 AU (two parameters). We also fit for power-law exponents for the companion mass ratio distributions for the brown dwarf companions and gas giant populations separately (two parameters). Finally, we fit for the normalization of both populations (two parameters).  

The data are fitted to 50 point estimates of companion frequency over specified mass ranges and orbital separations found in the literature. Please read the paper available at (archive) for details. As our model is fitted in the context of distributions of mass ratios of companions to host stars, you need to select the stellar mass of the host, as well as model parameters for the function form of the fits (best fits from our paper are defaults). Finally one must choose the mass range of companion of interest, as well as the orbital separation range of interest. The model is integrated to provide the expectation value of our model over the ranges indicated for both populations.  

If you are interested in using this model for a given target list and sensitivity curves to predict survey yields, please contact us to learn more.""")


##############################################################################
#Section 2 - Host Star Parameters
##############################################################################

st.subheader("Host Star Parameters")


# Radio button to select stellar type
st_type = st.radio(
    "Select a Stellar Spectral Type to update the brown dwarf log-normal distributions: M dwarfs follow Winters et al. (2019); FGK stars follow Raghavan et al. (2010); and A stars follow De Rosa et al. (2014), corrected for physical separation (x1.35 DM91). Updated values will be reflected in the sliders and graphs below. Note that you can fine-tune the slider values using the left/right arrow keys.",
    ("M Dwarfs", "FGK", "A Stars")
)

#the session updates are delayed, displays previous values
if st_type == "M Dwarfs":
    s_bd = 1.21
    mu_bd = 1.43             #Winters
elif st_type == "FGK":
    s_bd = 1.68
    mu_bd = 1.70
elif st_type == "A Stars":
    s_bd = 0.79
    mu_bd = 2.72            # De Rosa (x1.35 DM91)

s_m =  np.log(10**s_bd)    # Winters
mu_m = np.log(10**mu_bd)

# User input for mass parameters
host_mass = st.number_input("Host Mass ($\mathrm{M_{\odot}}$)", min_value=0.0001, max_value=10.0, value=0.3, step=0.001)

##############################################################################
#Section 3 - Model Parameters
##############################################################################
# Create columns for a vertical arrangement of sliders
st.subheader("Model Parameters")
st.write(
    "Define the variables for the parametric model(s) of planets and brown dwarfs. "
    r"$\alpha$ represents the power-law index for the companion-to-star mass ratio ($q$), "
    r"$A$ is the normalization factor, and $\mu$ and $\sigma$ are the mean and standard deviation of the log-normal orbital distribution."
)
col1, col2 = st.columns(2)

# Conversion constant from natural log to log_10
constant = 2.302585092994046
mu_natural = 1.30    #ln
sigma_pl_ln = 0.215   #ln
mu_pl_value = mu_natural/constant     #log10
sigma_pl_value = sigma_pl_ln/constant #log10

#if "mean_bd" not in st.session_state:
#    st.session_state.mean_bd = 1.43
#if "sigma_bd" not in st.session_state:
#    st.session_state.sigma_bd = 1.21
#if "stellar_type" not in st.session_state:
#    st.session_state.stellar_type = "M Dwarfs"
#if "force_refresh" not in st.session_state:
#    st.session_state.force_refresh = False  # Initialize the flag for forcing refresh

# Brown Dwarf parameters in col1
with col1:
    alpha_bd = st.slider(r'$\mathrm{\alpha_{bd}}$', min_value=-2.0, max_value=2.0, value=-0.292, step=0.01)
    A_bd_ln = st.slider(r'$\mathrm{ln(A_{bd})}$', min_value=-10.0, max_value=5.0, value=-1.41, step=0.01)
    mean_bd = st.slider(
        r'$\mathrm{log_{10}(\mu_{bd})}$',
        min_value=0.0,
        max_value=3.0,
        value=mu_bd,
        step=0.01,
        key="mean_bd_slider"
    )
    sigma_bd = st.slider(
        r'$\mathrm{log_{10}(\sigma_{bd})}$',
        min_value=0.0,
        max_value=3.0,
        value=s_bd,
        step=0.01,
        key="sigma_bd_slider"
    )

# Giant Planet parameters in col2
with col2:
    alpha_gp = st.slider(r'$\mathrm{\alpha_{pl}}$', min_value=0.0, max_value=3.0, value=1.29, step=0.01)
    A_pl_ln = st.slider(r'$\mathrm{ln(A_{pl})}$', min_value=-10.0, max_value=5.0, value=-4.77, step=0.01)
    mu_pl = st.slider(
        r'$\mathrm{log_{10}(\mu_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=mu_pl_value,
        step=0.01
    )
    sigma_pl = st.slider(
        r'$\mathrm{log_{10}(\sigma_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=sigma_pl_value,  # Default or static value
        step=0.01
    )

# Calculate updated values
A_bd = np.exp(A_bd_ln)
A_pl = np.exp(A_pl_ln)

##############################################################################
#Section 4 - Companion Parameters
##############################################################################

st.subheader("Companion Parameters")
Jup_min, Jup_max = st.slider(
    "Companion Mass Range ($\mathrm{M_{Jup}}$)",
    min_value=0.01,
    max_value=300.0,
    value=(1.0, 100.0))

# Mass ratio calculations
q_Jupiter = 0.001/host_mass
d_q_gp = np.logspace(np.log10(0.1*q_Jupiter), np.log10(0.1), 500) # 30 Earth mass to 0.1 host star mass
d_q_bd = np.logspace(np.log10(3*q_Jupiter), np.log10(1), 500)  # 3 Jupiter mass to 1 host star mass

# Brown Dwarf model distribution (mass ratio)
a2_bd = d_q_bd ** -alpha_bd
# Giant Planet model distribution (mass ratio)
a2_gp = d_q_gp ** -alpha_gp

# Create plots for both distributions side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
# Plotting normalized mass ratio distributions
ax1.plot(d_q_bd, a2_bd, color='r', linewidth = 2,label='Brown Dwarf Model')
ax1.plot(d_q_gp, a2_gp, color='blue',linewidth = 2, label='Giant Planet Model')

ax1.set_xscale('log')
ax1.set_yscale('log')

# Create a custom formatter for the x-axis
def log_formatter(x, pos):
    return f'{x:.3g}'  # Format with general float, trims unnecessary zeros

# Apply the custom formatter for the x-axis
ax1.xaxis.set_major_formatter(FuncFormatter(log_formatter))


# Calculate mass ratio limits
mass_ratio_min = Jup_min * q_Jupiter
mass_ratio_max = Jup_max * q_Jupiter

# Add vertical lines for mass ratio limits
ax1.axvline(x=mass_ratio_min, color='green', linestyle='--', label=f'Min Mass Ratio = {mass_ratio_min:.2f}')
ax1.axvline(x=mass_ratio_max, color='purple', linestyle='--', label=f'Max Mass Ratio = {mass_ratio_max:.2f}')

ax1.axvspan(mass_ratio_min, mass_ratio_max, color='gray', alpha=0.3)

# Configure mass ratio distribution plot
ax1.set_xlabel('Mass Ratio q', fontsize=20, labelpad=10.4)
ax1.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title("Mass Ratio Distribution", fontsize=18)

# User input for orbital separations using sliders
a_min, a_max = st.slider(
    "Orbital Separation Range (AU)",
    min_value=0.01,
    max_value=3000.0,
    value=(1.0, 100.0))

# Define orbital separation distributions for Brown Dwarfs and Giant Planets in natural log units
def orbital_dist_bd(a):
    return (np.exp(-(np.log(a) - mu_m) ** 2 / (2 * s_m**2))) #/ (np.sqrt(2 * np.pi) * s_m*a)

def orbital_dist_pl(a):
    return (np.exp(-(np.log(a) - mu_natural) ** 2 / (2 * sigma_pl_ln**2))) #/ (np.sqrt(2 * np.pi) * sigma_pl_ln*a)

# Create a log-scaled range for plotting
a_values = np.logspace(-3, 5, 500, base=10)

# Plotting the distributions
ax2.plot(np.log10(a_values), orbital_dist_bd(a_values), linewidth = 2, label='Brown Dwarf Model', color='r')
ax2.plot(np.log10(a_values), orbital_dist_pl(a_values), linewidth = 2, label='Giant Planet Model', color='blue')

def log10_formatter(x, pos):
    return r"$10^{{{:.0f}}}$".format(x)

# Add vertical lines for min and max separations
ax2.axvline(x=np.log10(a_min), color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
ax2.axvline(x=np.log10(a_max), color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')
ax2.axvspan(np.log10(a_min), np.log10(a_max), color='gray', alpha=0.3)

ax2.xaxis.set_major_formatter(FuncFormatter(log10_formatter))

ax2.set_ylim(0,1.5)
# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_title("Semi-Major Axis Distribution", fontsize=18)

# Display the plots
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
mass_ratio_values = np.linspace(Jup_min * q_Jupiter, Jup_max * q_Jupiter, 500)
a_values_m = np.linspace(a_min,a_max, 500)

# Calculate frequency for Brown Dwarfs using integration with `np.trapz`
f_bd = A_bd*np.trapz([orbital_dist_bd(a)/(np.sqrt(2 * np.pi)*s_m*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_bd for d_q_i in mass_ratio_values], mass_ratio_values)

# Re-run integration over corrected distribution ranges
f_pl = A_pl*np.trapz([orbital_dist_pl(a)/(np.sqrt(2*np.pi)*sigma_pl_ln*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)

# Display results in Streamlit
st.write(f"Mean Number of Planets Per Star:", f_pl)
st.write(f"Mean Number of Brown Dwarfs Per Star:", f_bd)
