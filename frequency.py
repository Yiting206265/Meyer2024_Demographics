"""
Occurrence Rate Estimator for Giant Planets and Brown Dwarfs

This Streamlit application calculates and visualizes the frequency of giant planets and 
brown dwarfs around different stellar types based on a companion population model.
The model uses log-normal distributions for orbital separations and power-law distributions
for mass ratios to estimate companion frequencies.

Author: Yiting Li
Date: August 2025
Paper Reference: Meyer et al. 2025, Demographics of Planetary and Brown Dwarf Companions
"""

import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Constants
ln10 = np.log(10)  # Natural log of 10 for proper normalization of log10 distributions

# Add global CSS for consistent styling throughout the app
st.markdown("""
<style>
/* Main title styling */
.title-container {
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.main-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}
.subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 18px;
    text-align: center;
    font-style: italic;
}

/* Math equation styling */
.centered-equation {
    text-align: center;
    margin: 20px 0;
    font-size: 18px;
}

/* Section header styling */
.section-header {
    color: #1E88E5;
    font-size: 28px;
    font-weight: bold;
    margin: 30px 0 20px 0;
    padding: 10px;
    border-bottom: 2px solid #1E88E5;
    text-align: left;
}

/* Subsection header styling */
.subsection-header {
    color: #5E35B1;
    font-size: 22px;
    font-weight: bold;
    margin: 20px 0 15px 0;
    padding: 8px;
    border-left: 4px solid #5E35B1;
    background-color: #f8f9fa;
    padding-left: 15px;
    border-radius: 0 5px 5px 0;
}

/* Table styling */
div.stTable {
    border-radius: 5px;
    overflow: hidden;
    box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
}

/* Tool section styling */
.tool-header {
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
    text-align: center;
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 12px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>

<!-- Main title -->
<div class='title-container'>
    <div class='main-title'>Occurrence Rate Estimator</div>
    <div class='subtitle'>for Planets and Brown Dwarfs</div>
    <div class='subtitle'>Companion population model from Meyer et al. (2025)</div>
</div>
""", unsafe_allow_html=True)

##############################################################################
#Section 1 - Intro Paragraph
##############################################################################


st.write("""Welcome to the on-line tool based on Meyer et al. (submitted) meant to provide estimates of the expectation values of the mean number of gas giant planets per star and the mean number of brown dwarfs per star generated from our model. The model assumes that the companion mass ratio of gas giants and brown dwarf companions does not vary with orbital separation. However, it explicitly treats brown dwarf companions as an extension of stellar mass companions drawn from the same orbital separations as a function of host star mass. 

In the paper we fit the orbital distribution of gas giants and find that a log-normal function provides a good fit, with a peak near 3 AU (two parameters). We also fit for power-law exponents for the companion mass ratio distributions for the brown dwarf companions and gas giant populations separately (two parameters). Finally, we fit for the normalization of both populations (two parameters).""")

##############################################################################
#Section 1.5 - Mathematical Model
##############################################################################

st.markdown("<div class='section-header'>Mathematical Model</div>", unsafe_allow_html=True)

st.write(r"""
Our model combines two components: gas giant planets and brown dwarf companions. The total companion frequency is expressed as:
""")

# Use st.latex for proper rendering of equations
st.latex(r"N_{TOTAL} = \int{\phi_{p}(x) \times \psi_{p}(q) dq dx} + \int{\phi_{bd}(x) \times \psi_{bd}(q) dq dx}")

st.write(r"""
where $\phi_{p}(x)$ and $\phi_{bd}(x)$ are log-normal orbital separation distributions for planets and brown dwarfs respectively, and $\psi_{p}(q)$ and $\psi_{bd}(q)$ are power-law mass ratio distributions.

For planets: 
""")

# Use st.latex for proper rendering of equations
st.latex(r"\phi_{p}(x) = \frac{A_{p} e^{-(x - \mu_p)^2/2 \sigma_p^2}}{\sqrt{2\pi}\sigma_p \times \ln(10)}")

st.write(r"""
where $x = \log_{10}(a)$

For brown dwarfs, we adopt log-normal distributions from the literature for different stellar types (M, FGK, A), with parameters shown in the table below.

Our model assumes the companion mass ratio distributions are independent of orbital separation.
""")

# Display the table of companion frequency and log-normal separation distribution
st.write("**Table 1: Companion Frequency (CF) & Log-Normal Separation Distribution vs. Host Type**")

table_data = {
    'Spectral Type': ['M', 'FGK', 'A'],
    'CF': ['0.236', '0.61', '0.219'],
    'μ (base-10)': ['1.43', '1.70', '2.72'],
    'σ (base-10)': ['1.21', '1.68', '0.79']
}

st.table(table_data)

st.write("""The data are fitted to 51 point estimates of companion frequency over specified mass ranges and orbital separations found in the literature. Please read the paper available at (archive) for details. As our model is fitted in the context of distributions of mass ratios of companions to host stars, you need to select the stellar mass of the host, as well as model parameters for the function form of the fits (best fits from our paper are defaults). Finally one must choose the mass range of companion of interest, as well as the orbital separation range of interest. The model is integrated to provide the expectation value of our model over the ranges indicated for both populations.  

If you are interested in using this model for a given target list and sensitivity curves to predict survey yields, please contact us to learn more.""")

# Add visual separation with a horizontal line
st.markdown("---")

# Add styled header for the tool section
st.markdown("<div class='tool-header'>Frequency Calculation Tool</div>", unsafe_allow_html=True)

##############################################################################
#Section 2 - Host Star Parameters
##############################################################################

st.markdown("<div class='section-header'>Host Star Parameters</div>", unsafe_allow_html=True)

# Default values for sliders from user-provided values
ln_A_bd_default = -1.407  # ln(A_bd)
ln_A_pl_default = -4.720  # ln(A_p)
alpha_bd_default = -0.292  # β
alpha_gp_default = 1.296  # α

# Radio button to select stellar type
st_type = st.radio(
    "Select a Stellar Spectral Type to update the brown dwarf log-normal distributions: M dwarfs follow Winters et al. (2019); FGK stars follow Raghavan et al. (2010); and A stars follow De Rosa et al. (2014), corrected for physical separation (x1.35 DM91). Updated values will be reflected in the sliders and graphs below. Note that you can fine-tune the slider values using the left/right arrow keys.",
    ("M Dwarfs", "FGK", "A Stars"),
    index=1  # Default to FGK (index 1)
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

# User input for mass parameters
host_mass = st.number_input(
    "Host Mass ($\mathrm{M_{\odot}}$)",
    min_value=0.0001,
    max_value=10.0,
    value=1.0,  # Default to 1 solar mass
    step=0.01,
    format="%.2f"
)

##############################################################################
#Section 3 - Frequency Calculation Tool
##############################################################################
# Create columns for a vertical arrangement of sliders
st.markdown("<div class='section-header'>Model Parameters</div>", unsafe_allow_html=True)
st.write(
    "Define the variables for the parametric model(s) of planets and brown dwarfs. "
    r"$\alpha$ represents the power-law index for the companion-to-star mass ratio ($q$), "
    r"$A$ is the normalization factor, and $\mu$ and $\sigma$ are the mean and standard deviation of the log-normal orbital distribution."
)
col1, col2 = st.columns(2)

# Conversion constant from natural log to log_10
ln10 = np.log(10)

# Brown Dwarf parameters in col1
with col1:
    alpha_bd = st.slider(r'$\mathrm{\beta}$', min_value=-2.0, max_value=2.0, value=alpha_bd_default, step=0.01)
    # Convert ln(A_bd) to A_bd for display, with appropriate range and step size
    A_bd = st.slider(r'$\mathrm{A_{bd}}$', min_value=0.0001, max_value=1.0, value=np.exp(ln_A_bd_default), step=0.0001, format="%.4f")
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
    alpha_gp = st.slider(r'$\mathrm{\alpha}$', min_value=-2.0, max_value=2.0, value=alpha_gp_default, step=0.01)
    # Convert ln(A_pl) to A_pl for display
    A_pl = st.slider(r'$\mathrm{A_{pl}}$', min_value=0.0001, max_value=0.1, value=np.exp(ln_A_pl_default), step=0.0001, format="%.4f")
    mu_pl = st.slider(
        r'$\mathrm{log_{10}(\mu_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=1.299/ln10,
        step=0.01
    )
    sigma_pl = st.slider(
        r'$\mathrm{log_{10}(\sigma_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=np.exp(0.215)/ln10,  # Default or static value
        step=0.01
    )

s_m =  np.log(10**sigma_bd)    # Winters
mu_m = np.log(10**mean_bd)

##############################################################################
#Section 4 - Companion Parameters
##############################################################################

st.markdown("<div class='section-header'>Companion Parameters</div>", unsafe_allow_html=True)

# Create two columns for input fields
col_mass1, col_mass2 = st.columns(2)

# Input fields for companion mass range
with col_mass1:
    Jup_min = st.number_input(
        "Minimum Companion Mass ($\mathrm{M_{Jup}}$)",
        min_value=0.03,
        value=1.0,
        step=0.1,
        format="%.4f"
    )

with col_mass2:
    Jup_max = st.number_input(
        "Maximum Companion Mass ($\mathrm{M_{Jup}}$)",
        value=100.0,
        step=0.1,
        format="%.2f"
    )

# Add validation
if Jup_min >= Jup_max:
    st.error("Minimum companion mass must be less than maximum companion mass.")
    Jup_min = min(Jup_min, Jup_max - 0.01)

# Mass ratio calculations
q_Jupiter = 0.001/host_mass

# Extended mass ratio range from 0.0001 to 1
full_q_range = np.logspace(-4, 0, 1000)  # From 0.0001 to 1

# Calculate power-law distributions with the same slopes across the full range and apply normalization factors
a2_bd_extended = (full_q_range ** -alpha_bd)  # Using same alpha_bd with A_bd normalization
a2_gp_extended = (full_q_range ** -alpha_gp)  # Using same alpha_gp with A_pl normalization

# Calculate the combined distribution (sum of both models)
combined_values = a2_bd_extended + a2_gp_extended

# Create plots for both distributions side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

# Plotting extended mass ratio distributions
ax1.plot(full_q_range, a2_bd_extended, color='r', linewidth=2, label='Brown Dwarf Model')
ax1.plot(full_q_range, a2_gp_extended, color='blue', linewidth=2, label='Giant Planet Model')
ax1.plot(full_q_range, combined_values, color='orange', linewidth=2, label='Sum of the two')

# Set both axes to log scale
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("Mass Ratio Distribution", fontsize=18)

# Create a custom formatter for the x-axis
def log_formatter(x, pos):
    return r"$10^{{{:.0f}}}$".format(x)  # Format as 10^n

# Apply the custom formatter for the x-axis
ax1.xaxis.set_major_formatter(FuncFormatter(log_formatter))

# Calculate mass ratio limits
mass_ratio_min = Jup_min * q_Jupiter
mass_ratio_max = Jup_max * q_Jupiter

# Add vertical lines for mass ratio limits
#ax1.axvline(x=mass_ratio_min, color='green', linestyle='--', label=f'Min Mass Ratio = {mass_ratio_min:.2f}')
#ax1.axvline(x=mass_ratio_max, color='purple', linestyle='--', label=f'Max Mass Ratio = {mass_ratio_max:.2f}')
#
#ax1.axvspan(mass_ratio_min, mass_ratio_max, color='gray', alpha=0.3)

# Configure mass ratio distribution plot
ax1.set_xlabel('Mass Ratio q', fontsize=20, labelpad=10.4)
ax1.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title("Mass Ratio Distribution", fontsize=18)

# User input for orbital separations using number inputs
col_sep1, col_sep2 = st.columns(2)

with col_sep1:
    a_min = st.number_input(
        "Minimum Orbital Separation (AU)",
        value=1.0,
        step=0.1,
        format="%.4f"
    )

with col_sep2:
    a_max = st.number_input(
        "Maximum Orbital Separation (AU)",
        value=100.0,
        step=0.1,
        format="%.2f"
    )

# Add validation
if a_min >= a_max:
    st.error("Minimum orbital separation must be less than maximum orbital separation.")
    a_min = min(a_min, a_max - 0.01)

# Define orbital separation distributions for Brown Dwarfs and Giant Planets
# These functions return the shape of the log-normal distribution in natural log space
# without the normalization factor, which is added separately in the integration
def orbital_dist_bd(a):
    # Convert log10 values from UI to natural log
    return np.exp(-(np.log(a) - mu_m) ** 2 / (2 * s_m**2))

def orbital_dist_pl(a):
    return np.exp(-(np.log(a) - np.log(10**mu_pl)) ** 2 / (2 * np.log(10**sigma_pl)**2))

# Create a log-scaled range for plotting
a_values = np.logspace(-2, 4, 500, base=10)

# Calculate orbital distribution values for plotting using the formula:
# φ(x) = A * e^((log(x)-μ)²/(2σ²)) / (2πσ)

# For brown dwarfs
bd_plot_values = []
for a in a_values:
    # Calculate log-normal PDF in log10 space
    bd_pdf = np.exp(-(np.log(a) - mu_m) ** 2 / (2 * s_m**2))/np.sqrt(2*np.pi)/s_m
    bd_plot_values.append(A_bd * bd_pdf)

# For giant planets
pl_plot_values = []
for a in a_values:
    # Calculate log-normal PDF in log10 space
    pl_pdf = np.exp(-(np.log(a) - mu_pl) ** 2 / (2 * sigma_pl**2))/np.sqrt(2*np.pi)/sigma_pl
    pl_plot_values.append(A_pl * pl_pdf)

ax2.plot(a_values, bd_plot_values, linewidth=2, label='Brown Dwarf Model', color='r')
ax2.plot(a_values, pl_plot_values, linewidth=2, label='Giant Planet Model', color='blue')

def log10_formatter(x, pos):
    return r"$10^{{{:.0f}}}$".format(x)

# Add vertical lines for min and max separations
# ax2.axvline(x=np.log10(a_min), color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
# ax2.axvline(x=np.log10(a_max), color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')
# ax2.axvspan(np.log10(a_min), np.log10(a_max), color='gray', alpha=0.3)

ax2.xaxis.set_major_formatter(FuncFormatter(log10_formatter))

# Set x-axis to log scale and appropriate y-axis limits
ax2.set_xscale('log')

# Automatically set y-axis limits based on the data
ax2.set_ylim(0, 0.07)

# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_title("Semi-Major Axis Distribution", fontsize=18)

# Display the plots
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
mass_ratio_values = np.linspace(Jup_min * q_Jupiter, Jup_max * q_Jupiter, 1000)

# Use logarithmic spacing for orbital separation to better sample the distribution
a_values_m = np.logspace(np.log10(max(a_min, 0.01)), np.log10(a_max), 1000)

# Calculate mass ratio distribution integrals
mass_ratio_integral_bd = np.trapz([d_q_i ** -alpha_bd for d_q_i in mass_ratio_values], mass_ratio_values)
mass_ratio_integral_gp = np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)

# For orbital distributions, we need to properly normalize by the PDF
orbital_values_bd = [orbital_dist_bd(a)/(np.sqrt(2 * np.pi)*s_m*a) for a in a_values_m]
orbital_values_pl = [orbital_dist_pl(a)/(np.sqrt(2 * np.pi)*np.log(10**sigma_pl)*a) for a in a_values_m]

# Calculate orbital distribution integrals
orbital_integral_bd = np.trapz(orbital_values_bd, a_values_m)
orbital_integral_pl = np.trapz(orbital_values_pl, a_values_m)

# Calculate frequency for Brown Dwarfs and Giant Planets
f_bd = A_bd * orbital_integral_bd * mass_ratio_integral_bd
f_pl = A_pl * orbital_integral_pl * mass_ratio_integral_gp

# Display results in Streamlit
st.write(f"Mean Number of Planets Per Star:", f_pl)
st.write(f"Mean Number of Brown Dwarfs Per Star:", f_bd)

st.write("")
st.write("*Note: These values represent the expected number of companions per star within the specified mass ratio and orbital separation ranges.*")
