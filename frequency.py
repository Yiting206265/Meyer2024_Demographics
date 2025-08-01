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
st.latex(r"N_{TOTAL} = \int{\phi_{pl}(x) \times \psi_{pl}(q) dq dx} + \int{\phi_{bd}(x) \times \psi_{bd}(q) dq dx}")

st.write(r"""
where $\phi_{pl}(x)$ and $\phi_{bd}(x)$ are log-normal orbital separation distributions for planets and brown dwarfs respectively, and $\psi_{pl}(q)$ and $\psi_{bd}(q)$ are power-law mass ratio distributions.

For the mass ratio distributions (power-law):
""")

# Add the power-law formula for mass ratio distributions
st.latex(r"\psi_{pl}(q) = q^{-\alpha}")
st.latex(r"\psi_{bd}(q) = q^{-\beta}")

st.write(r"""
where $\alpha$ and $\beta$ are the power-law indices for planets and brown dwarfs respectively.

For the orbital separation distributions (log-normal):
""")

# Use st.latex for proper rendering of equations - with explicit log10 notation for planets
st.latex(r"\phi_{pl}(x) = \frac{A_{pl} e^{-(x - \log_{10}(\mu_{pl}))^2/2 (\log_{10}(\sigma_{pl}))^2}}{\sqrt{2\pi}\log_{10}(\sigma_{pl})}")

# Add the formula for brown dwarfs with explicit log10 notation
st.latex(r"\phi_{bd}(x) = \frac{A_{bd} e^{-(x - \log_{10}(\mu_{bd}))^2/2 (\log_{10}(\sigma_{bd}))^2}}{\sqrt{2\pi}\log_{10}(\sigma_{bd})}")

st.write(r"""
where $x = \log_{10}(a)$ is the logarithm (base-10) of the semi-major axis in AU.

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
        value=0.215/ln10,  # Default or static value
        step=0.01
    )

# Convert log10 parameters to natural log for calculations
# These are the correct conversions from log10 to natural log
sigma_bd_ln = sigma_bd * ln10  # Convert sigma from log10 to natural log
mean_bd_ln = mean_bd * ln10    # Convert mean from log10 to natural log

# Define orbital separation distributions for Brown Dwarfs and Giant Planets
# These functions return the shape of the log-normal distribution in natural log space
# without the normalization factor, which is added separately in the integration
def orbital_dist_bd(a):
    # This function returns the unnormalized log-normal distribution in natural log space
    # mean_bd_ln and sigma_bd_ln are already converted from log10 to natural log
    return np.exp(-(np.log(a) - mean_bd_ln) ** 2 / (2 * sigma_bd_ln**2))

def orbital_dist_pl(a):
    # Convert log10 values to natural log correctly
    # sigma_pl is in log10 units, so we need to multiply by ln(10) to convert to natural log
    # mu_pl is already in log10 space, so we convert to natural log space with ln(10)
    # This returns just the exponential part without normalization
    return np.exp(-(np.log(a) - mu_pl*ln10) ** 2 / (2 * (sigma_pl*ln10)**2))

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


#x axis should be a function of q


# Calculate the combined distribution (sum of both models)
combined_values = a2_bd_extended + a2_gp_extended

##############################################################################
#Section 5 - Orbital Separation Range
##############################################################################

# User input for orbital separations using number inputs
col_sep1, col_sep2 = st.columns(2)

with col_sep1:
    a_min = st.number_input(
        "Minimum Orbital Separation (AU)",
        value=1.0,
        step=0.1,
        format="%.2f"
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

# Create a single plot for the frequency distribution vs mass ratio
fig, ax = plt.subplots(figsize=(12, 8))

# Create mass ratio values for plotting with fixed range from 10^-3 to 1
# Calculate mass ratio range based on companion mass inputs
min_q = max(Jup_min * q_Jupiter, 1e-3)  # Ensure minimum is at least 10^-3
max_q = min(Jup_max * q_Jupiter, 1.0)    # Ensure maximum is at most 1.0

# Create logarithmically spaced mass ratio values within the calculated range
q_values = np.logspace(np.log10(min_q), np.log10(max_q), 100)

# Calculate the joint distribution df/(dq dloga) for each mass ratio
bd_joint_freq = []
pl_joint_freq = []
total_joint_freq = []

# For each mass ratio, calculate the joint distribution at a representative separation
# We'll use the mean of the log separation range as a representative value
log_a_mean = (np.log10(max(a_min, 0.01)) + np.log10(a_max)) / 2
a_mean = 10**log_a_mean

for q in q_values:
    # Brown dwarf joint distribution at this mass ratio
    # Calculate log-normal PDF in natural log space for orbital distribution
    # The normalization factor for a log-normal distribution in natural log space is 1/(sqrt(2π)×σ×a)
    bd_pdf_a = orbital_dist_bd(a_mean) / (np.sqrt(2*np.pi) * sigma_bd_ln * a_mean)
    
    # Calculate power-law for mass ratio
    bd_pdf_q = q ** -alpha_bd
    
    # Combine to get joint distribution df/(dq dloga)
    bd_joint = A_bd * bd_pdf_a * bd_pdf_q
    bd_joint_freq.append(bd_joint)
    
    # Giant planet joint distribution at this mass ratio
    # Calculate log-normal PDF in natural log space
    # The normalization factor for a log-normal distribution in natural log space is 1/(sqrt(2π)×σ×a)
    sigma_pl_ln = sigma_pl * ln10  # Convert sigma from log10 to natural log
    mu_pl_ln = mu_pl * ln10       # Convert mu from log10 to natural log
    pl_pdf_a = np.exp(-(np.log(a_mean) - mu_pl_ln)**2 / (2 * sigma_pl_ln**2)) / (np.sqrt(2*np.pi) * sigma_pl_ln * a_mean)
    
    # Calculate power-law for mass ratio
    pl_pdf_q = q ** -alpha_gp
    
    # Combine to get joint distribution df/(dq dloga)
    pl_joint = A_pl * pl_pdf_a * pl_pdf_q
    pl_joint_freq.append(pl_joint)
    
    # Total joint frequency
    total_joint_freq.append(bd_joint + pl_joint)

# Plot the joint frequency distribution df/(dq dloga) vs mass ratio
ax.plot(q_values, bd_joint_freq, color='r', linewidth=2, label='Brown Dwarf Model')
ax.plot(q_values, pl_joint_freq, color='blue', linewidth=2, label='Giant Planet Model')
ax.plot(q_values, total_joint_freq, color='orange', linewidth=2, label='Total Frequency')

# Set axes to log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Configure plot
ax.set_xlabel('Mass Ratio q', fontsize=20, labelpad=10.4)
ax.set_ylabel('df/(dq dloga)', fontsize=20, labelpad=10.4)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(loc='upper right', fontsize=14)
ax.set_title("Companion Frequency Distribution", fontsize=22)

# Calculate mass ratio limits
mass_ratio_min = Jup_min * q_Jupiter
mass_ratio_max = Jup_max * q_Jupiter

# Add vertical lines for mass ratio limits
# ax.axvline(x=mass_ratio_min, color='green', linestyle='--', label=f'Min Mass Ratio')
# ax.axvline(x=mass_ratio_max, color='purple', linestyle='--', label=f'Max Mass Ratio')
# ax.axvspan(mass_ratio_min, mass_ratio_max, color='gray', alpha=0.1)


# Display the plot
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
# Use logarithmic spacing for mass ratio to better sample the distribution
mass_ratio_values = np.logspace(
    np.log10(max(Jup_min * q_Jupiter, 1e-3)),  # Ensure minimum is at least 10^-3
    np.log10(min(Jup_max * q_Jupiter, 1.0)),   # Ensure maximum is at most 1.0
    1000
)

# Use logarithmic spacing for orbital separation to better sample the distribution
a_values_m = np.logspace(np.log10(max(a_min, 0.01)), np.log10(a_max), 1000)

# Calculate mass ratio distribution integrals
mass_ratio_integral_bd = np.trapz([d_q_i ** -alpha_bd for d_q_i in mass_ratio_values], mass_ratio_values)
mass_ratio_integral_gp = np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)

# For orbital distributions, we need to properly normalize by the PDF
# The normalization factor for a log-normal distribution in natural log space is 1/(sqrt(2π)×σ×a)
orbital_values_bd = [orbital_dist_bd(a)/(np.sqrt(2 * np.pi)*sigma_bd_ln*a) for a in a_values_m]
orbital_values_pl = [orbital_dist_pl(a)/(np.sqrt(2 * np.pi)*(sigma_pl*ln10)*a) for a in a_values_m]

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
