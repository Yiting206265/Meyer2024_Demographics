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
st.latex(r"\phi_{pl}(x) = \frac{A_{pl} e^{-(x - \mu_{pl})^2/2 \sigma_{pl}^2}}{x \sqrt{2\pi}\sigma_{pl}}")

# Add the formula for brown dwarfs with explicit log10 notation
st.latex(r"\phi_{bd}(x) = \frac{A_{bd} e^{-(x - \mu_{bd})^2/2 \sigma_{bd}^2}}{x \sqrt{2\pi}\sigma_{bd}}")

st.write(r"""
where $x = \log_{10}(a)$ is the logarithm (base-10) of the semi-major axis in AU, and both $\mu_{pl}$ and $\mu_{bd}$ are in units of $\log_{10}(a)$.

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
alpha_bd_default = -0.292 # β
alpha_gp_default = 1.296  # α
mu_pl_default = 1.299     #ln(mu_p)
sigma_pl_default = 0.215  #ln(sigma_p)

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
        value=mu_pl_default/ln10,
        step=0.01
    )
    sigma_pl = st.slider(
        r'$\mathrm{log_{10}(\sigma_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=sigma_pl_default/ln10,  # Default or static value
        step=0.01
    )

#All the variables are given in log10, so I am going to convert everything into natural log for calculation

# Convert log10 parameters to natural log for calculations
sigma_bd_ln = sigma_bd * ln10
mean_bd_ln = mean_bd * ln10
mu_pl_ln = mu_pl*ln10
sigma_pl_ln = sigma_pl*ln10

def orbital_dist_bd(a):
    # Log-normal distribution for orbital separation
    mu_a_bd_ln = mean_bd_ln
    return np.exp(-(np.log(a) - mu_a_bd_ln)**2 / (2 * sigma_bd_ln**2))

def orbital_dist_pl(a):
    # Log-normal distribution for orbital separation
    mu_a_gp_ln = mu_pl_ln
    return np.exp(-(np.log(a) - mu_a_gp_ln)**2 / (2 * sigma_pl_ln**2))

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
        min_value=0.03,
        max_value=85.0,
        value=85.0,
        step=0.1,
        format="%.2f"
    )

# Add validation
if Jup_min >= Jup_max:
    st.error("Error: Minimum companion mass must be less than the maximum companion mass.")
    st.stop()

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

# Create a fresh figure for the plot
plt.close('all')  # Close any existing figures
fig, ax = plt.subplots(figsize=(10, 8))

# Create mass ratio values for plotting with fixed range from 10^-3 to 1
# Calculate mass ratio range based on companion mass inputs
min_q = max(Jup_min * q_Jupiter, 1e-3)  # Ensure minimum is at least 10^-3
max_q = min(Jup_max * q_Jupiter, 1.0)    # Ensure maximum is at most 1.0

# Define the range for orbital distances and mass ratios using correct log_10 ranges
# Use logarithmic spacing for mass ratio to better sample the distribution
mass_ratio_values = np.logspace(
    np.log10(max(Jup_min * q_Jupiter, 1e-3)),  # Ensure minimum is at least 10^-3
    np.log10(min(Jup_max * q_Jupiter, 1.0)),   # Ensure maximum is at most 1.0
    1000
)



# Define functions for mass ratio and orbital separation distributions
def mass_fctn_bd(q):
    # Brown dwarf slope should be rising (positive exponent), so flip sign of alpha_bd
    return q ** (-1 * alpha_bd)

def mass_fctn_pl(q):
    # Planet slope is negative
    return q ** (-1 * alpha_gp)

def surface_den_bd(a):
    return orbital_dist_bd(a) / (np.sqrt(2*np.pi) * sigma_bd_ln * a)

def surface_den_pl(a):
    return orbital_dist_pl(a) / (np.sqrt(2*np.pi) * sigma_pl_ln * a)

# Function to calculate dN/dlogq by integrating over separation range
def dN_bd(q):
    # Integrate the orbital distribution over the separation range and multiply by mass function
    # Use fixed integration limits for consistency
    orbital_integral = integrate.quad(surface_den_bd, max(a_min, 0.01), a_max)[0]
    return A_bd * mass_fctn_bd(q) * orbital_integral

def dN_pl(q):
    # Integrate the orbital distribution over the separation range and multiply by mass function
    # Use fixed integration limits for consistency
    orbital_integral = integrate.quad(surface_den_pl, max(a_min, 0.01), a_max)[0]
    return A_pl * mass_fctn_pl(q) * orbital_integral

# Pre-calculate orbital integrals (these are constant for all q values)
bd_orbital_integral = integrate.quad(surface_den_bd, max(a_min, 0.01), a_max)[0]
pl_orbital_integral = integrate.quad(surface_den_pl, max(a_min, 0.01), a_max)[0]

# Define specific mass ratio ranges for plotting caps from sliders
# Planets (red curve) are at lower mass, Brown Dwarfs (blue curve) are at higher mass
q_pl_min = Jup_min * q_Jupiter
q_pl_max = 0.1  # This remains fixed as per user's model definition
q_bd_min = 3 * q_Jupiter # This remains fixed as per user's model definition
q_bd_max = Jup_max * q_Jupiter

# Define specific mass ratio ranges for each population
q_pl_min = 3 * q_Jupiter
q_pl_max = 85 * q_Jupiter
q_bd_min = 0.03 * q_Jupiter
q_bd_max = 0.1

bd_freq = []
pl_freq = []
total_freq = []

for q in mass_ratio_values:
    # Get the direct mass function values
    bd_mass = mass_fctn_bd(q)
    pl_mass = mass_fctn_pl(q)

    # Calculate dN/dlog(q) using the direct mass function, applying caps
    # dN/dlog(q) = A * mass_fctn(q) * orbital_integral * q * ln(10)
    bd_val = A_bd * bd_mass * bd_orbital_integral * q * np.log(10) if q_bd_min <= q <= q_bd_max else 0
    pl_val = A_pl * pl_mass * pl_orbital_integral * q * np.log(10) if q_pl_min <= q <= q_pl_max else 0
    
    # Append to arrays
    bd_freq.append(bd_val)
    pl_freq.append(pl_val)
    total_freq.append(bd_val + pl_val)

# Convert lists to numpy arrays for plotting
bd_freq_array = np.array(bd_freq)
pl_freq_array = np.array(pl_freq)
total_freq_array = np.array(total_freq)

# Replace zeros with NaN for cleaner plotting (to avoid vertical lines)
bd_freq_array[bd_freq_array == 0] = np.nan
pl_freq_array[pl_freq_array == 0] = np.nan
total_freq_array[mass_ratio_values < q_pl_min] = np.nan
# Cap total curve to start and end with the planet model's domain
total_freq_array[mass_ratio_values > q_pl_max] = np.nan

# Plot the frequency distribution dN/dlogq vs log(q)
log_mass_ratio_values = np.log10(mass_ratio_values)
ax.plot(log_mass_ratio_values, pl_freq_array, color='r', linewidth=2, label='Giant Planet Model')
ax.plot(log_mass_ratio_values, bd_freq_array, color='blue', linewidth=2, label='Brown Dwarf Model')
ax.plot(log_mass_ratio_values, total_freq_array, color='orange', linewidth=2, label='Total Frequency')

# Configure plot
ax.set_xlabel('log(q)', fontsize=20, labelpad=10.4)
ax.set_ylabel('dN / dlog(q)', fontsize=20, labelpad=10.4)
ax.set_xlim(-4, -0.5)
# Dynamically adjust y-axis limit
try:
    # Find the maximum value of the visible total frequency curve
    max_y = np.nanmax(total_freq_array)
    # Set the y-axis limit to be slightly above the max value for better visualization
    ax.set_ylim(0, max_y * 1.1)
except ValueError:
    # Default limit if the array is all NaN
    ax.set_ylim(0, 0.3)
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



# Calculate the mean number of companions by integrating the final frequency distributions
# We integrate dN/dlog(q) over dlog(q) to get the total number N.

# Create copies of the frequency arrays to avoid modifying the data used for plotting
pl_freq_for_integration = np.copy(pl_freq_array)
bd_freq_for_integration = np.copy(bd_freq_array)

# Replace NaNs with 0 for the integration
pl_freq_for_integration[np.isnan(pl_freq_for_integration)] = 0
bd_freq_for_integration[np.isnan(bd_freq_for_integration)] = 0

# Integrate dN/dlog(q) over log(q) using the trapezoidal rule
mean_num_pl = np.trapz(pl_freq_for_integration, log_mass_ratio_values)
mean_num_bd = np.trapz(bd_freq_for_integration, log_mass_ratio_values)

# Display results in Streamlit
st.write(f"Mean Number of Planets Per Star: mean_num_pl")
st.write(f"Mean Number of Brown Dwarfs Per Star: mean_num_bd")

st.write("")
st.write("*Note: These values represent the expected number of companions per star within the specified mass ratio and orbital separation ranges.*")
