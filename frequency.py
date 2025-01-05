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

##############################################################################
#Section 1 - Model Parameters
##############################################################################
# Display title
st.title("Composite Model for Exoplanet and Brown Dwarf Companions")
st.caption("Based on Meyer et al. (2025)")

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
mu_natural = 1.32    #ln
sigma_pl_ln = 0.53   #ln
mu_pl_value = mu_natural/constant     #log10
sigma_pl_value = sigma_pl_ln/constant #log10

if "mean_bd" not in st.session_state:
    st.session_state.mean_bd = 1.43
if "sigma_bd" not in st.session_state:
    st.session_state.sigma_bd = 1.21
if "stellar_type" not in st.session_state:
    st.session_state.stellar_type = "M Dwarfs"
if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False  # Initialize the flag for forcing refresh

# Brown Dwarf parameters in col1
with col1:
    alpha_gp = st.slider(r'$\mathrm{\alpha_{pl}}$', min_value=0.0, max_value=3.0, value=1.43, step=0.01)
    A_bd_ln = st.slider(r'$\mathrm{ln(A_{bd})}$', min_value=-10.0, max_value=0.0, value=-3.78, step=0.01)
    mean_bd = st.slider(
        r'$\mathrm{log_{10}(\mu_{bd})}$',
        min_value=0.0,
        max_value=3.0,
        value=st.session_state.mean_bd,
        step=0.01,
        key="mean_bd_slider"
    )
    sigma_bd = st.slider(
        r'$\mathrm{log_{10}(\sigma_{bd})}$',
        min_value=0.0,
        max_value=3.0,
        value=st.session_state.sigma_bd,
        step=0.01,
        key="sigma_bd_slider"
    )

# Giant Planet parameters in col2
with col2:
    alpha_bd = st.slider(r'$\mathrm{\alpha_{bd}}$', min_value=-2.0, max_value=2.0, value=-0.36, step=0.01)
    A_pl_ln = st.slider(r'$\mathrm{ln(A_{pl})}$', min_value=-10.0, max_value=0.0, value=-5.52, step=0.01)
    mu_pl = st.slider(
        r'$\mathrm{log_{10}(\mu_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=1.43,  # Default or static value
        step=0.01
    )
    sigma_pl = st.slider(
        r'$\mathrm{log_{10}(\sigma_{pl})}$',
        min_value=0.0,
        max_value=3.0,
        value=1.21,  # Default or static value
        step=0.01
    )

##############################################################################
#Section 2 - Host Star Parameters
##############################################################################

st.subheader("Host Star Parameters")

# Define function to update session state based on stellar type
def update_stellar_type():
    st.session_state.force_refresh = not st.session_state.force_refresh  # Toggle the trigger
    
# Add custom styling for the Update button
st.markdown(
    """
    <style>
    .update-button {
        background-color: #4CAF50;  /* Green background */
        color: white;  /* White text */
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .update-button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Radio button to select stellar type
st_type = st.radio(
    "Select a Stellar Spectral Type, then click the 'Update' button to update the brown dwarf log-normal distributions for the selected stellar type. Ensure you first select the stellar type, followed by clicking the 'Update' button, in that order. The updated values should be reflected in the slider bars above.",
    ("M Dwarfs", "FGK", "A Stars"),  # Without the "Update" button here
    index=("M Dwarfs", "FGK", "A Stars").index(st.session_state.stellar_type),
    on_change=update_stellar_type
)

# Custom "Update" button with different styling
update_button = st.button(
    "Update",
    key="update_button",
    help="Click to update the parameters based on the selected stellar type",
    use_container_width=True
)

# Check if the "Update" button was clicked
if update_button:
    update_stellar_type()  # Call the function to update the parameters

#the session updates are delayed, displays previous values
if st_type == "M Dwarfs":
    s_m =  np.log(10**1.21)    # Winters
    mu_m = np.log(10**1.43)
    st.session_state.mean_bd = 1.43
    st.session_state.sigma_bd = 1.21
elif st_type == "FGK":
    s_m = np.log(10**1.68)
    mu_m = np.log(50)
    st.session_state.mean_bd = 1.68
    st.session_state.sigma_bd = np.log10(50)
elif st_type == "A Stars":
    s_m = np.log(10**0.92)
    mu_m = np.log(522)            # De Rosa (x1.35 DM91)
    st.session_state.mean_bd = np.log10(522)
    st.session_state.sigma_bd = 0.92

# Update slider values based on the session state
# Store the current stellar type in session state
st.session_state.stellar_type = st_type

# Calculate updated values
A_bd = np.exp(A_bd_ln)
A_pl = np.exp(A_pl_ln)

# User input for mass parameters
host_mass = st.slider("Host Mass ($\mathrm{M_{\odot}}$)", min_value=0.001,max_value=5.0,value=0.3,step=0.001)

##############################################################################
#Section 3 - Companion Parameters
##############################################################################

st.subheader("Companion Parameters")
Jup_min, Jup_max = st.slider(
    "Companion Mass Range ($\mathrm{M_{Jup}}$)",
    min_value=0.01,
    max_value=300.0,
    value=(1.0, 100.0))

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
ax1.plot(d_q, a2_bd, color='r', linewidth = 2,label='Brown Dwarf Model')
ax1.plot(d_q, a2_gp, color='blue',linewidth = 2, label='Giant Planet Model')

ax1.set_xscale('log')
ax1.set_yscale('log')

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

##############################################################################
#Section 4 - Sub-Jupiter Model
##############################################################################

st.subheader("Sub-Jupiter Model (< 1 MJ)")
st.write("For planets with masses less than 1 Jupiter mass, we have developed a customized sub-Jupiter model. The orbital distribution within 10 AU follows the same pattern as the giant planet model (cf. Fulton 2021), but with a log-flat distribution and constant density beyond 10 AU.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

# Plotting mass ratio distributions
ax1.plot(d_q, a2_gp, color='C1', linewidth=2,label='Sub-Jupiter Model')
ax1.set_xscale('log')
ax1.set_yscale('log')

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

# Define orbital separation distributions for sub-Jupiters
def orbital_dist_subJupiter(a):
    if a <= 10:
        return orbital_dist_pl(a)
    else:
        return 0.19

# Create a log-scaled range for plotting
a_values = np.logspace(-3, 5, 500, base=10)  # Values from 0.01 to 1000, log-scaled
sub_jupiter_values = [orbital_dist_subJupiter(a) for a in a_values]
# Plotting the distributions
ax2.plot(np.log10(a_values),sub_jupiter_values, label='Sub-Jupiter Model', color='C1')

# Add vertical lines for min and max separations
ax2.axvline(x=np.log10(a_min), color='green', linestyle='--', label=f'Min Separation = {a_min:.2f} AU')
ax2.axvline(x=np.log10(a_max), color='purple', linestyle='--', label=f'Max Separation = {a_max:.2f} AU')
ax2.axvspan(np.log10(a_min), np.log10(a_max), color='gray', alpha=0.3)

# Configure semi-major axis distribution plot
ax2.set_xlabel('Semi-Major Axis (ln AU)', fontsize=20, labelpad=10.4)
ax2.set_ylabel('Probability Density', fontsize=20, labelpad=10.4)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_title("Semi-Major Axis Distribution", fontsize=18)
ax2.set_ylim(0,1.5)
ax2.xaxis.set_major_formatter(FuncFormatter(log10_formatter))

# Display the plots
st.pyplot(fig)

# Define the range for orbital distances and mass ratios using correct log_10 ranges
mass_ratio_values = np.linspace(Jup_min * q_Jupiter, Jup_max * q_Jupiter, 500)
if a_min<=10 and a_max <=10:
    a_values_m = np.linspace(a_min,a_max, 500)
    f_subJ =  A_pl*np.trapz([orbital_dist_pl(a)/(np.sqrt(2*np.pi)*sigma_pl_ln*a) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
elif a_min<=10 and a_max>10:
    a_values_m1 = np.linspace(a_min,10, 500)
    f_subJ1 = A_pl* np.trapz([orbital_dist_pl(a)/(np.sqrt(2*np.pi)*sigma_pl_ln*a) for a in a_values_m], a_values_m) * np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
    a_values_m2 = np.linspace(10,a_max, 500)
    f_subJ2 = A_pl* np.trapz([0.19/(a*np.log(a_max/10)) for a in a_values_m2], a_values_m2) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
    f_subJ = f_subJ1 + f_subJ2
elif a_min>10 and a_max >10:
    a_values_m = np.linspace(a_min,a_max, 500)
    f_subJ = A_pl* np.trapz([0.19/(a*np.log(a_max/10)) for a in a_values_m], a_values_m) * \
       np.trapz([d_q_i ** -alpha_gp for d_q_i in mass_ratio_values], mass_ratio_values)
       
# Display results in Streamlit
st.write(f"Mean Number of Companions Per Star:", f_subJ)
