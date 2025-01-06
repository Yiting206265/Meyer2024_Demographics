## Import modules
import numpy as np
import random
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import interpolate, stats, integrate
from astropy import constants as con
import pandas as pd
import glob
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
import pickle
import time
import sys
import os
import json
import scipy
import streamlit as st
from functions_companion_predictions_JWST import *

# Ensure compatibility with np.trapz in case scipy.integrate.simps is unavailable
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = np.trapz

# Set the title of the app
st.title("Yield Prediction")

####################################################################################
# Section 1 - Contrast curves
####################################################################################

st.header("Contrast Curves")
# File uploader for JSON file
uploaded_file = st.file_uploader("Upload your JSON file. It should contain a big dictionary where the first-level keys are target names, the second-level keys are filter bands, and the third-level keys are pairs of 5-sigma contrast (named '5sig_maskmag') and separation values (named 'seps_arcsec'). An example can be found below:", type="json", key='up1')

# Load the JSON data
if uploaded_file is None:
    f = open('./files/calcons_06052024.json')
else:
    f = uploaded_file

def get_interpolated_mass(input_luminosity, input_age):
    # Function to get interpolated mass based on given luminosity and age

    # Extract relevant data from the existing model
    # Example model data (replace this with your actual model data)
    mod_ages = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
                0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090,0.100,
                0.120,0.150,0.200,0.300,0.400,0.500,0.600,0.700,0.800,0.900,
                1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0]

    model_age = find_nearest(mod_ages, input_age)
    modat = np.genfromtxt('./BT_settl_models.dat', usecols=[0,1,2,3,8,9,10,11,12,13])
    mok = np.where(modat[:,0] == model_age)[0]
    model_mass = modat[mok,1]
    model_luminosity = modat[mok,3]

    # Perform linear interpolation/extrapolation based on luminosity and age

    # Check if the given input luminosity is within the range of model luminosities
    if input_luminosity < min(model_luminosity) or input_luminosity > max(model_luminosity):
        # Perform extrapolation if input luminosity is outside the model range
        # Here, a simple linear extrapolation is used (replace with your preferred method)
        interpolated_mass = np.interp(input_luminosity, model_luminosity, model_mass)
    else:
        # Perform interpolation within the model data range
        interpolated_mass = np.interp(input_luminosity, model_luminosity, model_mass)

    return interpolated_mass
    
try:
    calcons_json = json.load(f)

    # Display the data in a pandas DataFrame
    df = pd.DataFrame(calcons_json).T  # Transpose to get the correct orientation
    st.write("Data Preview:")
    st.dataframe(df)

    # List of names to plot
    names = df.index.tolist()  # Get the names from the DataFrame index

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(8, 5))

    # Initialize lists for storing separation and sensitivity
    sep = []
    mag_sens = []

    # Plot each object's sensitivity curve
    for name in names:
        seps_arcsec = calcons_json[name]['F444W']['seps_arcsec']
        sensitivity_mag = calcons_json[name]['F444W']['5sig_maskmag']
        
        ax.plot(seps_arcsec, sensitivity_mag, label=name)
        sep.append(seps_arcsec)
        mag_sens.append(sensitivity_mag)

    # Customize the plot
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Separation (arcsec)")
    ax.set_ylabel("Magnitude F444W")

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Convert lists to numpy arrays for further processing
    curves = np.array([sep, mag_sens])

    ### Get contrast curves ###
    contr_sep_arr = curves[0]        # Separation in arcsec
    contr_mag_arr = curves[1]        # contrast in mag
    
except json.JSONDecodeError:
    st.error("Error loading JSON file. Please ensure it is correctly formatted.")
except Exception as e:
    st.error(f"An error occurred: {e}")

####################################################################################
# Section 2 - Target file
####################################################################################

# File uploader for the input object file
st.header("Target Information")
uploaded_file = st.file_uploader("Upload your target stars file with the following information: stellar age, distance, stellar mass, and magnitude in a specific filter band.", type=["txt", "csv"], key='up2')

if uploaded_file is None:
    file = open('./files/mdwarfs_w1w2.txt')
else:
    file = uploaded_file
try:
    # Load the data into a DataFrame, assuming space-separated values
    df = pd.read_csv(file, delim_whitespace=True,header=0)
    # Display the DataFrame in Streamlit
    st.write("Data Preview:")
    st.dataframe(df)
except Exception as e:
    st.error(f"An error occurred while reading the file: {e}")

try:
    objnam = df['name']
    age = df['age']
    dist = df['distance']
    star_mass = df['mass']
    inmag = df['Magnitude']          # <-- Relevant for getting magnitude band correct!
    # Converting age from Myrs as in input file to Gyrs as in model grid
    age /= 1e3
    # number of stars in survey
    n_star = len(objnam)
except:
    print('File not uploaded')

####################################################################################
# Section 3 - Generation of Companions
####################################################################################

st.header("Generation of Companions")
st.write("Below we will create random sets of orbital parameters using Poisson statistics and the following priors: uniform priors for the longitude of the ascending node and the longitude of periastron, cosine priors for inclination, and Gaussian priors for eccentricity N~(0, 0.3) (Hogg2010). We assume the Meyer 2025 et al. model. For each set of orbital parameters, we generate evenly spaced in time and assess detectability across an even grid of companion masses and semimajor axes. Each survey is a realization. The same set of orbital parameters is applied to every target. For low-mass planets, we utilize the BEX evolutionary models (Linder2019), while for higher-mass companions, we use the ATMO 2020 models (Phillips2020).")

# Radio button to select stellar type
st.write( "Select the stellar spectral type and configure the model parameters. Default values are based on the Meyer 2025 model.")
st_type = st.radio("Spectral Type:",("M Dwarfs", "FGK", "A Stars"))

#values in log base-10
if st_type == "M Dwarfs":
    mean_bd = 1.43
    sigma_bd = 1.21 # Winters
elif st_type == "FGK":
    mean_bd = 1.68
    sigma_bd = np.log10(50) #Raghavan
elif st_type == "A Stars":
    mean_bd = np.log10(522)# De Rosa (x1.35 DM91)
    sigma_bd = 0.92
mu_m = np.log(10**mean_bd)
s_m =  np.log(10**sigma_bd)

# Parameters stored in a class
p = input_pams()
p.q_flag_bd = 1    # 1 == Yes for q-ratio, otherwise m will be in solar masses.
p.q_flag_pl = 1

# Model parameters
col1, col2 = st.columns(2)

# Conversion constant from natural log to log_10
constant = 2.302585092994046
mu_natural = 1.32    #ln
sigma_pl_ln = 0.53   #ln
mu_pl_value = mu_natural/constant     #log10
sigma_pl_value = sigma_pl_ln/constant #log10

# Brown Dwarf parameters in col1
with col1:
    alpha_bd = st.slider(r'$\mathrm{\alpha_{bd}}$', min_value=-2.0, max_value=2.0, value=-0.36, step=0.01)
    A_bd_ln = st.slider(r'$\mathrm{ln(A_{bd})}$', min_value=-10.0, max_value=0.0, value=-3.78, step=0.01)
    mean_bd = st.slider(
        r'$\mathrm{log_{10}(\mu_{bd})}$',
        min_value=0.0,
        max_value=3.0,
        value=mean_bd,
        step=0.01,
        key="mean_bd_slider"
    )
    sigma_bd = st.slider(
        r'$\mathrm{log_{10}(\sigma_{bd})}$',
        min_value=0.0,
        max_value=3.0,
        value=sigma_bd,
        step=0.01,
        key="sigma_bd_slider"
    )

# Giant Planet parameters in col2
with col2:
    alpha_gp = st.slider(r'$\mathrm{\alpha_{pl}}$', min_value=0.0, max_value=3.0, value=1.43, step=0.01)
    A_pl_ln = st.slider(r'$\mathrm{ln(A_{pl})}$', min_value=-10.0, max_value=0.0, value=-5.52, step=0.01)
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


#### PLANET PARAMETERS ###
p.alpha = 1.43
p.A_pl = np.exp(-5.52)  #1 Normalization Variable
mu_natural = 1.32   #ln
sigma_pl_ln = 0.53  #ln
p.median_loga = mu_natural
p.sigma = sigma_pl_ln
#### BD PARAMETERS ###
p.A_bd = np.exp(-3.78)
p.alpha_bd = -0.36 #was 0.25, depends on
p.median_loga_bd = mean_bd   ## Extrapolated from Solar to BD
p.sigma_bd = sigma_bd #SPHERE SHINE survey


st.write("Enter user input parameters, then click run for the simulations. Specify the normalization frequency within a certain separation-mass space. Then specify the limits for which the random companions will be generated.")

# user input
# Planets
# Number of real planets
p.n_real = st.slider("Number of Surveys/Realizations", min_value=1, max_value=10000, value=100)

# Normalization bounds for planets
st.subheader("Normalization Limits for Planets")
# Planet frequency
p.P_pl = st.slider("Planet Frequency", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
p.an_min_pl, p.an_max_pl = st.slider(
    "Planet Separation Range (AU)",
    min_value=0.1,max_value=1000.0,value=(0.01, 100.0))
# Minimum and maximum planet mass (in solar masses)
p.mn_min_pl, p.mn_max_pl = st.slider(
    "Planet Mass Range ($\mathrm{M_{Jup}}$)",
    min_value=0.1,max_value=200.0,value=(1.0, 75.0))

p.mn_min_pl = p.mn_min_pl*0.0009545942
p.mn_max_pl = p.mn_max_pl*0.0009545942

# Planet limits
st.subheader("Simulation Limits for Planets")
p.a_min_pl, p.a_max_pl = st.slider(
    "Simulation Semi-Major Axis Range (AU)",
    min_value=0.1,max_value=1000.0,value=(0.1, 300.0))
p.m_min_pl, p.m_max_pl = st.slider(
    "Simulation Planet Mass Range ($M_{Jup}$)",
    min_value=0.1,max_value=200.0,value=(0.1, 75.0))

p.m_min_pl = p.m_min_pl*0.0009545942
p.m_max_pl = p.m_max_pl*0.0009545942

# Brown Dwarfs
# BD limits
st.subheader("Normalization Limits for Brown Dwarfs")
p.P_bd = st.slider(
    "Brown Dwarf Frequency",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01)

p.an_min_bd, p.an_max_bd = st.slider(
    "BD Separation Range (AU)",
    min_value=0.1,
    max_value=1000.0,
    value=(0.0, 100.0)
)

p.mn_min_bd, p.mn_max_bd = st.slider(
    "BD Mass Range ($M_{Jup}$)",
    min_value=0.1,
    max_value=200.0,
    value=(3.0, 100.0)
)

p.mn_min_bd *=0.0009545942
p.mn_max_bd *=0.0009545942

# BD mass limits
st.subheader("Simulation Limits for Brown Dwarfs")
# Minimum and maximum brown dwarf mass (in solar masses)
p.m_min_bd, p.m_max_bd = st.slider(
    "Simulation BD Mass Range ($\mathrm{M_{Jup}}$)",
    min_value=0.1,
    max_value=200.0,
    value=(3.0, 100.0)
)

p.m_min_bd*=0.0009545942
p.m_max_bd*=0.0009545942

# BD semi-major axis limits
p.a_min_bd, p.a_max_bd = st.slider(
    "Simulation BD Semi-Major Axis Range (AU)",
    min_value=0.1,
    max_value=200.0,
    value=(0.0, 108.0)
)

#Evolutionary Model
st.subheader("Model Assumptions")
pl_type = st.radio("Planet Model:", ("Super-Jupiter (> 1 MJ)", "Sub-Jupiter (< 1 MJ)"))

if pl_type == "Super-Jupiter (> 1 MJ)":
    p.planet_sma = "lognormal"
elif pl_type == "Sub-Jupiter (< 1 MJ)":
    p.planet_sma = "flat"

# Generating masses, SMAs, eccentricities and inclinations
d = generate_distributions(p)

#############################################
# Define distribution Models
######################################################

# If distribution assumes the sub-Jupiter distribution:
#if p.planet_sma == 'flat':
#    d.adis = 1 / (d.a * np.log10(p.a_max_pl / p.a_min_pl))  # Restructure sma distribution after normalizing

constant = 0.19
def orbital_dist_subJupiter(a):
    if a <= 10:
        return (np.exp(-(np.log(a) - p.median_loga) ** 2/(2* p.sigma ** 2)))#/(np.sqrt(2*np.pi)*sigma_pl_ln*a)
    else:
        return constant#
        
if p.planet_sma == 'flat':
    a_min = p.a_min_pl
    a_max = p.a_max_pl

    if a_min<=10 and a_max <=10:
        a_values_m = np.linspace(a_min, a_max, 1000)#/(np.sqrt(2*np.pi)*2*p.sigma*a)
        adis =  [orbital_dist_subJupiter(a)/(np.sqrt(2*np.pi)*2*p.sigma*a) for a in a_values_m]
    elif a_min<=10 and a_max>10:
        a_values_m1 = np.linspace(a_min,10, 500)#/(np.sqrt(2*np.pi)*2*p.sigma*a)
        f_subJ1 =  [orbital_dist_subJupiter(a)/(np.sqrt(2*np.pi)*2*p.sigma*a) for a in a_values_m1]
        a_values_m2 = np.linspace(10,a_max, 500)
        f_subJ2 = [constant/(a*np.log(a_max/10)) for a in a_values_m2]
        adis = list(f_subJ1) + list(f_subJ2)
    elif a_min>10 and a_max >10:
        a_values_m = np.linspace(a_min,a_max, 1000)
        adis = [constant/(a*np.log(a_max/a_min)) for a in a_values_m]
        
    adis_flat = np.array(adis)

## Generating and detecing planet and BD properties ##
######################################################
# This is the important part
######################################################

# Defining all vectors for detection statistics
n_real = p.n_real
nn = np.zeros(n_real)

w_pl = np.zeros(n_real)
z_pl = np.zeros(n_real)
pl_detection = np.zeros(n_real)
pl_detection_prob = np.zeros(n_star)
pl_generated_prob = np.zeros(n_star)
combined_pl_generated = np.zeros(n_real)
combined_pl_detected = np.zeros(n_real)
pl_detected_fraction = np.zeros(n_star)

w_bd = np.zeros(n_real)
z_bd = np.zeros(n_real)
bd_detection = np.zeros(n_real)
bd_detection_prob = np.zeros(n_star)
bd_generated_prob = np.zeros(n_star)
combined_bd_generated = np.zeros(n_real)
combined_bd_detected = np.zeros(n_real)
bd_detected_fraction = np.zeros(n_star)

##START calculation
p.model = p.model_bd = st.radio("Evolutionary Model", ("BEX"))
p.band = p.band_bd = st.radio("Filter Band", ("F356W","F444W","F1000W","F1500W","F2100W"))
n_star = len(objnam)

# Assuming n_star, objnam, output_path, p, d, star_mass, dist, age, contr_sep_arr, contr_mag_arr, etc. are defined
# Loop over stars
if st.button('Run'):
    loop_timer = timeit.default_timer()

    # Open output file
    p.ninst = 1
    Fout = open('PL_out.dat', 'w')  # Open file
    Fout1 = open('BD_out.dat', 'w')  # Open file
    
    Fout.write("# mag_pl    log_mass    log_q    log_a    d_au    sep_as    incl    ecc    flag    ii    jj\n")
    Fout1.write("# mag_pl    log_mass    log_q    log_a    d_au    sep_as    incl    ecc    flag    ii    jj\n")

    for ii in range(0, n_star):
        # Estimate time until completion (SD)
        timeB = timeit.default_timer()
        
        #####
        ## BD probability
        #####
        #### Scaling BD mass-distribution with host star mass  ####
        mdis_bd=d.m**(-p.alpha_bd)
        ## Choose mass- or mass-ratio distribution
        if p.q_flag_bd == 1:
            d.q = (d.m/star_mass[ii])
            qdis_bd = (d.q)**-p.alpha_bd
            i = np.where( (d.q>= p.mn_min_bd) & (d.q<= p.mn_max_bd))[0]
            intm = integrate.simps(qdis_bd[i],np.log10(d.q[i]))
        else:
            i = np.where( (d.m>= p.mn_min_bd) & (d.m<= p.mn_max_bd))[0]
            intm = integrate.simps(mdis_bd[i], np.log10(d.m[i]))
        j = np.where( (d.a>= p.an_min_bd) & (d.a<= p.an_max_bd))[0]
        inta = integrate.simps(d.adis_bd[j], np.log10(d.a[j]))
        k_bd = p.P_bd/(intm*inta)
        
        # Can't have companion mass greater than stellar mass
        if p.m_max_bd > star_mass[ii]:
            tmp_m_max_bd = star_mass[ii]
        else:
            tmp_m_max_bd = p.m_max_bd
        if p.q_flag_bd == 1:
            Pbd = prob_mean(d.q, qdis_bd, d.a, d.adis_bd, p.m_min_bd/star_mass[ii], tmp_m_max_bd/star_mass[ii],
            p.a_min_bd, p.a_max_bd, k_bd, p)
        else:
            Pbd = prob_mean(d.m, mdis_bd, d.a, d.adis_bd, p.m_min_bd, tmp_m_max_bd,
            p.a_min_bd, p.a_max_bd, k_bd, p)
     
        #####
        ## Planet probability
        #####
        if p.planet_sma == 'flat':
            adis_pl = adis_flat
        else:
            adis_pl = d.adis

        if p.q_flag_pl == 1:
            d.q = d.mdis_ref.val/star_mass[ii]
            qdis_pl = (d.mdis_ref.val/star_mass[ii])**-p.alpha
            i = np.where( (d.q>= p.mn_min_pl) & (d.q<= p.mn_max_pl))[0]
            intm = integrate.simps(qdis_pl[i],np.log10(d.q[i]))
        else:
            i = np.where( (d.m>= p.mn_min_pl) & (d.m<= p.mn_max_pl))[0]
            intm = integrate.simps(mdis[i], np.log10(d.m[i]))
        j = np.where( (d.a>= p.an_min_pl) & (d.a<= p.an_max_pl))[0]
        inta = integrate.simps(adis_pl[j], np.log10(d.a[j]))
        k_pl = p.P_pl/(intm*inta)
        
        if p.q_flag_pl == 1:
            qdis_pl = (d.mdis_ref.val/star_mass[ii])**(-p.alpha)
#            d.k_pl = 1/p.A_pl
            if p.planet_sma == 'lognormal':
                Ppl = prob_mean(d.q, qdis_pl, d.a, adis_pl, p.m_min_pl/star_mass[ii], p.m_max_pl/star_mass[ii],
                p.a_min_pl, p.a_max_pl, k_pl, p)         # Removed p.m_max_pl and now going for 0.1*star_mass[ii] of star mass
            if p.planet_sma == 'flat':
                Ppl = prob_mean(d.q, qdis_pl, d.a, adis_pl, p.m_min_pl/star_mass[ii], p.m_max_pl/star_mass[ii],
                p.a_min_pl, p.a_max_pl, k_pl, p)        # Removed p.m_max_pl and now going for 0.1*star_mass[ii] of star mass
                #print(Ppl.P_mean) #planet frequency
                
        #####
        ## Importing contrast curves and evolutionary models
        #####
        age_pl = age[ii]
        pl_detected_fraction[ii]=0.
        ii_flag=str(ii+1)
        # Check that contrast mag and sep are finite values
        iokc = np.where(np.isfinite(contr_sep_arr[ii]) &
            np.isfinite(contr_mag_arr[ii]))[0]
        contr_mag =  contr_mag_arr[ii][iokc]
        contr_sep = contr_sep_arr[ii][iokc]

        #change-new
        if p.model_bd == 'BEX':
            modint_bd = interpol_mod_BEX('./models/BEX_evol_mags_-2_MH_0.00.dat', age_pl)
            mass_bd = modint_bd.mass_pl
            if p.band_bd == 'F356W':
                mag_band_bd = modint_bd.modint[21]  #25: F1000W, 22: F444W, 21: F356W, 10:M band
            elif p.band_bd == 'F444W':
                mag_band_bd = modint_bd.modint[22]
            elif p.band_bd == 'F1000W':
                mag_band_bd = modint_bd.modint[25]
            elif p.band_bd == 'F1500W':
                mag_band_bd = modint_bd.modint[27]
            elif p.band_bd == 'F2100W':
                mag_band_bd = modint_bd.modint[29]

    #change-new-
        if p.model == 'BEX':
            modint = interpol_mod_BEX('./models/BEX_evol_mags_-2_MH_0.00.dat', age_pl)
            mass_pl = modint.mass_pl
            if p.band == 'F356W':
                mag_band = modint.modint[21]  #25: F1000W, 22: F444W, 21: F356W, 10:M band
            elif p.band == 'F444W':
                mag_band = modint.modint[22]
            elif p.band == 'F1000W':
                mag_band = modint.modint[25]
            elif p.band == 'F1500W':
                mag_band = modint.modint[27]
            elif p.band == 'F2100W':
                mag_band = modint.modint[29]

    #########################################################################
        # Print probabilities
        print("New planet frequency: %1.4f" % Ppl.P_mean)
        print("New BD frequency: %1.4f" % Pbd.P_mean)
        
        # ----------------------------------------------------------------------
        # Loop over the realizations of the survey we want to run
        for jj in range(0, n_real):
            
            #######
            ## Selection of # planets and BDs from Poisson distributions with
            ## mean k_pl and k_bd, respectively
            #######
            
            N_pl = np.random.poisson(Ppl.P_mean)
            N_bd = np.random.poisson(Pbd.P_mean)

            ######################
            ## PLANET generation
            ######################
            # Loop over the planets we generated
            
            if N_pl >= 1:
                pl_generated_prob[ii] += 1
                combined_pl_generated[jj] += 1
                
                ## Drawing masses, SMAs, inclinations and eccentricities from the
                # correct planet distribution for the n planets we created
                log_mass = my_rand_dist(N_pl, Ppl.logm, Ppl.mdisn)
                if p.q_flag_pl == 1:
                    log_mass += np.log10(star_mass[ii])
                log_a = my_rand_dist(N_pl, Ppl.loga, Ppl.adisn)
                if p.planet_sma == 'flat':
                    log_a = np.log10(my_rand_dist(N_pl, 10**Ppl.loga, Ppl.adisn))
                incl = my_rand_dist(N_pl, d.i, d.idis)
                ecc = my_rand_dist(N_pl, d.e, d.edis)
                log_q = log_mass-np.log10(star_mass[ii])
                nn[jj] = nn[jj]+1
                ############
                # Calculate physical separation of planets from the mean and
                # eccentric anomalies,  the inclination, and the eccentricity.
                # (e.g. http://farside.ph.utexas.edu/teaching/celestial/Celestialhtml/node33.html),
                
                for k in range(0, N_pl):    # but what if N_pl is 1...?
                    sep_au = get_sep_au(d.eta, incl[k], ecc[k], log_a[k])
                    sep_as = sep_au/dist[ii]#; print(sep_as,contr_sep.min())
                    
                    mag_pl, mag_limit = get_limits_func(log_mass[k], mass_pl,
                        mag_band, sep_as, dist[ii], contr_sep, contr_mag)

                    cii = comp_in_inst(mag_pl, mag_limit, sep_as, contr_sep)
                        
                    #print(mag_pl, mag_limit)
                    w_pl[jj] += cii.w_cp
                    pl_detection[jj] += cii.cp_detection
                    z_pl[jj] += cii.z_cp
                    numbout = cii.numbout
                    
                    Fout.write("%1.3f        %1.3f        %1.3f    %1.3f    %1.1f    %1.3f    %1.3f    %1.3f    %i    %i    %i\n" % \
                            (mag_pl, log_mass[k], log_q[k], log_a[k], sep_au, sep_as,
                            incl[k], ecc[k], numbout, ii, jj))
     
                                    
            # endfor
            # The following is outside k but inside jj

            if pl_detection[jj] >= 1:
                pl_detection_prob[ii]+=1
                combined_pl_detected[jj]+=1
                pl_detected_fraction[ii]+=pl_detection[jj]
            pl_detection[jj]=0
    # ----------------------------------------------------------------------
            
            ##########################
            ## BD Generation
            ##########################
            if N_bd >= 1:
                bd_generated_prob[ii] += 1
                combined_bd_generated[jj] += 1
                
                ## Drawing masses, SMAs, inclinations and eccentricities from the
                # correct BD distribution for the n BDs we created
                
                log_mass = my_rand_dist(N_bd, Pbd.logm, Pbd.mdisn)
                if p.q_flag_bd == 1:
                    log_mass += np.log10(star_mass[ii])
                log_a = my_rand_dist(N_bd, Pbd.loga, Pbd.adisn)
                incl = my_rand_dist(N_bd, d.i, d.idis)
                ecc = my_rand_dist(N_bd, d.e, d.edis)
                log_q = log_mass-np.log10(star_mass[ii])
                nn[jj] = nn[jj]+1

                for k in range(0, N_bd):    # but what if N_pl is 1...?
                    sep_au = get_sep_au(d.eta, incl[k], ecc[k], log_a[k])
                    sep_as = sep_au/dist[ii]
                    mag_bd, mag_limit_bd = get_limits_func(log_mass[k], mass_bd,
                        mag_band_bd, sep_as, dist[ii], contr_sep, contr_mag)

                    cii = comp_in_inst(mag_bd, mag_limit_bd, sep_as, contr_sep)
                        
                    w_bd[jj] += cii.w_cp
                    bd_detection[jj] += cii.cp_detection
                    z_bd[jj] += cii.z_cp
                    numbout = cii.numbout+2        # BDs have +2 compared to planets

                    Fout1.write("%1.3f        %1.3f        %1.3f    %1.3f    %1.1f    %1.3f    %1.3f    %1.3f    %i    %i    %i\n" % \
                    (mag_bd, log_mass[k], log_q[k], log_a[k], sep_au, sep_as,
                    incl[k], ecc[k], numbout, ii, jj))
                    
            # Outside k but inside jj
            if bd_detection[jj] >= 1:
                bd_detection_prob[ii]+=1
                combined_bd_detected[jj]+=1
                bd_detected_fraction[ii]+=bd_detection[jj]
            bd_detection[jj]=0
            
    Fout.close()
    Fout1.close()
##

#    #######################
#    #### OVERALL STATS ####
#    #######################
tot_pl = sum(w_pl) + sum(z_pl)
tot_bd = sum(w_bd) + sum(z_bd)
P_det_cre = float((sum(w_pl)+sum(w_bd))/(tot_pl + tot_bd))
null_pl = np.exp(-float(sum(pl_detected_fraction)/n_real))
null_bd = np.exp(-float(sum(bd_detected_fraction)/n_real))
#
plep = len(combined_pl_generated[combined_pl_generated>=1])/n_real
bdep = len(combined_bd_generated[combined_bd_generated>=1])/n_real
pldp = len(combined_pl_detected[combined_pl_detected>=1])/n_real
bddp = len(combined_bd_detected[combined_bd_detected>=1])/n_real

st.write("Created %i planets" % int(tot_pl))
st.write("Detected %i planets" % sum(w_pl))
st.write("Created %i BDs" % int(tot_bd))
st.write("Detected %i BDs" % sum(w_bd))
st.write("P = detected / created = %1.4f" % P_det_cre)
st.write(" =============================================== ")
st.write("Overall planet existence probability: %1.4f" % float(plep))
st.write("Overall planet detection probability: %1.4f" % float(pldp))
st.write("Overall # of planets: %1.3f" % float(sum(pl_detected_fraction) / n_real))
st.write("Overall BD existence probability: %1.4f" % float(bdep))
st.write("Overall BD detection probability: %1.4f" % float(bddp))
st.write("Overall # of BDs: %1.3f" % float(sum(bd_detected_fraction) / n_real))
st.write(" = = = = = = = = = = = = = = = = = = = =")
st.write("Planet null-detection probability: %1.3f" % float(null_pl))
st.write("BD null-detection probability: %1.3f" % float(null_bd))

n = p.n_real
n_real = p.n_real

dat_pl = np.genfromtxt('./PL_out.dat', names=True)
dat_bd = np.genfromtxt('./BD_out.dat', names=True)

# Convert to pandas DataFrame for easier saving
df_pl = pd.DataFrame(dat_pl)
df_bd = pd.DataFrame(dat_bd)

# Save DataFrames as temporary CSV files and provide download buttons
df_pl_csv = df_pl.to_csv(index=False).encode('utf-8')
df_bd_csv = df_bd.to_csv(index=False).encode('utf-8')

# Download buttons
st.download_button(
    label="Download Planets Data",
    data=df_pl_csv,
    file_name="PL_out.csv",
    mime="text/csv"
)

st.download_button(
    label="Download BD Data",
    data=df_bd_csv,
    file_name="BD_out.csv",
    mime="text/csv"
)

try:
    flag_pl = dat_pl['flag']
    flag_bd = dat_bd['flag']
    jj_pl = dat_pl['jj']
    jj_bd = dat_bd['jj']
    lm_pl = dat_pl['log_mass']
    lm_bd = dat_bd['log_mass']
    la_pl = dat_pl['log_a']
    la_bd = dat_bd['log_a']
    la_as_pl = dat_pl['sep_as']
    la_as_bd = dat_bd['sep_as']
    mag_pl = dat_pl['mag_pl']
    mag_bd = dat_bd['mag_pl']

    # Planets
    gen_pl = np.append(np.where(flag_pl == 0)[0], np.where(flag_pl == 1)[0])
    gen_pl = np.append(gen_pl, np.where(flag_pl == 6)[0])
    gen_pl = np.append(gen_pl, np.where(flag_pl == 7)[0])

    det_pl1 = np.where(flag_pl == 1)[0]
    det_pl2 = np.where(flag_pl == 7)[0]
    det_pl = np.append(det_pl1, det_pl2)

    nr_pl = int(n*(len(det_pl) / len(gen_pl)))        # n-ratio planets
    rg_pl = np.unique(np.random.choice(gen_pl, n))          # Random generated
    rd_pl = np.unique(np.random.choice(det_pl, nr_pl))       # Random detected

    #----------------------------------------------------------------------
    #BDs
    gen_bd = np.append(np.where(flag_bd == 2)[0], np.where(flag_bd == 3)[0])
    gen_bd = np.append(gen_bd, np.where(flag_bd == 8)[0])
    gen_bd = np.append(gen_bd, np.where(flag_bd == 9)[0])

    det_bd1 = np.where(flag_bd == 3)[0]
    det_bd2 = np.where(flag_bd == 9)[0]
    det_bd = np.append(det_bd1, det_bd2)

    nr_bd = int(n*(len(det_bd) / len(gen_bd)))        # n-ratio planets
    rg_bd = np.unique(np.random.choice(gen_bd, n))          # Random generated
    rd_bd = np.unique(np.random.choice(det_bd, nr_bd))       # Random detected

    # ----------------------------------------------------------------------

    # Convert to earht masses  10**log_mass[rn_pl] * 332946 # Mearth

    #Plotting planets

    source_generated = ColumnDataSource(data={'separation': 10**la_pl[gen_pl], 'magnitude': mag_pl[gen_pl]})
    source_detected = ColumnDataSource(data={'separation': 10**la_pl[det_pl], 'magnitude': mag_pl[det_pl]})

    # Initialize the Bokeh figure
    p = figure(title="Planets - "+pl_type,
               x_axis_label="Separation (AU)", y_axis_label="Apparent Magnitude",
               x_axis_type="log", y_range=(27, 0), width=600, height=400)

    # Plot generated and detected planets
    p.circle('separation', 'magnitude', source=source_generated, color="black", size=7, alpha=0.2, legend_label=f"# Generated: {len(gen_pl)}")
    p.circle('separation', 'magnitude', source=source_detected, color="red", size=7, alpha=0.7, legend_label=f"# Detected: {len(det_pl)}")

    # Plot contrast curves from contr_sep_arr and contr_mag_arr
    for i in range(len(contr_sep_arr)):
        source_contrast = ColumnDataSource(data={'separation': contr_sep_arr[i]*dist[i], 'magnitude': contr_mag_arr[i]})
        p.line('separation', 'magnitude', source=source_contrast, color="grey", line_width=1.5, alpha=0.6)

    # Customize appearance
    p.legend.title = "Legend"
    p.legend.location = "top_right"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"

    # Display the plot in Streamlit
    st.bokeh_chart(p)
    
    #mass-magnitude space
    source_generated = ColumnDataSource(data={'separation': 10**la_pl[gen_pl], 'mass': 10**lm_pl[gen_pl]/0.0009545942})
    source_detected = ColumnDataSource(data={'separation': 10**la_pl[det_pl], 'mass': 10**lm_pl[det_pl]/0.0009545942})

    # Initialize the Bokeh figure
    p = figure(title="Planets - "+pl_type,
               x_axis_label="Separation (AU)", y_axis_label="Mass (MJ)",
               x_axis_type="log", y_range=(0, 1), width=600, height=400)

    # Plot generated and detected planets
    p.circle('separation', 'mass', source=source_generated, color="black", size=7, alpha=0.2, legend_label=f"# Generated: {len(gen_pl)}")
    p.circle('separation', 'mass', source=source_detected, color="red", size=7, alpha=0.7, legend_label=f"# Detected: {len(det_pl)}")

#    # Plot contrast curves from contr_sep_arr and contr_mag_arr
#    for i in range(len(contr_sep_arr)):
#        source_contrast = ColumnDataSource(data={'separation': contr_sep_arr[i]*dist[i], 'magnitude': contr_mag_arr[i]})
#        p.line('separation', 'magnitude', source=source_contrast, color="grey", line_width=1.5, alpha=0.6)

#    for name in range(names):
#        seps_arcsec = calcons_json[names[name]]['F444W']['seps_arcsec']
#        sensitivity_mag = calcons_json[names[name]]['F444W']['5sig_maskmag']
#        
#        lum = []
#        for i in np.array(sensitivity_mag):
#            lum.append((get_interpolated_mass(np.log10(184*10**(-0.4*i)*4*np.pi*dist[name]**2), age[name])))
#        
#        plt.plot(seps_arcsec, sensitivity, label=name)
#        
    # Customize appearance
    p.legend.title = "Legend"
    p.legend.location = "top_right"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"

    # Display the plot in Streamlit
    st.bokeh_chart(p)
    

    #Plotting BDs

    source_generated = ColumnDataSource(data={'separation': 10**la_bd[gen_bd], 'magnitude': mag_bd[gen_bd]})
    source_detected = ColumnDataSource(data={'separation': 10**la_bd[det_bd], 'magnitude': mag_bd[det_bd]})

    # Initialize the Bokeh figure
    p = figure(title="Brown Dwarfs",
               x_axis_label="Separation (AU)", y_axis_label="Apparent Magnitude",
               x_axis_type="log", y_range=(27, 0), width=600, height=400)

    # Plot generated and detected planets
    p.circle('separation', 'magnitude', source=source_generated, color="black", size=7, alpha=0.2, legend_label=f"# Generated: {len(gen_bd)}")
    p.circle('separation', 'magnitude', source=source_detected, color="blue", size=7, alpha=0.7, legend_label=f"# Detected: {len(det_bd)}")

    # Plot contrast curves from contr_sep_arr and contr_mag_arr
    for i in range(len(contr_sep_arr)):
        source_contrast = ColumnDataSource(data={'separation': contr_sep_arr[i]*dist[i], 'magnitude': contr_mag_arr[i]})
        p.line('separation', 'magnitude', source=source_contrast, color="grey", line_width=1.5, alpha=0.6)

    # Customize appearance
    p.legend.title = "Legend"
    p.legend.location = "top_right"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"

    # Display the plot in Streamlit
    st.bokeh_chart(p)
    
    #Plotting BDs

    source_generated = ColumnDataSource(data={'separation': 10**la_bd[gen_bd], 'mass': 10**lm_bd[gen_bd]/0.0009545942})
    source_detected = ColumnDataSource(data={'separation': 10**la_bd[det_bd], 'mass': 10**lm_bd[det_bd]/0.0009545942})

    # Initialize the Bokeh figure
    p = figure(title="Brown Dwarfs",
               x_axis_label="Separation (AU)", y_axis_label="Mass (MJ)",
               x_axis_type="log", y_range=(0, 100), width=600, height=400)

    # Plot generated and detected planets
    p.circle('separation', 'mass', source=source_generated, color="black", size=7, alpha=0.2, legend_label=f"# Generated: {len(gen_bd)}")
    p.circle('separation', 'mass', source=source_detected, color="blue", size=7, alpha=0.7, legend_label=f"# Detected: {len(det_bd)}")

#    # Plot contrast curves from contr_sep_arr and contr_mag_arr
#    for i in range(len(contr_sep_arr)):
#        source_contrast = ColumnDataSource(data={'separation': contr_sep_arr[i]*dist[i], 'magnitude': contr_mag_arr[i]})
#        p.line('separation', 'magnitude', source=source_contrast, color="grey", line_width=1.5, alpha=0.6)

    # Customize appearance
    p.legend.title = "Legend"
    p.legend.location = "top_right"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"

    # Display the plot in Streamlit
    st.bokeh_chart(p)

    # Set up your Streamlit application
    st.subheader("Detection Probability Distribution")
    jj_pl = np.random.randint(0, n_real, size=n_real)  # Example data
    jj_bd = np.random.randint(0, n_real, size=n_real)  # Example data
    flag_pl = np.random.randint(0, len(objnam), size=n_real)     # Example data
    flag_bd = np.random.randint(0, len(objnam), size=n_real)    # Example data

    # Initialize arrays for detected and undetected planets
    w_pl = []      # detected planets
    w_bd = []
    z_pl = []     # undetected planets
    z_bd = []

    # Populate w_pl, w_bd, z_pl, z_bd
    for i in range(n_real):
        jj_ipl = np.where(jj_pl == i)[0]
        jj_ibd = np.where(jj_bd == i)[0]
        
        # Undetected
        ud_ipl = np.where(flag_pl[jj_ipl] == 0)[0]
        ud_ibd = np.where(flag_bd[jj_ibd] == 2)[0]
        z_pl.append(len(ud_ipl))
        z_bd.append(len(ud_ibd))

        # Detected
        d_ipl1 = np.where(flag_pl[jj_ipl] == 1)[0]
        d_ipl2 = np.where(flag_pl[jj_ipl] == 7)[0]
        d_ipl = np.append(d_ipl1, d_ipl2)
        d_ibd1 = np.where(flag_bd[jj_ibd] == 3)[0]
        d_ibd2 = np.where(flag_bd[jj_ibd] == 9)[0]
        d_ibd = np.append(d_ibd1, d_ibd2)
        
        w_pl.append(len(d_ipl))
        w_bd.append(len(d_ibd))

    # Outside of for-loop, make some statistics
    pp = np.array(w_pl) + np.array(w_bd)

    # Detection Probability Distribution
    bb = np.arange(len(objnam)) - 0.5
    hh1 = np.histogram(pp, bins=bb, density=True)

    # Create a Bokeh figure
    p = figure(title="Detected Planets and BDs", plot_width=600, plot_height=400)

    # Histogram for planets
    hist_pl, edges_pl = np.histogram(w_pl, bins=hh1[1], density=True)
    p.quad(top=hist_pl, bottom=0, left=edges_pl[:-1], right=edges_pl[1:],
           fill_color="green", fill_alpha=0.3, line_color="green", legend_label="Detected Planets")

    # Histogram for brown dwarfs
    hist_bd, edges_bd = np.histogram(w_bd, bins=bb, density=True)
    p.quad(top=hist_bd, bottom=0, left=edges_bd[:-1], right=edges_bd[1:],
           fill_color="red", fill_alpha=0.3, line_color="red", legend_label="Detected BDs")

    # Add line plot for histogram data
    x_hist = (hh1[1][1:] + hh1[1][:-1]) / 2  # Bin centers
    p.line(x_hist, hh1[0], line_color="blue", line_width=2, legend_label="Histogram")

    # Customize the plot
    p.xaxis.axis_label = "# Detections"
    p.yaxis.axis_label = "PDF"
    p.x_range.start = -0.5
    p.x_range.end = 5
    p.legend.title = "Legend"
    p.legend.label_text_font_size = "12pt"
    p.legend.title_text_font_size = "14pt"

    # Display the plot in Streamlit
    st.bokeh_chart(p)
except:
    pass
