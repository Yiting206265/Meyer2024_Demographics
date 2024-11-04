## Import modules
##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import interpolate
from astropy import constants as con
from scipy import stats
from scipy import integrate
import pandas as pd
import glob
from IPython import display
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
import pickle
import time
import sys
import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
# Import separate file containing useful functions
from functions_companion_predictions_JWST import *
import scipy
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = np.trapz

# Set the title of the app
st.title("Yield Predictions")

# File uploader for JSON file

uploaded_file = st.file_uploader("Upload your JSON file. The file should contain a dictionary of the name of the targets, the filter band, the 5-sigma contrast and separation.", type="json", key='up1')


if uploaded_file is None:
    f = open('./files/calcons_06052024.json')
else:
    f = uploaded_file

#if uploaded_file is not None:
    # Load the JSON data
try:
    
    calcons_json = json.load(f)

    # Display the data in a pandas DataFrame
    df = pd.DataFrame(calcons_json).T  # Transpose to get the correct orientation
    st.write("Data Preview:")
    st.dataframe(df)

    # List of names to plot
    names = df.index.tolist()  # Get the names from the DataFrame index

    # Create a figure for plotting
    fig, ax = plt.subplots()

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
    ax.legend()
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



    
# File uploader for the input object file
uploaded_file = st.file_uploader("Upload your object file (must be in text format)", type=["txt", "csv"], key='up2')

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
    inmag = df['W1']          # <-- Relevant for getting magnitude band correct!
    # Converting age from Myrs as in input file to Gyrs as in model grid
    age /= 1e3
    # number of stars in survey
    n_star = len(objnam)
except:
    print('File not uploaded')

st.title("Generation of Companions")

# Parameters stored in a class
p = input_pams()

#hardcoded
# Use q-ratio instead of masses for BDs?
p.q_flag_bd = 1    # 1 == Yes for q-ratio, otherwise m will be in solar masses.
p.q_flag_pl = 1
#### PLANET PARAMETERS ###
p.alpha = -1.39
p.A_pl = np.exp(-5.28)  #1 Normalization Variable
p.median_loga = 0.57 #SHINE Vigan 2021 #peak of log-normal distribution
p.sigma = 0.73
#### BD PARAMETERS ###
p.A_bd = np.exp(-4.08)
p.alpha_bd = 0.3 #was 0.25, depends on
p.median_loga_bd = 1.43   ## Extrapolated from Solar to BD
p.sigma_bd = 1.21 #SPHERE SHINE survey

#user input
# Number of real planets
p.n_real = st.slider("Number of Real Planets (p.n_real)", min_value=1, max_value=1000, value=100)


# Normalization bounds for planets
st.subheader("Normalization Bounds for Planets")
# Planet frequency
p.P_pl = st.number_input("Planet Frequency (p.P_pl)",  0.01, None, step=None, format=None, key=None)
p.an_min_pl = st.slider("Minimum Planet Separation (AU) (p.an_min_pl)", min_value=0.01, max_value=1000.0, value=0.01)
p.an_max_pl = st.slider("Maximum Planet Separation (AU) (p.an_max_pl)", min_value=1.0, max_value=1000.0, value=100.0)

# Minimum and maximum planet mass (in solar masses)
p.mn_min_pl = st.slider("Minimum Planet Mass (Jupiter Masses) (p.mn_min_pl)", min_value=0.001, max_value=200.0, value=1.0)
p.mn_max_pl = st.slider("Maximum Planet Mass (Jupiter Masses) (p.mn_max_pl)", min_value=0.001, max_value=200.0, value=75.0)
p.mn_min_pl = p.mn_min_pl*0.0009545942
p.mn_max_pl = p.mn_max_pl*0.0009545942

# Planet limits
st.subheader("Planet Limits")
p.a_min_pl = st.slider("Minimum Semi-Major Axis (AU) (p.a_min_pl)", min_value=0, max_value=1000, value=0)
p.a_max_pl = st.slider("Maximum Semi-Major Axis (AU) (p.a_max_pl)", min_value=0, max_value=1000, value=1000)

# Change parameters for planets
p.m_min_pl = st.slider("Minimum Planet Mass (in Jupiter Masses) (p.m_min_pl)", min_value=0.0009545942, max_value=200.0, value=1.0)
p.m_max_pl = st.slider("Maximum Planet Mass (in Jupiter Masses) (p.m_max_pl)", min_value=0.0009545942, max_value=200.0, value=75.0)
p.m_min_pl = p.m_min_pl*0.0009545942
p.m_max_pl = p.m_max_pl*0.0009545942


# BD limits
st.subheader("Brown Dwarf Limits")
p.P_bd = st.number_input("Brown Dwarf Frequency (p.P_pl)",  0.01, None, step=None, format=None, key=None)
p.an_min_bd = st.slider("Minimum BD Separation (AU) (p.an_min_bd)", min_value=0.0, max_value=10.0, value=0.0)
p.an_max_bd = st.slider("Maximum BD Separation (AU) (p.an_max_bd)", min_value=1.0, max_value=150.0, value=108.91175375)

# Minimum and maximum brown dwarf mass (in solar masses)
p.mn_min_bd = st.slider("Minimum BD Mass (Jupiter Masses) (p.mn_min_bd)", min_value=0.0009545942, max_value=200.0, value=3.0)
p.mn_max_bd = st.slider("Maximum BD Mass (Jupiter Masses) (p.mn_max_bd)", min_value=0.0009545942, max_value=200.0, value=100.0)
p.mn_min_bd *=0.0009545942
p.mn_max_bd *=0.0009545942

# BD mass limits
p.m_min_bd = st.slider("Minimum BD Mass (Jupiter Masses) (p.m_min_bd)", min_value=0.0009545942, max_value=200.0, value=3.0)
p.m_max_bd = st.slider("Maximum BD Mass (Jupiter Masses) (p.m_max_bd)", min_value=0.0009545942, max_value=200.0, value=100.0)
p.m_min_bd*=0.0009545942
p.m_max_bd*=0.0009545942


# BD semi-major axis limits
p.a_min_bd = st.slider("Minimum BD Semi-Major Axis (AU) (p.a_min_bd)", min_value=0.0, max_value=100.0, value=0.0)
p.a_max_bd = st.slider("Maximum BD Semi-Major Axis (AU) (p.a_max_bd)", min_value=0.0, max_value=200.0, value=108.0)

p.bd_sma = st.radio("Brown Dwarf SMA Distribution", ("lognormal", "logflat"))
p.planet_sma = st.radio("Planet SMA Distribution", ("lognormal", "logflat"))

#p.output_file = 'results_Kellen_superJ.dat'

#############################################
# Generating masses, SMAs, eccentricities and inclinations
d = generate_distributions(p)

if p.planet_sma == 'logflat':
    d.adis = 1 / (d.a * np.log10(p.a_max_pl / p.a_min_pl))    # Restructure sma distribution after normalizing

## Generating and detecing planet and BD properties ##
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

### Check if output directory exists, otherwise create it
#if not os.path.exists(output_path):
#    os.makedirs(output_path)
#    print("Created output path in:", output_path)
#    

##START_run
p.model = p.model_bd = st.radio("Evol Model", ("BEX", "Placeholder"))
p.band = p.band_bd = st.radio("Filter Band", ("F444W", "Placeholder"))
n_star = len(objnam)


pl_yield = []
bd_yield = []

import random
import numpy as np
from scipy import integrate

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
        
#    except Exception as e:
#        print(e)
#    finally:
#        Fout.close()  # Ensure the file is closed
#

    for ii in range(0, n_star):
        # Estimate time until completion (SD)
        timeB = timeit.default_timer()
        
        #####
        ## BD probability
        #####
        #### Scaling BD mass-distribution with host star mass  ####
        mdis_bd=d.m**p.alpha_bd
        ## Choose mass- or mass-ratio distribution
        if p.q_flag_bd == 1:
            d.q = (d.m/star_mass[ii])
            qdis_bd = (d.q)**p.alpha_bd
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
            adis_pl = 1 / (d.a * np.log10(p.a_max_pl / p.a_min_pl))
        else:
            adis_pl = d.adis


        if p.q_flag_pl == 1:
            qdis_pl = d.mdis_ref.val * star_mass[ii]**-p.alpha
            if p.planet_sma == 'lognormal':
                Ppl = prob_mean(d.q, qdis_pl, d.a, adis_pl, p.m_min_pl/star_mass[ii], p.m_max_pl/star_mass[ii],
                p.a_min_pl, p.a_max_pl, d.k_pl, p)         # Removed p.m_max_pl and now going for 0.1*star_mass[ii] of star mass
            if p.planet_sma == 'flat':
                Ppl = prob_mean(d.q, qdis_pl, d.a, adis_pl, p.m_min_pl/star_mass[ii], p.m_max_pl/star_mass[ii],
                p.a_min_pl, p.a_max_pl, d.k_pl, p)        # Removed p.m_max_pl and now going for 0.1*star_mass[ii] of star mass
        
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
            if p.band_bd == 'F444W':
                mag_band_bd = modint_bd.modint[22]  #25: F1000W, 22: F444W, 21: F356W, 10:M band

    #change-new-
        if p.model == 'BEX':
            modint = interpol_mod_BEX('./models/BEX_evol_mags_-2_MH_0.00.dat', age_pl)
            mass_pl = modint.mass_pl
            if p.band == 'F444W':
                mag_band = modint.modint[22]  #25: F1000W, 22: F444W, 21: F356W, 10:M band

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
#

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

dat_pl = np.genfromtxt('PL_out.dat', names=True)
dat_bd = np.genfromtxt('BD_out.dat', names=True)

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

source_generated = ColumnDataSource(data={'separation': la_as_pl[gen_pl], 'magnitude': mag_pl[gen_pl]})
source_detected = ColumnDataSource(data={'separation': la_as_pl[det_pl], 'magnitude': mag_pl[det_pl]})

# Initialize the Bokeh figure
p = figure(title="Planets",
           x_axis_label="Separation (arcseconds)", y_axis_label="Apparent Magnitude",
           x_axis_type="log", y_range=(27, 0), width=800, height=400)

# Plot generated and detected planets
p.circle('separation', 'magnitude', source=source_generated, color="black", size=7, alpha=0.2, legend_label=f"# Generated: {len(gen_pl)}")
p.circle('separation', 'magnitude', source=source_detected, color="red", size=7, alpha=0.2, legend_label=f"# Detected: {len(det_pl)}")

# Plot contrast curves from contr_sep_arr and contr_mag_arr
for i in range(len(contr_sep_arr)):
    source_contrast = ColumnDataSource(data={'separation': contr_sep_arr[i], 'magnitude': contr_mag_arr[i]})
    p.line('separation', 'magnitude', source=source_contrast, color="grey", line_width=1.5, alpha=0.6)

# Customize appearance
p.legend.title = "Legend"
p.legend.location = "top_right"
p.xaxis.axis_label_text_font_size = "12pt"
p.yaxis.axis_label_text_font_size = "12pt"

# Display the plot in Streamlit
st.bokeh_chart(p)

#Plotting BDs

source_generated = ColumnDataSource(data={'separation': la_as_bd[gen_bd], 'magnitude': mag_bd[gen_bd]})
source_detected = ColumnDataSource(data={'separation': la_as_bd[det_bd], 'magnitude': mag_bd[det_bd]})

# Initialize the Bokeh figure
p = figure(title="Brown Dwarfs",
           x_axis_label="Separation (arcseconds)", y_axis_label="Apparent Magnitude",
           x_axis_type="log", y_range=(27, 0), width=800, height=400)

# Plot generated and detected planets
p.circle('separation', 'magnitude', source=source_generated, color="black", size=7, alpha=0.2, legend_label=f"# Generated: {len(gen_bd)}")
p.circle('separation', 'magnitude', source=source_detected, color="red", size=7, alpha=0.2, legend_label=f"# Detected: {len(det_bd)}")

# Plot contrast curves from contr_sep_arr and contr_mag_arr
for i in range(len(contr_sep_arr)):
    source_contrast = ColumnDataSource(data={'separation': contr_sep_arr[i], 'magnitude': contr_mag_arr[i]})
    p.line('separation', 'magnitude', source=source_contrast, color="grey", line_width=1.5, alpha=0.6)

# Customize appearance
p.legend.title = "Legend"
p.legend.location = "top_right"
p.xaxis.axis_label_text_font_size = "12pt"
p.yaxis.axis_label_text_font_size = "12pt"

# Display the plot in Streamlit
st.bokeh_chart(p)


# Set up your Streamlit application
st.title("Detection Probability Distribution")
jj_pl = np.random.randint(0, n_real, size=n_real)  # Example data
jj_bd = np.random.randint(0, n_real, size=n_real)  # Example data
flag_pl = np.random.randint(0, 8, size=n_real)     # Example data
flag_bd = np.random.randint(0, 10, size=n_real)    # Example data

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
bb = np.arange(31) - 0.5
hh1 = np.histogram(pp, bins=bb, density=True)

# Create the first plot
st.subheader("Detection Probability Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(w_pl, bins=hh1[1], density=True, color='g', alpha=.3, label='Detected Planets')
ax1.hist(w_bd, bins=bb, density=True, color='red', alpha=.3, label='Detected BDs')
ax1.plot(hh1[0], '.-', label='Histogram')
ax1.set_ylabel('PDF', fontsize=18)
ax1.set_xlabel('# Detections', fontsize=18)
ax1.set_xlim([-0.5, 5])
ax1.legend()
st.pyplot(fig1)