import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator

# Written by Michael R. Meyer, University of Michigan 
# Updated June 30, 2024 
# New Update October 28, 2024 

# The parameters of the two component model from Meyer et al. (2024)

# Brown Dwarf Companions 
# beta is the power-law exponent for the brown dwarf binary companion mass ratio distribution dN/dq ~ q^(beta) Meyer et al. 2024

beta = -0.3

# 68 % confidence intervals (hereafter CI) for beta [-0.34, 0.30]

# mean_bd = 1.70 is the mean of the log-10-normal in log-AU log(50) Raghavan et al. 2010 (physical separation) for FGK stars
# mean_bd = 1.43 for M dwarfs log-10(27) Winters et al. (2019) - physical separation not projected. Meyer et al. 2024 
# mean_bd = 2.72 for A stars log-10(525) De Rosa et al. (2014) - physical separation not projected. Meyer et al. 2024

mean_bd = 1.43

# square-root of the variance for companion orbital density log-10-normal distribution. R10 sigma_bd = 1.68 
# sigma_bd = 1.21 M dwarf Winters et al. (2019) aghavan et al. 20
# sigmba_bd = 0.79 A stars De Rosa et al. (2014)

sigma_bd = 1.21

# amplitude of the BD new fit 
A_bd_ln = -4.08 # with 68 % CI [-0.47,0.7]

A_bd = np.exp(A_bd_ln)

# power-law planet mass function values in q for M-FGK-A stars Meyer et al. 2024   
# Note power-law is fit as (- alpha) so alpha here is positive, but multiplied by (-1) in mass function below. 

alpha_pl = 1.39 # 68 % CI [-0.13,0.12] 

# In this code, we transform mu and sigma from natural log to log_10 in the orbital separation distribution.
# sigma is log_base10 of log-normal while fits are in natural ln. 
# sigma_logbase10 = sigma_naturallog / constant where constant = ln(10) = 2.302585092994046  
# Meyer et al. (2024) 6-25-24
#
A_pl_ln = -5.28 # 68 % CI [-0.83,0.93] 
mu_ln = 1.31  # 68 % CI [-0.22,0.18] 
sigma_pl_ln = 0.52 # 68 % CI [-0.07,0.06]

constant = 2.302585092994046

A_pl = np.exp(A_pl_ln)
mu_pl = (mu_ln)/constant
sigma_natural = np.exp(sigma_pl_ln)
sigma_pl = sigma_natural/constant
print(A_pl, sigma_pl, mu_pl, A_bd)

st.header("Two Component Model for Exoplanet and Brown Dwarf Companions (Meyer et al. 2024)")
st.write(r"$A_{pl}$:", A_pl)
st.write(r"$\sigma_{pl}$:", sigma_pl)
st.write(r"$\mu_{pl}$:", mu_pl)
st.write(r"$A_{bd}$:", A_bd)
st.write(r"$\sigma_{bd}$:", sigma_bd)
st.write(r"$\mu_{bd}$:", mean_bd)


# User input parameter host_mass in Solar masses, companion mass range of interest in Jupiter masses (min, max) 
# and orbital range of interest in AU (min, max) 

host_mass = st.number_input("Host Mass ($\mathrm{M_{\odot}}$)", 0.01, None, step=None, format=None, key=None)
Jup_min = st.number_input("Companion Minimum Mass ($\mathrm{M_{Jup}}$)", 0.01, None, step=None, format=None, key=None)
Jup_max = st.number_input("Companion Maximum Mass ($\mathrm{M_{Jup}}$)", 0.01, None, step=None, format=None, key=None)
q_Jupiter = 0.001*host_mass # Msun

a_min = st.number_input("Orbital Minimum Separation (AU)", 0.01, None, step=None, format=None, key=None)
a_max = st.number_input("Orbital Maximum Separation (AU)", 0.01, None, step=None, format=None, key=None)

# Defining the functions for mass and orbital separation distibutions for both brown dwarf and planets 

def mass_fctn_bd(q):
    return (np.log10(q)**(beta))#dq
def orbital_dist_bd(a):
    return (A_bd*np.exp((-(np.log10(a)-(mean_bd))**2.)/(2.*sigma_bd**2.)))/(2.0*np.pi*sigma_bd*a)#da
def mass_fctn_pl(q):
    return (np.log10(q)**(-alpha_pl))#dm
def orbital_dist_pl(a):
     return (A_pl/(2*np.pi*sigma_pl*a))*np.exp((-(np.log10(a)-(mu_pl))**2.)/(2.*sigma_pl**2.))#da

# mass function ranges are given in Jupiter masses, but passed in q (which depends on host star mass)
# orbital distributions are given in AU. 

f_bd = (integrate.quad(mass_fctn_bd, (np.log10(Jup_min/q_Jupiter)), (np.log10(Jup_max/q_Jupiter)))[0]*integrate.quad(orbital_dist_bd,a_min,a_max)[0])

f_pl = (integrate.quad(mass_fctn_pl, (np.log10(Jup_min/q_Jupiter)), (np.log10(Jup_max/q_Jupiter)))[0]*integrate.quad(orbital_dist_pl,a_min,a_max)[0])

#Output is the mean number of companions per star for the brown dwarf component (f_ bd) and planet part (f_pl).

print(f_bd,f_pl)
st.text("Frequency of Brown Dwarfs: {}".format(f_bd))
st.text("Frequency of Planets: {}".format(f_pl))
