# functions_companion_tool_JWST.py
# ----------------------------------------------------------------------
# Description:
"""
	Script to import functions to be used in BD_tool_general_mass.py
	//Companion_tool.ipynb
	
	Last update: 10-june-2022
	//Per Calissendorff
"""
# ----------------------------------------------------------------------
# Import modules
import numpy as np
from scipy import interpolate
from astropy import constants as con
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import pickle
import timeit
import time
import sys
import os


# ----------------------------------------------------------------------
## Read input parameters ##
###########################
# ~ class input_pams():
	# ~ def __init__(self, input_file):
		# ~ F = open(input_file, 'r')
		# ~ p = F.readlines()
		# ~ F.close()
		# ~ n = len(p)
		
		# ~ ####
		# ~ self.mod_file = p[0].split(' ',1)[0]
		# ~ self.output_file_1 = p[1].split(' ',1)[0]
		# ~ self.output_file_2 = p[2].split(' ',1)[0]
		# ~ self.n_real = int(p[3].split(' ',1)[0])
		# ~ self.ninst = p[4].split(' ',1)[0]
		# ~ self.band = p[5].split(' ',1)[0]
		# ~ self.band2 = p[6].split(' ',1)[0]
		# ~ self.alpha = float(p[7].split(' ',1)[0])
		# ~ self.planet_sma = ''.join(p[8].split(' ',1)[0].split()).lower()
		# ~ self.beta = float(p[9].split(' ',1)[0])
		# ~ self.median_loga = float(p[10].split(' ',1)[0])
		# ~ self.sigma = float(p[11].split(' ',1)[0])
		# ~ self.P_pl = float(p[12].split(' ',1)[0])
		# ~ self.mn_min_pl = float(p[13].split(' ',1)[0])
		# ~ self.mn_max_pl = float(p[14].split(' ',1)[0])
		# ~ self.an_min_pl = float(p[15].split(' ',1)[0])
		# ~ self.an_max_pl = float(p[16].split(' ',1)[0])
		# ~ self.m_min_pl = float(p[17].split(' ',1)[0])
		# ~ self.m_max_pl = float(p[18].split(' ',1)[0])
		# ~ self.a_min_pl = float(p[19].split(' ',1)[0])
		# ~ self.a_max_pl = float(p[20].split(' ',1)[0])
		# ~ self.model = p[21].split(' ',1)[0]
		# ~ self.model_bd = p[22].split(' ',1)[0]
		# ~ self.alpha_bd = float(p[23].split(' ',1)[0])
		# ~ self.median_loga_bd = float(p[24].split(' ',1)[0])
		# ~ self.sigma_bd = float(p[25].split(' ',1)[0])
		# ~ self.P_bd = float(p[26].split(' ',1)[0])
		# ~ self.mn_min_bd = float(p[27].split(' ',1)[0])
		# ~ self.mn_max_bd = float(p[28].split(' ',1)[0])
		# ~ self.an_min_bd = float(p[29].split(' ',1)[0])
		# ~ self.an_max_bd = float(p[30].split(' ',1)[0])
		# ~ self.m_min_bd = float(p[31].split(' ',1)[0])
		# ~ self.m_max_bd = float(p[32].split(' ',1)[0])
		# ~ self.a_min_bd = float(p[33].split(' ',1)[0])
		# ~ self.a_max_bd = float(p[34].split(' ',1)[0])

# Class to store parameters
class input_pams():
	pass


# Making a class that can refer to a stored value, not overwriting variable
class refval(object):
	def __init__(self, value): self.val=value
		## Dependable functions ##
		##########################

## Making some generic lognormal function to integrate
def lognorm_func(a, med_loga, sig):
	return np.exp(-(np.log10(a)-med_loga)**2/2/(sig)**2)

# Generic power law function to integrate
def powlaw_func(m, alph):
	return m**alph

# Generic uniform distribution function to integrate
# ~ def uniform_func(a, b):
	# ~ return 1/(b-a)

# IDL function INT_TABULAR(X, F)
def idl_tabulate(x, f, p=5) :
	""" Function similar to IDL INT_TABULAR(X,F)
		but will yield slightly different results because of intrinsic
		differences between Python and IDL in calculations and interpolation
		"""
	def newton_cotes(x, f) :
		if x.shape[0] < 2 :
			return 0
		rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
		weights = integrate.newton_cotes(rn)[0]
		return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
	ret = 0
	for idx in range(0, x.shape[0], p - 1) :
		ret += newton_cotes(x[idx:idx + p], f[idx:idx + p])
	return ret
########################################################################
	## Create mass and separation distribution ##
class generate_distributions():
	def __init__(self, p):
				
		## 1) Stellar mass
		n_sampl=1000
		t_min = 0.00001
		t_max = 1.0
		self.m = 10**(np.arange(n_sampl).astype(float)/(n_sampl-1) *
			np.log10(t_max/t_min) + np.log10(t_min)) # logarithmic
		## 2) AU
		n_sampl=1000
		t_min = 0.001
		t_max = 1000
		self.a = 10**(np.arange(n_sampl).astype(float)/(n_sampl-1) *
			np.log10(t_max/t_min) + np.log10(t_min)) # logarithmic 

		## 3) (1 alt)  q-ratio
		# ~ n_sampl=1000
		# ~ t_min = 1e-5
		# ~ t_max = 1.0
		# ~ self.q = 10**(np.arange(n_sampl).astype(float)/(n_sampl-1) *
			# ~ np.log10(t_max/t_min) + np.log10(t_min)) # logarithmic 		

		# Should below be other than arange(100)(?)
		self.i = np.pi/100 * np.arange(100)
		self.e = np.arange(0,100.)/100.+0.01

		# ----------------------------------------------------------------------
		## BD distribution
		# ~ self.adis_bd = np.exp(-(np.log10(self.a)-p.median_loga_bd)**2/2./(p.sigma_bd)**2)

		# ~ if p.bd_sma == 'model1':
			# ~ bd_m_ln = np.exp(-(np.log10(self.a)-1.43)**2/2./(1.21)**2)/(1.21 * np.sqrt(2*np.pi))
			# ~ bd_fgk_ln = np.exp(-(np.log10(self.a)-1.70)**2/2./(1.68)**2)/(1.68 * np.sqrt(2*np.pi))
			# ~ bd_a_ln = np.exp(-(np.log10(self.a)-2.72)**2/2./(0.79)**2)/(0.79 * np.sqrt(2*np.pi))
			# ~ self.adis_bd = bd_m_ln + bd_fgk_ln + bd_a_ln
		# ~ else:
		# To be consistent with Susemeihl & Meyer (2021), Meyer et al. (in prep)
		# Susemiehl & MM 			
		self.adis_bd = p.A_bd * np.exp(-(np.log10(self.a)-p.median_loga_bd)**2/2./(p.sigma_bd)**2)/(p.sigma_bd * np.sqrt(2*np.pi))
	
		## Planet distribution
		# Select functional shape for planet SMA distribution (powerlaw, lognorm)
		if p.planet_sma == 'powerlaw':
			self.adis = self.a**p.beta
		elif p.planet_sma == 'lognormal':
			self.adis = np.exp(-(np.log10(self.a)-p.median_loga)**2/2./(p.sigma)**2)/ (np.sqrt(2*np.pi) * p.sigma)		# Now consistent with BD?
			# (2*np.pi*sigma1b*a) # =df/dr ### df/dlog(r) = alog(10)*r*df/dr
		elif p.planet_sma == 'flat':
			# ~ self.adis = np.ones(len(self.a)) * (1/(p.an_max_pl - p.an_min_pl))
			self.adis = 1 / (self.a * np.log10(p.an_max_pl / p.an_min_pl))
			# ~ self.adis = np.ones(len(self.a)) * (1/np.log10(max(d.a)/min(d.a)))
		elif p.planet_sma == 'model1':
			# ~ self.adis = np.exp(-(np.log10(self.a)-p.median_loga)**2/2./(p.sigma)**2) \
				# ~ / (np.sqrt(2*np.pi) * p.sigma * self.a)
			self.adis = np.exp(-(np.log10(self.a)-p.median_loga)**2/2./(p.sigma)**2) 

		# Lognormal is discussed below
		# Select functional shape for planet mas distribution
		# snigle powerlaw
		self.mdis_ref = refval(self.m**(p.alpha))	# possibly modified later as fraction of mass

		# Normalization
		i = np.where( (self.m>= p.mn_min_pl) & (self.m<= p.mn_max_pl))[0]
		j = np.where( (self.a>= p.an_min_pl) & (self.a<= p.an_max_pl))[0]
		
		# ~ intm = idl_tabulate(np.log10(self.m[i]), self.mdis_ref.val[i])
		# ~ inta = idl_tabulate(np.log10(self.a[j]), self.adis[j])
        
		intm = integrate.simps(self.mdis_ref.val[i], np.log10(self.m[i]))
		inta = integrate.simps(self.adis[j], np.log10(self.a[j]))
		self.k_pl = p.P_pl/(intm * inta)

		# Select functional shape for inclination distribution
		# (assumed to be same for BDs and planets)
		self.idis=np.sin(self.i)

		# Select functional shape for eccentricity distribution
		# (assumed to be same for BDS and planets)
		self.edis=self.e*np.exp(-self.e**2/(2.*0.3**2))

		##
		# Generating mean anomalies (angle ranging from 0 to 360 deg) for orbit calculations
		# (http://farside.ph.utexas.edu/teaching/celestial/Celestialhtml/node33.html)
		self.eta = np.arange(101).astype(float)/float(101-1)*2.*np.pi
########################################################################


# Calculate mean of Poisson distribution and correct normalizations
class prob_mean:
	def __init__(self, m, mdis, a, adis, m_min, m_max, a_min, a_max, k, p):

		i = np.where( (m >= m_min) & (m <= m_max))[0]
		j = np.where( (a >= a_min) & (a <= a_max))[0]
		# ~ int_m = idl_tabulate(np.log10(m[i]), mdis[i])
		# ~ int_a = idl_tabulate(np.log10(a[j]), adis[j])
		int_m = integrate.simps(mdis[i], np.log10(m[i]))
		int_a = integrate.simps(adis[j], np.log10(a[j]))

		# Mean of the Poisson distirbution generating BDs and planets
		self.P_mean = k * (int_m * int_a)
		# Generating the distribution of masses and SMA with correct normalization
		self.logm = np.log10(m[i])		# Useful?
		self.loga = np.log10(a[j])
		self.mdisn = k * mdis[i]
		self.adisn = k * adis[j]

# ~ class prob_mean_BD:
	# ~ def __init__(self, m, mdis, a, adis, m_min, m_max, a_min, a_max, k, p):

		# ~ i = np.where( (m >= m_min) & (m <= m_max))[0]
		# ~ j = np.where( (a >= a_min) & (a <= a_max))[0]
		# ~ int_m = idl_tabulate(np.log10(m[i]), mdis[i])
		# ~ int_a = idl_tabulate(np.log10(a[j]), adis[j])		

		# ~ # Mean of the Poisson distirbution generating BDs and planets
		# ~ if p.bd_sma == 'model1':
			# ~ self.P_mean = p.A_bd * int_m * int_a
		# ~ else:
			# ~ self.P_mean = k * (int_m * int_a)
		# ~ # Generating the distribution of masses and SMA with correct normalization
		# ~ self.logm = np.log10(m[i])
		# ~ self.loga = np.log10(a[j])
		# ~ self.mdisn = k * mdis[i]
		# ~ self.adisn = k * adis[j]
########################################################################


# ----------------------------------------------------------------------
### IRDIS ###

# Read contrast curves from file
def read_contrastcurves(fil):
	data = np.genfromtxt(fil, usecols=[0,1,2,3], dtype=float)
	
	# Create a dictionary for the contrast curve
	out = {}
	out['targ'] = ['bright', 'median', 'faint']
	out['contrast_radii'] = [data[:,0], data[:,0], data[:,0]]
	out['contrast'] = [data[:,1],data[:,2],data[:,3]]
	return out
########################################################################


# Linear extrapolation using polyfit in logspace (all points)
def log_lin(x,y, xnew):
	lf = np.poly1d(np.polyfit(np.log10(x),y, 1))
	return lf(np.log10(xnew))
########################################################################

# linear extrapolation with interpolation in logspace (point by point)
def log_interp(x,y,xnew):
	lint = interpolate.interp1d(np.log10(x),y,kind='linear',fill_value='extrapolate')
	return lint(np.log10(xnew))

########################################################################
	## SPHERE stuff
def SPHERE_stuff(p, inmag):
	curves = []
	curves2 = []
	if p.instrument_name == 'IRDIS':
		irdis_append = [4,5,5.5]
		irdis_mag_limits = [5.55,7.3]	# magnitude "separators" for three-way contrast limits
		irdis_backgrd = 21.5	
		curves = load_sphere_curves(p.contrast_file[0], p.output_path, inmag=inmag,
			appnd=irdis_append, mag_limits=irdis_mag_limits,backgrd=irdis_backgrd)

	### ADD HERE OTHER INSTRUMENTS LIKE IFS ###
	if (p.ninst == '2'):
		if (p.instrument_name2=='IRDIS'):
			print("Add stuff for IRDIS... this part not done yet")
		elif (p.instrument_name2=='IFS'):
			ifs_append = []
			ifs_backgrd = 20.5	
			ifs_mag_limits = refval(irdis_mag_limits).val
			curves2 = load_sphere_curves(p.contrast_file[0], p.output_path, inmag=inmag,
			appnd=ifs_append, mag_limits=ifs_mag_limits,backgrd=ifs_backgrd)
				
	return curves, curves2

########################################################################
### JWST Stuff ####
########################################################################

## Ydwarf version -- all-in-one
def NIRCam_Ydwarfs(cc_file, inmag):
	curv_arrs = np.genfromtxt(cc_file, usecols=[0,1], skip_header=1)
	sep = curv_arrs[:,0]			# arcsec
	mag_contr = curv_arrs[:,1]		# magnitudes
	sep_out = []
	sensitivity = []
	for i in range(0, len(inmag)):
		sens= mag_contr + inmag[i]
		sep_out.append(sep)
		sensitivity.append(sens)
	return sep_out, sensitivity
	
## Updated NIRCam contrast curves from Ell, 2023 (april, updated august)
def NIRCam_real(ncp, inmag):
    tars = glob.glob(ncp[0]+'*-cal_save.json')
    n = len(tars)
    sep = []
    contrast = []
    contr_sens = []
    for i in range(0, n):
        with open(tars[i], 'r') as f:
            con_data = json.load(f)
            sep.append(con_data['seps'])
            contr_sens.append(con_data['corr_cons'])
            contrast.append(con_data['corr_cons'])
        
    curves = np.empty([2, np.shape(sep)[0], np.shape(sep)[1]])
    curves[0,:,:] = sep
    curves[1,:,:] = -2.5*np.log10(contr_sens)
    # ~ curves[2,:,:] =
    contr_sep_arr, contr_mag_arr = check_NIRCam_contrasts(curves, inmag)
    return [contr_sep_arr, contr_mag_arr, contrast]


# NIRCam_stuff
# ~ ncp = [rep+'MDwarfs/Mdwarfs/', p.band, subopt, nircam_pams]
def NIRCam_stuff(ncp, inmag):
	curv_arrs = load_NIRCam_contrast(ncp[0], ncp[1], ncp[2], ncp[3])
	contr_sep_arr, contr_mag_arr = check_NIRCam_contrasts(curv_arrs, inmag)
	return [contr_sep_arr, contr_mag_arr]
	

# Load NIRCam contrast curves
def load_NIRCam_contrast(rep, filt, subopt, nircam_pams):	
	# ~ fstr = '_'+filt+'_MASK430R_CIRCLYOT_5sig_'+subopt+'.csv'
	fstr = '_'+filt+'_MASK335R_CIRCLYOT_5sig_'+subopt+'.csv'
	tars = []
	for file in os.listdir(rep):
		if file.endswith(fstr):
			tars.append(file)
	
	n = len(tars)
	sep = []
	contr_sens = []
	sens_mag = []
	for i in range(0, n):
		df = pd.read_csv(rep+tars[i])
		sep.append(df.Arcsec)
		contr_sens.append(df['Contr_'+nircam_pams])
		sens_mag.append(df['Sen_'+nircam_pams])
	curves = np.empty([3, np.shape(sep)[0], np.shape(sep)[1]])
	curves[0,:,:] = sep							# Arcsec
	curves[1,:,:] = -2.5*np.log10(contr_sens)	# Contrast in magnitude
	curves[2,:,:] = sens_mag					# Magnitude sensitivity
	return curves

########################################################################
## simple_limit_contrast_curves and limits.
def check_NIRCam_contrasts(curves, mag):
	sensitivity = []
	sep_out = []	
	sep = curves[0]
	mag_contr = curves[1]
	if len(curves) < 3:
		backgrd = 28.0 * np.ones(len(sep))	# Close to SNR=10 for F444W in AB mag
	else:
		backgrd = curves[2]		# This background might be wrong		
	
	for i in range(0, len(mag)):
		sens = mag[i] + mag_contr[i]
		sok = np.where(sens>=backgrd[i])[0]		# where star + contrast greater than background
		if len(sok)>0:
			sens[sok] = backgrd[i][sok]			# set sensitivity limit at background
		# ~ sok = np.where(sens<backgrd)[0]		
		sep_out.append(sep[i])
		sensitivity.append(sens)
	return sep_out, sensitivity

# Scale and limit contrast curves
def scale_and_limit_contrast_curves(sep_arr,curve_arr,mag,appnd,mag_limits,backgrd):
	
	### Create contrast curve for each object 
	# ~ sensitivity = np.empty([np.shape(curve_arr)[1]+len(appnd), len(mag)])
	# ~ sep_out = np.empty([np.shape(curve_arr)[1]+len(appnd), len(mag)])
	sensitivity = []
	sep_out = []
	for i in range(0, len(mag)):
		star_mag = mag[i]
		if (star_mag <= mag_limits[0]):
			delta_mag = curve_arr[0]
			sep = sep_arr[0]
			# Bright
		elif ((star_mag > mag_limits[0]) & (star_mag <= mag_limits[1])):
			delta_mag = curve_arr[1]
			sep = sep_arr[0]
		elif (star_mag > mag_limits[1]):
			delta_mag=curve_arr[2]
			sep = sep_arr[0]
		
		##
		# Replace proper selection with just taking the brightest curve 
		# every time!
		##
		
		sens = delta_mag+star_mag
		if len(appnd) > 0:
			# Linear extrapolation in logspace
			lin = log_lin(sep[-2:],sens[-2:],appnd)
			# ~ lin = log_interp(sep[-2:],sens[-2:],appnd)	# interpolation instead of polynomial
			sep = np.append(sep,appnd)
			sens = np.append(sens,lin)
		
		sok = np.where(sens<backgrd)[0]
		# ~ sens = sens[sens<background]
		# ~ sep_out[:,i] = sep[sok]
		# ~ sensitivity[:,i] = sens[sok]
		sep_out.append(sep[sok])
		sensitivity.append(sens[sok])
		
	return sep_out, sensitivity
		
# ----------------------------------------------------------------------
# BTSettle models (extrapolated a bit?)
## Old version
class interpol_mod_BTSettl():
	def __init__ (self, input_model, age_star):	
		"""
		Indices: 0) = age (Gyr)
		1) = M/Ms, 2) = Teff, 3) = L/Ls, 4) = logg, 5) = R/Rs, 6) = D,  
		7) = Li/Li0, 8) = J, 9) = H, 10) = K, 11) = L, 12) = M, 13) = NIRC335
		
		"""
	
		mod_ages = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
			0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090,0.100,
			0.120,0.150,0.200,0.300,0.400,0.500,0.600,0.700,0.800,0.900,
			1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0]
		
		mod_age = find_nearest(mod_ages, age_star)
		
		n_sampl = 11
		t_max = 1e-2
		t_min = 1e-5
		newmasses = 10**(np.arange(n_sampl)/(n_sampl-1) * 
			np.log10(t_max/t_min)+np.log10(t_min))
		
		# Read models
		modat = np.genfromtxt(input_model, usecols=[0,1,2,3,8,9,10,11,12,13])
		mok = np.where(modat[:,0] == mod_age)[0]
		mass_pl = modat[mok,1]
		cub = modat[mok]

		# Keeping only lower masses 
		newmass_low = newmasses[newmasses < min(mass_pl)]
		x = [cub[0][1], cub[1][1], cub[2][1]]
		
		# ~ cubout = np.empty([len(newmass_low)+len(mass_pl), np.shape(cub)[1]-2])
		cubout = []
		for i in range(2, np.shape(cub)[1]):
			## Linear extrapolation of lower masses			
			# ~ y = [cub[0][i], cub[1][i], cub[2][i]]
			# ~ ylow = log_lin(x, y, newmass_low)	
			ylong = [it[i] for it in cub]
			
			## Cubic extrapolation of lower masses
			# ~ cubint = interpolate.interp1d(mass_pl, ylong, 
				# ~ kind='cubic', fill_value='extrapolate')
			# ~ ylow = cubint(newmass_low)

			# ~ cubout[:,i-2] = np.append(ylow, ylong)
			cubout.append(ylong)		# No extrapolation
			
		# Save output interpolated cube of model values
		# ~ newmass_pl = np.append(newmass_low, mass_pl)
		# ~ self.modint = cubout
		# ~ self.mass_pl = newmass_pl	
		
		# No extrapolation!
		self.modint = cubout
		self.mass_pl = mass_pl
  
  

## COND models
class interpol_mod_COND():
	def __init__ (self, input_model, age_star):	
		"""
		Indices: 0) = age (Gyr)
		1) = M/Ms, 2) = Teff, 3) = L/Ls, 4) = g, 5) = R, 6) = Mv, 7) = Mr,
		8) = Mi, 9) = Mj, 10) = Mh, 11) = Mk, 12) = Mll, 13) = Mm, 14) = W1
		
		"""
	
		mod_ages = [0.001,0.005, 0.01, 0.05, 0.1, 0.12, 0.5, 1.0, 5.0, 10.0]
		mod_age = find_nearest(mod_ages, age_star)

		# Read models
		modat = np.genfromtxt(input_model, skip_header=2, usecols=[0,1,2,3,13])
		mok = np.where(modat[:,0] == mod_age)[0]
		mass_pl = modat[mok,1]
		cub = modat[mok]

		cubout = []
		for i in range(2, np.shape(cub)[1]):
			ylong = [it[i] for it in cub]
			cubout.append(ylong)		# No extrapolation
			
		# No extrapolation!
		self.modint = cubout
		self.mass_pl = mass_pl


## 
# ----------------------------------------------------------------------
# Function to read bhac15 models.
def read_bhac15(file_name, mod_age):
	""" Purpose: Read evolutionary table from Baraffe et al. 2015 named
	BHAC15_iso.2mass and place values in data cube
	
		Input: File name of table
		"""		
	def table_iterator():
		with open(file_name, 'r') as modfile:
			for line in modfile:
				if line.startswith("!  t (Gyr)"):
					cage = float(line[12:20])	# Get age of table content
					if (cage == mod_age):		# Only want the specific age given
						break					
			for line in modfile:				# read rest of lines
				if not (line.startswith('!') or line.startswith(' !')):	# ignore nasty comments
					lis = (line.strip()).split("  ")	# clean up spaces a bit
					larr = []							# redifine line-array
					for li in lis: 						# step through elements in line
						if len(li) > 1:					# ignore empty spaces
							lif = float(li)				# make number sfloat
							larr = np.append(larr, lif)	# append to line-array
					if len(larr) > 1:					# ignore empty arrays
						yield larr						# return the filled array
				if line.startswith('\n'):				# stop at linebreak
					break
								# Now we gather the correct lines into an array
	tline = []
	for line in table_iterator():
		tline.append(line)
	return tline
########################################################################
# read_BEX models for JWST and Mdwarfs
def read_BEX(file_name, mod_age):
	## BEX_evol_mags_-2_MH_0.00.dat
	ll = []									# Line List
	with open(file_name) as f:
		lines = f.readlines()[60:]
		for line in lines:
			if line.strip().startswith(str(mod_age)):
				# ~ tmp = line.strip().split('|')
				tmp = [float(x) for x in line.strip().split('|')]
				ll.append(tmp)
	return ll

## Class to load BEX models and interpolate them
class interpol_mod_BEX():
	def __init__(self, input_model, age_star):
		"""
		Indices: 0) = ages
		1) = mass (earths), 2) = Rad (Jup), 3) = Lum (Jup) 4) = Teff,
		5) = logg, 6) = J (NACO), 7) = H (NACO), 8) = Ks (NACO),
		9) = Lp (NACO), 10) = Mp (NACO), ... 13) = W1, 14) = W2, 15) = W3, 
		16) = W4, 17) = F115W, ... 21) = F356W, 22) = F444W, ... 
		33) = Y (SPHERE), 34) = J (SPHERE), ... 39) = H2 (SPHERE), ...
		"""

		mod_ages = np.arange(60, 90, 1)/10								# BEX log10(age) range
		mod_age = find_nearest(mod_ages, np.log10(age_star*1e9))		# age closest to models
		cub = read_BEX(input_model, mod_age)
	
		n_sampl = 11
		t_max = 1e-3
		t_min = 3e-5
		newmasses = 10**(np.arange(n_sampl)/(n_sampl-1) * 
			np.log10(t_max/t_min)+np.log10(t_min))
		
		# Gathering all masses into one array
		# ~ mass_conv = 3.0034893488507934e-06		# Earth to Solar masses
		mass_pl = np.array([it[1] for it in cub])*3e-6
		
		# ~ mass_pl_low = np.array([cub[0][1], cub[1][1], cub[2][1]])*3e-6 	# three lowest masses
		# Keeping only lower masses (this correct? IDL code i Don't understand)
		newmass_low = newmasses[newmasses < mass_pl[0]]

		cubout=[]
		for i in range(1, np.shape(cub)[1]):
			y = [x[i] for x in cub]
			# Linear extrapolation of lower masses
			cint = interpolate.interp1d(mass_pl, y, fill_value='extrapolate')
			cubint = cint(newmass_low)
			cubout.append(np.append(cubint,y))
			## No more extrapolation
			# ~ cubout.append(y)
			
		# Save output interpolated cube of model values
		newmass_pl = np.append(newmass_low, mass_pl)
		self.modint = cubout
		self.mass_pl = newmass_pl
		# ~ self.mass_pl = mass_pl

# To get column in the 2d (well 3d here?) array, do
# col = [row[i] for row in array]		# where array = ll

########################################################################
# Find nearest in array
def find_nearest(arr, val):
	array = np.asarray(arr)
	idx = (np.abs(array - val)).argmin()
	return array[idx]

########################################################################


# Function to load bhac15-cond models and interpolate them
class interpol_mod_BHAC15():
	def __init__ (self, input_model, age_star):
		"""
		Indices:
		1) = M/Ms, 2) = Teff, 3) = L/Ls, 4) = g, 5) = R/Rs, 6) = Li/Li0,
		7) = F070Wa ...  26) = F356Wa, ... 31 = F444Wa, ...
		55) = F345Wb, ... 60) = F444Wb, ... 84) = F356Wab, ... 89) = F444Wab
		
		"""
	
		if ('COND' in input_model) or ('DUSTY' in input_model):
			mod_ages = [0.001,0.005,0.010,0.050,0.100,0.120,0.500,1.0,5.0,10.] # Gyr
		else:
			mod_ages = [0.0005,0.001,0.002,0.003,0.004,0.005,0.008,0.010,0.015,
				0.020,0.025,0.030,0.040,0.050,0.080,0.100,0.1200,0.200,0.300,
				0.400,0.500,0.625,0.800,1.0,2.0,3.0,4.0,5.0,8.0,10.]
		
		mod_age = find_nearest(mod_ages, age_star)
		cub = read_bhac15(input_model, mod_age)
		
		# Linear extrapolation
		# Take three closest values and extrapolate to lower masses in
		# log-scale for each quantity
		
		n_sampl = 11
		t_max = 1e-2
		t_min = 1e-5
		newmasses = 10**(np.arange(n_sampl)/(n_sampl-1) * 
			np.log10(t_max/t_min)+np.log10(t_min))
		
		# Gathering all masses into one array
		# ~ mass_pl = []
		# ~ for i in range(0, np.shape(cub)[0]):
			# ~ mass_pl.append(cub[i][0])
		mass_pl = [it[0] for it in cub]
		
		mass_pl_low = [cub[0][0], cub[1][0], cub[2][0]]	# three lowest masses
		# Keeping only lower masses (this correct? IDL code i Don't understand)
		newmass_low = newmasses[newmasses < mass_pl_low[0]]
		
		# ~ cubint = np.empty([len(newmass_low), np.shape(cub)[1]-1])
		cubout = np.empty([len(newmass_low)+len(mass_pl), np.shape(cub)[1]-1])
		for i in range(1, np.shape(cub)[1]):
			## Linear extrapolation of lower masses			
			# ~ y = [cub[0][i], cub[1][i], cub[2][i]]
			# ~ ylow = log_lin(mass_pl_low, y, newmass_low)	
			
			ylong = [it[i] for it in cub]
			
			## Cubic extrapolation of lower masses
			cubint = interpolate.interp1d(mass_pl, ylong, 
				kind='quadratic', fill_value='extrapolate')
			ylow = cubint(newmass_low)

			##
			
			cubout[:,i-1] = np.append(ylow, ylong)
			
		# Save output interpolated cube of model values
		newmass_pl = np.append(newmass_low, mass_pl)
		self.modint = cubout
		self.mass_pl = newmass_pl

########################################################################

# 
# custom function to take integral of table?
def int_tabulated(x,y):
	wid = (x-np.roll(x,1))[1:]
	ff = (y[0:len(y)-1]+y[1:])/2.
	return sum(ff*wid)
########################################################################

		
# my_random_distribution.pro
def my_rand_dist(N, xdist, ydist):
	ydist = ydist/np.max(ydist)	# Normalize ydis
	xdist = np.append(np.append(xdist[0]-1e-6, xdist), xdist[-1]+1e-6)
	ydist = np.append(np.append(0, ydist), 0)
	
	minx = xdist.min()
	maxx = xdist.max()
	
	ingral = int_tabulated(xdist, ydist)/(maxx-minx)
	n2 = int(float(N/(ingral)*8))	# (?)
	
	dbx = xdist[1:-1]		# shave off ends
	dby = ydist[1:-1]
	
	if n2 <= 5e7:
		resx = np.random.uniform(0,1,n2)*(maxx-minx)+minx
		resy = np.random.uniform(0,1,n2)
		disty = np.interp(resx, dbx, dby)
		# ~ ydiff = ydist - resy
		iiok = np.where(disty-resy>=0)[0]
		while len(iiok) < 1:
			resx = np.random.uniform(0,1,n2)*(maxx-minx)+minx
			resy = np.random.uniform(0,1,n2)
			disty = np.interp(resx, dbx, dby)
			iiok = np.where(disty-resy>=0)[0]
		vout = (resx[np.where(disty-resy >= 0)[0]])#[0:N] # hmm?
	else:
		resx = np.random.uniform(0,1,2e7)*(maxx-minx)+minx
		resy = np.random.uniform(0,1,2e7)
		disty = np.interp(resx, dbx, dby)
		vout = resx[np.where(disty-resy>=0)[0]]
		# ~ index = 1
		while len(vout) < N:
			resx = np.random.uniform(0,1,2e7)*(maxx-minx)+minx
			resy = np.random.uniform(0,1,2e7)
			disty = np.interp(resx, dbx, dby)
			vout = resx[np.where(disty-resy>=0)[0]]
			# ~ index += 1
		# ~ vout = vout[0:N]
	return vout[0:N]

########################################################################

# Get eta
def my_get_eta0(eta, ecc):
	# Uniformly draw a normalized time
	t0 = np.random.uniform()
	# define two functions
	y1 = eta
	y2 = 2*np.pi*t0 + ecc*np.sin(eta)
	# solve for intersection point
	ys = y1/y2
	return np.interp(1, eta, ys)-np.pi

########################################################################
# For getting separation in AU
def get_sep_au(eta, inc, ecc, loga):
	eeta = my_get_eta0(eta, ecc)
	r_au = 10**loga * (1-ecc*np.cos(eeta))
	tana2 = np.sqrt(1.+ecc)/np.sqrt(1.-ecc)*np.tan(eeta/2.)
	cosa = (1-tana2**2)/(1+tana2**2)
	sina = np.sqrt(1-cosa**2)
	sep_au = r_au*np.sqrt(cosa**2 + sina**2 *(np.cos(inc))**2)
	return sep_au

# For getting magnitude of companion and magnitude limits
def get_limits_func(logm, mass_cp, mag, sep_as, d, contr_sep, contr_mag):
	# ~ mag_cpint = interpolate.interp1d(mass_cp, mag, kind='quadratic', fill_value='extrapolate')
	# ~ mag_cp = mag_cpint(10**logm) +5*np.log10(d/10)
	mag_cpint = interpolate.interp1d(np.log10(mass_cp), mag, kind='quadratic', fill_value='extrapolate') #absolute magnitude in the model
	# ~ mag_cpint = interpolate.interp1d(np.log10(mass_cp), mag, kind='linear',bounds_error=False, fill_value=(99,99))
	mag_cp = mag_cpint(logm) +5*np.log10(d/10) #convert to apparent
	# ~ mag_limint = interpolate.interp1d(contr_sep, contr_mag, kind='linear', bounds_error=False, fill_value=(0,0))
	# ~ mag_limit = mag_limint(sep_as)
	# ~ # Testing log?
	mag_limint = interpolate.interp1d(np.log10(contr_sep), contr_mag, kind='linear', bounds_error=False, fill_value=(0,0)) #changed contr_mag[-1]
	mag_limit = mag_limint(np.log10(sep_as))
	
	# ~ mag_limit = np.interp(sep_as, contr_sep, contr_mag)
	## Testing linear polynomial extrapolation for companion mass-magnitude
	#mag_cp = log_lin(mass_cp[:3], mag[:3], 10**logm) + 5*np.log10(d/10)
	
	return mag_cp, mag_limit


########################################################################
# Estimating companion detection by instrument
class comp_in_inst:
	def __init__(self, mag_cp, mag_lim, sep_as, contr_sep):
		self.w_cp = 0
		self.cp_detection = 0
		self.z_cp = 0
		self.numbout = 0
		if ((mag_cp <= mag_lim) and (sep_as >= min(contr_sep)) and (sep_as <= max(contr_sep))):
			self.w_cp += 1				# w_pl[jj] += 1
			self.cp_detection += 1		# pl_det[jj] += 1
			self.numbout = 1			# write this number out before ii 
		else:
			self.z_cp += 1		# z_pl[jj] += 1		
		
		
# Estimating companion detection by instrument when 2 are used.
class comp_in_2inst:
	def __init__(self, mag_cp, mag_cp2, mag_lim, mag_lim2, sep_as, contr_sep, contr_sep2, ninst):
		self.w_cp = 0
		self.cp_detection = 0
		self.z_cp = 0
		self.numbout = 0

		# Detected in instrument #1 only
		if ((mag_cp <= mag_lim) and (sep_as >= min(contr_sep))\
		and (sep_as <= max(contr_sep))) and not ((mag_cp2 <= mag_lim2)\
		and (sep_as >= min(contr_sep2)) and (sep_as <= max(contr_sep2))):
			self.w_cp += 1
			self.cp_detection += 1
			self.numbout = 1

		# Detected in instrument 2 only
		if ((mag_cp2 <= mag_lim2) and (sep_as >= min(contr_sep2))\
		and (sep_as <= max(contr_sep2))) and not ((mag_cp <= mag_lim)\
		and (sep_as >= min(contr_sep)) and (sep_as <= max(contr_sep))):
			self.w_cp += 1
			self.cp_detection += 1
			self.numbout = 6

		# Detected in both instrumetns
		if (mag_cp <= mag_lim):
			if (sep_as >= min(contr_sep)):
				if (sep_as <= max(contr_sep)):
					if (mag_cp2 <= mag_lim2):
						if (sep_as >= min(contr_sep2)):
							if (sep_as <= max(contr_sep2)):
								self.w_cp += 1
								self.cp_detection += 1
								self.numbout = 7
		else:
			self.z_cp += 1


# Load SPHERE IRDIS or IFS contrast curves
def load_sphere_curves(contrast_file,output_path,inmag,appnd, mag_limits, backgrd):
	""" Scaled versions of bright/median/faint averages of Maud's or Rafaele's curves
	(possibly extended to 6''"""
	contrst = read_contrastcurves(contrast_file)	# average=n_star
	contr_sep_arr, contr_mag_arr = scale_and_limit_contrast_curves(contrst['contrast_radii'],
		contrst['contrast'],inmag,appnd=appnd,mag_limits=mag_limits,backgrd=backgrd)
	return [contr_sep_arr, contr_mag_arr]

# ----------------------------------------------------------------------
# Function for making plots
def plot_func_compcode(infile):
	dat = np.genfromtxt(infile, names=True)
	flag = dat['flag']
	jj = dat['jj']
	log_mass = dat['log_mass']
	log_a = dat['log_a']
	sep_au = dat['d_au']
	sep_as = dat['sep_as']
	gen = np.where(flag == 0)[0]
	det = np.where(flag == 1)[0]

	# Take a random set of elements from generated array
	n = 2000
	ng = np.unique(np.random.choice(gen, n))

	# Planet distribution mass vs sep
	mass_gen = 10**log_mass[ng] * 332946 # Mearth
	mass_det = 10**log_mass[det] * 332946 # Mearth 
	sep_gen = sep_au[ng]
	sep_det = sep_au[det]

	#################################
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(log_a[ng], np.log10(mass_gen), 'k.', alpha=.3, label='Generated')
	plt.plot(log_a[det], np.log10(mass_det), 'b.', alpha=.3, label='Detected')
	ax.set_xlabel(r"log separation [AU]")
	ax.set_ylabel(r"log planet mass $[M_{\rm Earth}]$")
	#ax.set_title(p.planet_sma+",   alpha= "+str(p.alpha))
	ax.legend(markerfirst=False)

	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	plt.plot(10**log_a[ng], mass_gen, 'k.', alpha=.3, label='Generated')
	plt.plot(10**log_a[det], mass_det, 'b.', alpha=.3, label='Detected')
	ax2.set_xlabel(r"Separation [AU]")
	ax2.set_ylabel(r"Planet mass $[M_{\rm Earth}]$")
	#ax2.set_title(p.planet_sma+",   alpha= "+str(p.alpha))
	ax2.legend(markerfirst=False)

########################################################################
########################################################################
