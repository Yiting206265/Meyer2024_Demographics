
import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import interpolate
from astropy import constants as con
from scipy import stats
from scipy import integrate
import pandas as pd
import glob
import pickle
import timeit
import time
import sys
import os

# Set the title of the app
st.title("Page3")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Page 1", "Page 2", "Page 3"))
