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
import importlib

## Set the title of the app
#st.title("Main")
#
## Create a sidebar for navigation
#st.sidebar.title("Navigation")
#page = st.sidebar.radio("Go to", ("Page 1", "Page 2", "Page 3"))
#
## Load the corresponding page based on selection
#if page == "Page 1":
#    import frequency
#    importlib.reload(frequency)
#elif page == "Page 2":
#    import page2
#    importlib.reload(page2)
#    #page2.show()
#elif page == "Page 3":
#    import page3
#    page3.show()


# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Frequency", "Yield Prediction"))

# Load the corresponding page based on selection
if page == "Page 1":
    try:
        import frequency
        importlib.reload(frequency)
    except:
        pass
elif page == "Page 2":
    try:
    
        import page2
        importlib.reload(page2)
    except:
        pass
elif page == "Page 3":
    import page3
    page3.show()
