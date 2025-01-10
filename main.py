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

pages = {
    "Demographics": [
        st.Page("frequency.py", title="Frequency Calculator")
    ]
}

pg = st.navigation(pages)
pg.run()
