#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import time
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from shared_code.fun_loaddata import *  # Import only needed functions
from shared_code.fun_dfcspeed import *
from shared_code.fun_utils import set_figure_params, get_paths
from tqdm import tqdm



#%% Define paths, folders and hash
# ------------------------ Configuration ------------------------

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%%
# USE_EXTERNAL_DISK = True
# ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien_caillette/') if USE_EXTERNAL_DISK \
#         else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien_caillette/')
# RESULTS_DIR = ROOT / Path('results')
paths['speed'] = paths['results'] / 'speed'
# paths['speed'].mkdir(parents=True, exist_ok=True)

TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")

SAVE_DATA = True

WINDOW_PARAM = (5,100,1)
LAG=lag=1
TAU=tau=5

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"
PROCESSORS = -1
#%%
# ------------------------ Load Data ------------------------

ts_data = ts= np.load(TS_FILE, allow_pickle=True)['ts']
cog_data = pd.read_csv(COG_FILE)


print(f"Loaded {len(ts_data)} time series")
print(f"Loaded cognitive data for {len(cog_data)} animals")

assert len(ts_data) == len(cog_data), "Mismatch between time series and cognitive data entries."
#%%
#Remove the animals with 400 points

ts_data_filt = np.array([w for w in ts_data if w.shape[0] != 400])



prefix='dfc'
time_window_range = np.arange(WINDOW_PARAM[0],
                              WINDOW_PARAM[1] + 1,
                              WINDOW_PARAM[2])
n_animals, _ , regions = ts_data_filt.shape
ts = ts_data_filt
#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

#%% 
# ------------------------ Compute DFC ------------------------

def get_tnet_window_range(time_window_range, prefix='dfc'):
    """
    Get the range of window sizes for tnet files.
    Args:
        prefix (str): Prefix for the tnet files.
    Returns:
        list: List of window sizes.
    """
    def compute_for_window_size_new(ws,prefix='dfc'):
        print(f"Starting DFC computation for window_size={ws}")
        start = time.time()
        dfc_stream = handler_tnet_analysis(
            ts,
            prefix=prefix,
            window_size=ws,
            lag=lag,
            n_jobs=1,  # Important: Set to 1 to avoid nested parallelism
            save_path=paths[prefix],
        )
        stop = time.time()
        print(f"Finished window_size={ws} in {stop - start:.2f} sec")

    # Run parallel dfc stream over window sizes 
    start = time.time()
    Parallel(n_jobs=min(PROCESSORS, len(time_window_range)))(
        delayed(compute_for_window_size_new)(ws, prefix) for ws in time_window_range
    )

    stop = time.time()
    print(f'{prefix} stream computation time {stop-start}')

    # Check for missing prefix files and compute if necessary function
    missing_files = check_and_rerun_missing_files(
        paths[prefix], prefix, time_window_range, lag, n_animals, regions
    )
get_tnet_window_range(time_window_range, prefix='dfc')

# %%
