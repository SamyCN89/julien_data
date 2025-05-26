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
timeseries_folder = 'time_courses'
paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder=timeseries_folder,
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%%
data_ts = ts= np.load(paths['sorted'] / Path("ts_filtered_unstacked.npz"), allow_pickle=True)
ts_data_filt = np.array([w for w in ts_data if w.shape[0] != 400])


# TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
# COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")
PROCESSORS = -1

WINDOW_PARAM = (5,100,1)
LAG=lag=1
TAU=tau=5

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"
#%%
# ------------------------ Load Data ------------------------



print(f"Loaded {len(ts_data)} time series")
print(f"Loaded cognitive data for {len(cog_data)} animals")

assert len(ts_data) == len(cog_data), "Mismatch between time series and cognitive data entries."
#%%
#Remove the animals with 400 points




prefix='dfc'
time_window_range = np.arange(WINDOW_PARAM[0],
                              WINDOW_PARAM[1] + 1,
                              WINDOW_PARAM[2])
n_animals, _, regions = ts_data_filt.shape
ts = ts_data_filt
#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

#%% 
# ------------------------ Compute DFC ------------------------
import logging

logging.basicConfig(level=logging.INFO)

def handler_tenet_analysis(ts_data, prefix='dfc', window_size=7, lag=1, format_data='3D', save_path=None, n_jobs=-1):
    """
    Calculate temporal network analysis (dfc_stream, meta-connectivity) for time-series data.

    Parameters:
        ts_data (np.ndarray): 3D array (n_animals, n_regions, n_timepoints).
        window_size (int): Sliding window size.
        lag (int): Step size for the sliding window.
        format_data (str): '2D' for vectorized, '3D' for matrices.
        save_path (str): Directory to save results.
        n_jobs (int): Number of parallel jobs (-1 for all cores).

    Returns:
        np.ndarray: 4D array of DFC streams (n_animals, time_windows, roi, roi)
    """
    logger = logging.getLogger(__name__)

    #Set the parameters for the dfc_stream
    n_animals, _, nodes = ts_data.shape

    # Define the full save path based on parameters and save_path folder
    full_file_path = make_file_path(save_path, prefix, window_size, lag, n_animals, nodes)

    # Load from cache if possible
    print(f"full_file_path: {full_file_path}")
    if full_file_path is not None and full_file_path.exists():
        if prefix == 'dfc':
            return load_npz_cache(full_file_path, key="dfc_stream", label='dfc-stream')
        else:
            return load_npz_cache(full_file_path, key=prefix, label='meta-connectivity')
            # dfc_stream = load_cached_dfc(full_file_path)

    # Compute tenet streams in parallel
    logger.info(f"Computing {prefix} (window_size={window_size}, lag={lag})...")

    dfc_stream = np.array([ts2dfc_stream(
        ts_data[i], window_size, lag, format_data) 
        for i in range(n_animals)])
    # with parallel_backend("loky", n_jobs=n_jobs):
    #     dfc_stream = np.stack(Parallel()(
    #         delayed(ts2dfc_stream)(ts_data[i], window_size, lag, format_data) for i in range(n_animals)
    #     ))
    dfc_stream = dfc_stream.astype(np.float32)  # Convert to float32 for memory efficiency
    # get_save_path(save_path, window_size, lag, n_animals, nodes)
    # Save results if needed
    save_npz_stream(full_file_path, prefix='dfc_stream', dfc_stream=dfc_stream)
    return dfc_stream

def compute_for_1_window(ws, ts, prefix, lag, save_path):
    """
    Compute the prefix function for a given window size.
    """ 
    try:
        logging.info(f"Starting {prefix} computation for window_size={ws}")
        start = time.time()
        # print(f"Starting {prefix} computation for window_size={ws}")
        handler_tenet_analysis(
                    ts,
                    prefix=prefix,
                    window_size=ws,
                    lag=lag,
                    n_jobs=1,  # Important: Set to 1 to avoid nested parallelism
                    save_path=save_path,
                )
        stop = time.time()
        print(f"Finished window_size={ws} in {stop - start:.2f} sec")
    except Exception as e:
        logging.error(f"Error occurred during {prefix} computation for window_size={ws}: {e}")
        raise

def get_tenet4window_range(ts, time_window_range, prefix, paths, lag, n_animals, regions, processors=-1):
    """
    Get the range of window sizes for tenet files. 'DC AND 'MC' are the two prefixes implemented.
    Args:
        ts (roi, timepoints): Time series data.
        time_window_range (list): List of time window sizes.
        prefix (str): Prefix for the tenet files. 'dfc' for dynamic functional connectivity.
                   'mc' for meta-connectivity analysis.        
        lag (int): Lag value for the analysis.
        n_animals (int): Number of animals in the dataset.
        regions (list): List of regions in the dataset.
        processors (int): joblib. Number of processors to use for parallel computation.
    Returns:
        None
    """
    try:
        save_path = paths.get(prefix)
        if not save_path:
            raise ValueError(f"Invalid prefix '{prefix}'. Save path not found in paths dictionary.")
        # Run parallel dfc stream over window sizes
        start = time.time()
        Parallel(n_jobs=min(processors, len(time_window_range)))(
            delayed(compute_for_1_window)(ws, ts, prefix, lag, save_path) for ws in time_window_range
        )

        stop = time.time()
        print(f'{prefix} stream computation time {stop-start}')

        # Check for missing files and rerun if necessary
        missing_files = check_and_rerun_missing_files(
            save_path, prefix, time_window_range, lag, n_animals, regions
        )
        if missing_files:
            logging.warning(f"Missing files detected for {prefix}: {missing_files}")
            time_window_range = np.array(missing_files)
            # Rerun for missing files
            Parallel(n_jobs=min(processors, len(time_window_range)))(
                delayed(compute_for_1_window)(ws, ts, prefix, lag, save_path) for ws in time_window_range
            )
    except Exception as e:
        logging.error(f"Error occurred during {prefix} computation: {e}")
        raise
get_tenet4window_range(ts, time_window_range, prefix='dfc', paths=paths, lag=lag, n_animals=n_animals, regions=regions, processors=processors)
# %%
