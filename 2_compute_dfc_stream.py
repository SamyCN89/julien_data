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
data_timeseries = np.load(paths['sorted'] / Path("ts_filtered_unstacked.npz"), allow_pickle=True)
data_ts = data_timeseries['ts']
ts_data_filt = np.array([w for w in data_ts if w.shape[0] != 400])


# TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
# COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")
processors = -1

WINDOW_PARAM = (5,100,1)
LAG=lag=1
TAU=tau=5

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"
#%%
# ------------------------ Load Data ------------------------



print(f"Loaded {len(data_ts)} time series")
#%%
#Remove the animals with 400 points




prefix='dfc'
time_window_range = np.arange(WINDOW_PARAM[0],
                              WINDOW_PARAM[1] + 1,
                              WINDOW_PARAM[2])
n_animals, _, regions = ts_data_filt.shape
# ts = ts_data_filt
#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

#%% 
# ------------------------ Compute DFC ------------------------
import logging
import joblib 
logging.basicConfig(level=logging.INFO)

def validate_inputs(ts_data, window_size, lag):
    if not isinstance(ts_data, np.ndarray):
        raise TypeError("ts_data must be a numpy array.")
    if ts_data.ndim != 3:
        raise ValueError("ts_data must be a 3D array (n_animals, n_regions, n_timepoints).")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not isinstance(lag, int) or lag <= 0:
        raise ValueError("lag must be a positive integer.")
    
def handler_get_tenet(ts_data, prefix, window_size, lag, format_data='3D', save_path=None):
    """
    Generate temporal networks (dfc_stream, meta-connectivity) for time-series data.

    Parameters:
        ts_data (np.ndarray): 3D array (n_animals, n_regions, n_timepoints).
        window_size (int): Sliding window size.
        lag (int): Step size for the sliding window.
        format_data (str): '2D' for vectorized, '3D' for matrices.
        save_path (str): Directory to save results.

    Returns:
        np.ndarray: 4D array of DFC streams (n_animals, time_windows, roi, roi)
    """

    logger = logging.getLogger(__name__)
    n_animals, _, nodes = ts_data.shape

    # Define the full save path based on parameters and save_path folder
    file_path = make_file_path(save_path, prefix, window_size, lag, n_animals, nodes)
    logger.info(f'file path: {file_path}')

    #try loading from cache
    key = 'dfc_stream' if prefix == 'dfc' else prefix
    label = "dfc-stream" if prefix == "dfc" else "meta-connectivity"
    if file_path is not None and file_path.exists():
        logger.info(f"Loading from cache: {file_path}")
        try:
            return load_from_cache(file_path, key=key, label=label)
        except Exception as e:
            logger.error(f"Failed to load {label} (reason: {e}). Recomputing...")

    # Compute in parallel
    logger.info(f"Computing {prefix} (window_size={window_size}, lag={lag})...")
    results = np.array([ts2dfc_stream(
        ts_data[i], window_size, lag, format_data) 
        for i in tqdm(range(n_animals), desc=f'Computing {label}')])

    results = results.astype(np.float32)  # Convert to float32 for memory efficiency
    #Save results
    try:
        save2disk(file_path, prefix, key=results)
        logger.info(f'Saved results to {file_path}')
    except Exception as e:
        logger.error(f'Failed to save results: {e}')
    return results

def compute_for_1_window(ws, ts, prefix, lag, save_path):
    """
    Compute the analysis for a single window size.
    """ 
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Starting {prefix} computation for window_size={ws}")
        start = time.time()
        handler_get_tenet(
            ts,
            prefix=prefix,
            window_size=ws,
            lag=lag,
            save_path=save_path,
        )
        logger.info(f"Finished window_size={ws} in {time.time()-start:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during {prefix} computation for window_size={ws}: {e}")
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

        #set the processors
        processors = min(processors, joblib.cpu_count())
        logging.info(f'Starting analysis for {prefix}, n_jobs={processors}')

        start = time.time()
        Parallel(n_jobs=min(processors, len(time_window_range)))(
            delayed(compute_for_1_window)(ws, ts, prefix, lag, save_path) 
            for ws in tqdm(time_window_range, desc=f'Window sizes')
        )
        logging.info(f'{prefix} computation time {time.time()-start:.2f} seconds')

        # Handle missing files and rerun if necessary
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
        logger.error(f"Error occurred during {prefix} computation: {e}")
        raise
get_tenet4window_range(data_ts, time_window_range, prefix='dfc', paths=paths, lag=lag, n_animals=n_animals, regions=regions, processors=processors)
# %%
