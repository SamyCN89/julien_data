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
import sys
sys.path.append('../shared_code')

import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from fun_loaddata import *  # Import only needed functions
from fun_dfcspeed import parallel_dfc_speed_oversampled_series
from fun_utils import set_figure_params, get_root_path, get_paths
from tqdm import tqdm



#%% Define paths, folders and hash
# ------------------------ Configuration ------------------------

USE_EXTERNAL_DISK = True
ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien/') if USE_EXTERNAL_DISK \
        else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/')
RESULTS_DIR = ROOT / Path('results')
SPEED_DIR = RESULTS_DIR / 'speed'
SPEED_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESS_DATA = ROOT / 'preprocess_data'
TS_FILE = PREPROCESS_DATA / Path("ts_filtered_unstacked.npz")
COG_FILE = PREPROCESS_DATA / Path("cog_data_filtered.csv")

SAVE_DATA = True

WINDOW_PARAM = (5,100,1)
LAG=1
TAU=3

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"

#%%
# ------------------------ Load Data ------------------------

ts_data = np.load(TS_FILE, allow_pickle=True)['ts']
cog_data = pd.read_csv(COG_FILE)


print(f"Loaded {len(ts_data)} time series")
print(f"Loaded cognitive data for {len(cog_data)} animals")

assert len(ts_data) == len(cog_data), "Mismatch between time series and cognitive data entries."
#%%

# # =============================================================================
# # Plot cognitive data scores
# # =============================================================================
# aux_data_type = 2

# cog_wt_aux = (male_wt_data, female_wt_data, wt_data)[aux_data_type]
# cog_dki_aux =(male_dki_data, female_dki_data, dki_data)[aux_data_type]

# aux_l1 = ('Male','Female','All')
# cog_aux_labels = ['%s WT 2M'%(aux_l1[aux_data_type]), '%s WT 4M'%(aux_l1[aux_data_type]), '%s dKI 2M'%(aux_l1[aux_data_type]), '%s dKI 4M'%(aux_l1[aux_data_type])]

# #plot the aux_data_type
# plt.figure(1,figsize=(10, 6.5))
# plt.clf()

# plt.subplot(211)
# plt.title('Distribution of OiP scores for %s'%aux_l1[aux_data_type])
# plt.violinplot((cog_wt_aux['OiP_2M'], cog_wt_aux['OiP_4M'], cog_dki_aux['OiP_2M'], cog_dki_aux['OiP_4M']))#,cmap='C1')
# plt.xticks([1, 2, 3, 4], cog_aux_labels, fontsize=12)
# plt.axhline(0,c='k')
# plt.axhline(0.2,c='k',ls='--')
# plt.axhline(-0.2,c='k',ls='--')
# plt.ylabel('OiP score')

# plt.subplot(212)
# plt.title('Distribution of RO24h for %s'%aux_l1[aux_data_type])
# plt.violinplot((cog_wt_aux['RO24h_2M'], cog_wt_aux['RO24h_4M'], cog_dki_aux['RO24h_2M'], cog_dki_aux['RO24h_4M']))
# plt.xticks([1, 2, 3, 4], cog_aux_labels, fontsize=12)
# plt.axhline(0,c='k')
# plt.axhline(0.2,c='k',ls='--')
# plt.axhline(-0.2,c='k',ls='--')
# plt.ylabel('RO24h score')
# plt.tight_layout()
# if save_fig==True:
#     plt.savefig('fig/cog_data/oip_ro24h_%s_wt_dki.png'%aux_l1[aux_data_type])
#     plt.savefig('fig/cog_data/oip_ro24h_%s_wt_dki.pdf'%aux_l1[aux_data_type])


#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

#%% 
# ------------------------ Compute Speed ------------------------

vel_list = []
speed_medians = []

print("Starting dFC speed computation...")
start_time = time.time()

for ts in tqdm(ts_data, desc="Computing dFC speed"):
    median_speeds, speed_distribution = parallel_dfc_speed_oversampled_series(
        ts, WINDOW_PARAM, lag=LAG, tau=TAU, min_tau_zero=True, get_speed_dist=True
    )
    vel_list.append(speed_distribution)
    speed_medians.append(median_speeds)

elapsed_time = time.time() - start_time
print(f"Speed computation completed in {elapsed_time:.2f} seconds")

#%%
# ------------------------ Save Results ------------------------

if SAVE_DATA:
    vel_array = np.array(vel_list, dtype=object)
    medians_array = np.array(speed_medians)

    np.savez(
        SPEED_DIR / f"speed_dfc_{HASH_TAG}.npz",
        vel=vel_array,
        speed_median=medians_array,
    )
    print(f"Saved speed data to: {SPEED_DIR}")



# %%
