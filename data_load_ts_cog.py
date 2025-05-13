#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
#%%
import numpy as np
import os
import pandas as pd
import pickle
from pathlib import Path

import sys
sys.path.append('../shared_code')

from fun_loaddata import extract_hash_numbers
from fun_utils import filename_sort_mat, load_matdata, classify_phenotypes, make_combination_masks, make_masks
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import re
#%%

# =============================================================================
# This code loads the time series data and cognitive data from files
# =============================================================================

external_disk = False
if (external_disk==True):
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
else:    
    root = Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/')


# Replace os.listdir with pathlib's Path.iterdir
mat_files = [file.name for file in (root / 'time_courses').iterdir() if file.is_file()]
mat_files.sort(reverse=True)  # Sort in reverse order
def load_matdata_any_shape(folder_data, specific_folder, files_name):
    """
    Load time series from .mat files regardless of shape consistency.

    Returns
    -------
    ts_list : list of np.ndarray
        A list where each element is a loaded time series array (of shape [regions, timepoints]).
    shapes  : list of tuple
        The shapes of each loaded array.
    files   : list of str
        Filenames corresponding to the data.
    """
    ts_list = []
    shapes = []
    successful_files = []

    hash_dir = Path(folder_data) / specific_folder

    for file_name in files_name:
        file_path = hash_dir / file_name
        try:
            data = loadmat(file_path)['tc']
            ts_list.append(data)
            shapes.append(data.shape)
            successful_files.append(file_name)
            print(f"Loaded {file_name}: shape {data.shape}")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return ts_list, shapes, successful_files
#%% Load time series data
# =============================================================================
# Load time series data from .mat files
# =============================================================================
ts_list, ts_shapes, loaded_files = load_matdata_any_shape(root, 'time_courses', mat_files)
# Check if all loaded files have the same shape
if len(set(ts_shapes)) > 1:
    print("Warning: Not all loaded files have the same shape.")
#%%    
# Extract info using regex
cleaned = []
for name in loaded_files:
    # match = re.match(r"tc_Coimagine_(\w+)_0?(\d+)_\d+_seeds\.mat", name)
    match = re.match(r"tc_Coimagine_(\w+)_(\d+)_\d+_seeds\.mat", name)
    if match:
        group = match.group(1)
        id_ = match.group(2)
        cleaned.append((name, group, id_))
    else:
        print(f" No match for: {name}")

_, name_mouse, number_mouse = np.array(cleaned).T

ts_sorted_mouse = name_mouse + '_' + number_mouse
#%% Load cog data and intersect if there is in 2M and 4M
# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
cog_data_df     = pd.read_excel('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/mice_groups_comp_index.xlsx', sheet_name='mice_groups_comp_index')
region_labels        = np.loadtxt('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/all_ROI_coimagine.txt', dtype=str).tolist()
region_labels_clean = [label.replace("Both_", "") for label in region_labels]
#%%
# No cleaning needed
mouse_ids_ts = list(ts_sorted_mouse)
cog_data_df['mouse'] = cog_data_df['mouse'].astype(str)

# Match time series to cognitive data
valid_ids = set(cog_data_df['mouse'])
mouse_ids_ts_filtered = [m for m in mouse_ids_ts if m in valid_ids]
cog_data_sorted = cog_data_df.set_index('mouse').loc[mouse_ids_ts_filtered].reset_index()

#%%
# Check if any mouse IDs are missing from the cognitive data
missing = [m for m in mouse_ids_ts if m not in cog_data_df['mouse'].values]
if missing:
    print(" Missing from cognitive data:", missing)
else:
    print(f" Matched {len(mouse_ids_ts_filtered)} time series entries.")
#%%
# Only keep time series where the mouse is in cog_data_sorted
valid_mouse_ids = set(cog_data_sorted['mouse'])

# Filter ts_list and ts_sorted_mouse in the same order
ts_filtered = [ts for ts, mouse in zip(ts_list, ts_sorted_mouse) if mouse in valid_mouse_ids]
ts_filtered_names = [mouse for mouse in ts_sorted_mouse if mouse in valid_mouse_ids]

# Identify mice to exclude
excluded_mice = [mouse for mouse in ts_sorted_mouse if mouse not in cog_data_df['mouse'].values]

# Remove excluded mice from cognitive data and time series
cog_data_filtered = cog_data_sorted[~cog_data_sorted['mouse'].isin(excluded_mice)].reset_index(drop=True)
ts_filtered = [ts for ts, mouse in zip(ts_filtered, ts_filtered_names) if mouse not in excluded_mice]
ts_filtered_names = [mouse for mouse in ts_filtered_names if mouse not in excluded_mice]

print(f"Excluded {len(excluded_mice)} mice. Remaining mice: {len(ts_filtered_names)}")
#%%
cog_data_filtered["genotype"] = cog_data_filtered["grp"].str.split("_").str[0]
cog_data_filtered["treatment"] = cog_data_filtered["grp"].str.split("_").str[1]

#Hot encoding genotype and treatment
encoded_genotype = pd.get_dummies(cog_data_filtered["genotype"])
encoded_treatment = pd.get_dummies(cog_data_filtered["treatment"])
cog_data_filtered = pd.concat([cog_data_filtered, encoded_genotype , encoded_treatment], axis=1)

#%%
assert len(ts_filtered) == len(cog_data_sorted), "Mismatch in time series and cognitive data counts"

if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
    ts_array = np.stack(ts_filtered)
    print("Time series successfully stacked:", ts_array.shape)
else:
    print("Time series shapes are inconsistent — can't stack.")

#%%

#     by_col='Sexe',
#     primary_levels=genotypes_test,
#     by_levels=sexes_test,
#     is_2month_old=is_2month_old
# )

# mask_groups_per_sex = (mask_combo_oip, mask_combo_nor, mask_combo_gen)
# label_variables_per_sex = (label_combo_oip, label_combo_nor, label_combo_gen)
# #%% Save results
# # =============================================================================
# # Save results
# # =============================================================================
# #Cognitive data
# cog_data_filtered.to_csv(path_sorted / 'cog_data_sorted_2m4m.csv', index=False)

# #time series plus metadata
# If time series are all same shape and stackable:
if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
    ts_array = np.stack(ts_filtered)
    np.savez("ts_filtered.npz", ts=ts_array)
    print("Saved: ts_filtered.npz")
else:
    # If shapes differ, save as object array
    np.savez("ts_filtered_unstacked.npz", ts=np.array(ts_filtered, dtype=object))
    print("Saved unstacked: ts_filtered_unstacked.npz (inconsistent shapes)")

cog_data_filtered.to_csv("cog_data_filtered.csv", index=False)
print("Saved: cog_data_filtered.csv")


# np.savez(path_sorted / 'ts_and_meta_2m4m.npz',
#          ts=ts,  
#          n_animals=n_animals, 
#     total_tp=total_tp, 
#     regions=regions, 
#     is_2month_old=is_2month_old,
#     anat_labels=anat_labels,
#     )

# #mask and labels for groups
# with open(path_sorted / "grouping_data_oip.pkl", "wb") as f:
#     pickle.dump((mask_groups, label_variables), f)

# with open(path_sorted / "grouping_data_per_sex(gen_phen).pkl", "wb") as f:
#     pickle.dump((mask_groups_per_sex, label_variables_per_sex), f)

# #%% Load pre-process data
# # =============================================================================
# # # Load result
# # =============================================================================

# cog_data_filtered = pd.read_csv(path_sorted / 'cog_data_sorted_2m4m.csv')
# data_ts = np.load(path_sorted / 'ts_and_meta_2m4m.npz')

# ts=data_ts['ts']
# n_animals = data_ts['n_animals']
# total_tp = data_ts['total_tp']
# regions = data_ts['regions']
# is_2month_old = data_ts['is_2month_old']
# anat_labels= data_ts['anat_labels']

# %%
