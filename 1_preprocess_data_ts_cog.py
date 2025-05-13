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

# ------------------------Configuration------------------------
# Set the path to the root directory of your dataset
USE_EXTERNAL_DISK = True
ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien/') if USE_EXTERNAL_DISK \
        else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/')

TS_FOLDER = ROOT / 'time_courses'
COG_XLSX = ROOT / 'mice_groups_comp_index.xlsx'
REGION_LABELS = ROOT / 'all_ROI_coimagine.txt'
PREPROCESS_DATA = ROOT / 'preprocess_data'
PREPROCESS_DATA.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

#%%
def load_mat_timeseries(folder: Path) -> tuple:
    """
    Load time series from .mat files regardless of shape consistency.

    Returns
    -------
    ts_list : list of tuple
        A list where each element is a loaded time series array (of shape [regions, timepoints]).
    shapes  : list of tuple
        The shapes of each loaded array.
    files   : list of str
        Filenames corresponding to the data.
    """
    mat_files = sorted([f.name for f in folder.iterdir() if f.is_file()], reverse=True)
    ts_list, shapes, names = [], [], []
    for fname in mat_files:
        try:
            data = loadmat(folder / fname)['tc']
            ts_list.append(data)
            shapes.append(data.shape)
            names.append(fname)
            print(f"Loaded {fname}: shape {data.shape}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return ts_list, shapes, names

def extract_mouse_ids(filenames: list) -> list:
    cleaned = []
    for name in filenames:
        match = re.match(r"tc_Coimagine_(.+?)_(\d+)_\d+_seeds\.mat", name)
        if match:
            cleaned.append(f"{match.group(1)}_{match.group(2)}")
        else:
            print(f"Warning: No match for {name}")
    return cleaned

def main():
    # Load all time series data
    ts_list, ts_shapes, loaded_files = load_mat_timeseries(TS_FOLDER)
    # Check if all loaded files have the same shape
    if len(set(ts_shapes)) > 1:
        print("Warning: Not all loaded files have the same shape.")

    ts_ids = extract_mouse_ids(loaded_files)

    # =============================================================================
    # Load cognitive data from .xlsx document
    # =============================================================================
    #Load cognitive data
    cog_data     = pd.read_excel(COG_XLSX, sheet_name='mice_groups_comp_index')
    cog_data['mouse'] = cog_data['mouse'].astype(str)

    region_labels        = np.loadtxt(REGION_LABELS, dtype=str).tolist()
    region_labels_clean = [label.replace("Both_", "") for label in region_labels]

    matched_ids = [mid for mid in ts_ids if mid in cog_data['mouse'].values]
    # Filter cognitive data to include only matched mouse IDs and sorted by mouse IDs
    cog_data_filtered = cog_data.set_index('mouse').loc[matched_ids].reset_index()

    #List of time series that match the mouse IDs in the cognitive data, preserving the order
    ts_filtered = [ts for ts, id_ in zip(ts_list, ts_ids) if id_ in matched_ids]

    if len(ts_filtered) != len(cog_data_filtered):
        raise ValueError("Mismatch in time series and cognitive data entries.")

    # Extract group features
    split_grp = cog_data_filtered["grp"].str.split("_", expand=True)
    cog_data_filtered["genotype"] = split_grp[0]
    cog_data_filtered["treatment"] = split_grp[1]
    cog_data_filtered = pd.concat([cog_data_filtered,
                                    pd.get_dummies(split_grp[0]),
                                    pd.get_dummies(split_grp[1])], axis=1)

    # Save processed data
    if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
        ts_array = np.stack(ts_filtered)
        np.savez(PREPROCESS_DATA / "ts_filtered.npz", ts=ts_array)
        print(f"Saved: ts_filtered.npz with shape {ts_array.shape}")
    else:
        np.savez(PREPROCESS_DATA / "ts_filtered_unstacked.npz", ts=np.array(ts_filtered, dtype=object))
        print("Saved: ts_filtered_unstacked.npz")

    cog_data_filtered.to_csv(PREPROCESS_DATA / "cog_data_filtered.csv", index=False)
    print("Saved: cog_data_filtered.csv")

if __name__ == "__main__":
    main()
