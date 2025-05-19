#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""
#%% Import libraries
from cProfile import label
from pathlib import Path
from tkinter import W
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import time
import pandas as pd
import sys
sys.path.append('../shared_code')
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from fun_loaddata import *
from fun_dfcspeed import *
from fun_utils import set_figure_params

from joblib import Parallel, delayed


#%% Figure parameters 
# =============================================================================
# Figure's parameters
# =============================================================================


save_fig = set_figure_params(savefig=True)
PROCESSORS = -1 # Use all available processors
#%%
# ------------------------Configuration------------------------
# Set the path to the root directory of your dataset
USE_EXTERNAL_DISK = True
ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien/') if USE_EXTERNAL_DISK \
        else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/')

PREPROCESS_DATA = ROOT / 'preprocess_data'
RESULTS_DIR = ROOT / Path('results')
SPEED_DIR = RESULTS_DIR / 'speed'

FIG_DIR = ROOT / 'fig'
SPEED_FIG_DIR = FIG_DIR / 'speed'
SPEED_FIG_DIR.mkdir(parents=True, exist_ok=True)   

# Parameters speed
WINDOW_PARAM = (5,100,1)
LAG=1
TAU=3

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"

#%%
#OLD REMOVE

#%%
# ------------------------ Load Data ------------------------

#Here we load the preprocessed cognitive and time- series data

cog_data = pd.read_csv(PREPROCESS_DATA / "cog_data_filtered.csv")
data = np.load(PREPROCESS_DATA / "ts_filtered_unstacked.npz", allow_pickle=True)
ts_filtered = data["ts"]

vel_data = np.load(SPEED_DIR / f'speed_dfc_{HASH_TAG}.npz', allow_pickle=True)
vel = vel_data['vel']
speed_median = vel_data['speed_median']

# ------------------------ Preprocessing ------------------------

wt_index          = cog_data['WT']
mut_index          = cog_data['Dp1Yey']

tau_array           = np.append(np.arange(0,TAU), TAU) 
lentau              = len(tau_array)

# short_mid = 10
# mid_long = 31

short_mid = 13
mid_long = 40
limits = (short_mid, mid_long)  # match your original: short/mid/long split

# #Some important variables
n_animals = len(ts_filtered)
regions = ts_filtered[0].shape[1]
#%%

time_windows_range = np.arange(WINDOW_PARAM[0],WINDOW_PARAM[1]+1,WINDOW_PARAM[2])
aux_timewr = time_windows_range*2

vel_label = ('%s-%ss (short)'%(aux_timewr[0], aux_timewr[limits[0]]),
             '%s-%ss (mid)'%(aux_timewr[limits[0]], aux_timewr[limits[1]]),
             '%s-%ss (long)'%(aux_timewr[limits[1]], aux_timewr[-1]))
# n_animals = int(np.sum(male_index))
#%%
pooled_vel = pool_vel_windows(vel, lentau, limits, strategy="drop")
# pooled_vel = pool_vel_windows(vel, lentau, limits, strategy="pad")

# Remove NaNs inside each pooled array — Case 1 (per-animal, per-window cleanup)
aux_short2m_list = [arr[~np.isnan(arr)] for arr in pooled_vel["short"]]
aux_mid2m_list   = [arr[~np.isnan(arr)] for arr in pooled_vel["mid"]]
aux_long2m_list  = [arr[~np.isnan(arr)] for arr in pooled_vel["long"]]

# Replace wp_list with cleaned lists
wp_list = (aux_short2m_list, aux_mid2m_list, aux_long2m_list)

#%%
#For the dfc speed distribution window oversampling, get a windows pooling


#Genotyped based
wp_wt = get_population_wpooling(wp_list, wt_index)
wp_mut = get_population_wpooling(wp_list, mut_index)

# =============================================================================

#%%
# Compute the normalization histogram of the speed for each animal

def compute_normalized_histograms(*all_speeds, bins=250, range_=(0, 1.2)):
    """
    Compute normalized histograms for multiple speed data arrays, normalized to the global max.

    Args:
        *all_speeds: Variable number of 1D arrays (e.g., WT, Mut, etc.)
        bins: Number of histogram bins.
        range_: Tuple specifying the histogram range.

    Returns:
        bin_centers, list_of_normalized_counts
    """
    # Compute raw histograms for each group
    counts_list = []
    bin_edges = None
    for speeds in all_speeds:
        counts, bin_edges = np.histogram(speeds, bins=bins, range=range_, density=False)
        counts_list.append(counts)
    # Normalize by global max
    global_max = max([counts.max() for counts in counts_list])
    norm_counts_list = [counts / global_max for counts in counts_list]
    # norm_counts_list = [counts / counts.max() for counts in counts_list]
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, norm_counts_list


# =============================================================================
# Plot windows pool
# =============================================================================
def plot_wpool(wp_var1, wp_var2, name_data = 'all'):
    plt.figure(1, figsize=(12,10))
    plt.clf()
    # vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')
    
    # wp_var1 = wp_wt
    # wp_var2 = wp_mut
    for i in range(3):
        #Normalize the histograms
        bin_centers, norm_counts_list = compute_normalized_histograms(
                np.array(wp_var1[i]), 
                np.array(wp_var2[i])
            )
        # Plot the Normalized histogram
        plt.subplot(3,2,2*i+1)
        if i==0:
            plt.title('%s %s'%(vel_label[i],name_data))
            plt.ylabel('Rel Freq (to global max)')
        else:
            plt.title('%s'%vel_label[i])
        plt.plot(bin_centers, norm_counts_list[0], label="WT", linestyle='-', alpha=0.9)
        plt.plot(bin_centers, norm_counts_list[1], label="Mut", linestyle='-', alpha=0.9)
        # plt.hist((wp_var1[i], wp_var2[i]),label=('wt', 'mut'), histtype='step',bins=150, density=True)
        
        plt.xlim(0,1.2)

        plt.subplot(3,2,2*i+2)
        if i==0:
            plt.title('%s %s'%(vel_label[i],name_data))
        else:
            plt.title('%s'%vel_label[i])
        # plt.hist((wp_var1[i], wp_var2[i]),label=('wt', 'mut'), histtype='step',bins=150, density=True)
        plt.plot(bin_centers, norm_counts_list[0], label="WT", linestyle='-', alpha=0.9)
        plt.plot(bin_centers, norm_counts_list[1], label="Mut", linestyle='-', alpha=0.9)
        # plt.hist((wp_var1[i], wp_var2[i]),label=('wt', 'mut'), histtype='step',bins=150, density=True, log=True)
        plt.xlim(0,1.2)
        plt.ylim(10e-5, 10e-1)
        plt.yscale('log')
    # plt.xlabel('Freq[v]')
    plt.xlabel("dFC Speed")
    plt.legend()
    plt.tight_layout()
    
    if save_fig ==True:
        plt.savefig(SPEED_FIG_DIR / f'hist_speed_{HASH_TAG}_{name_data}.png')
        plt.savefig(SPEED_FIG_DIR / f'hist_speed_{HASH_TAG}_{name_data}.pdf')

plot_wpool(wp_wt, wp_mut, name_data = 'wt_mut')
# plot_wpool(wp_wt_female, wp_mut_female, name_data = 'female')
# plot_wpool(wp_wt_male, wp_mut_male, name_data = 'male')
#%%
# =============================================================================
# Cumsum
# =============================================================================

a = 1-np.sort(wp_wt[2])

c = 1 - np.sort(wp_mut[2])
p = 1 * np.arange(len(a))/(len(a)-1)
q = 1 * np.arange(len(c))/(len(c)-1)

plt.figure(11)
plt.clf()
plt.title('Cumulative distribution function')

plt.plot(a,p, label='wt')
plt.plot(c,q, label='mut')
plt.xlabel('dFC Speed')
plt.ylabel('1 - Cumulative probability')
plt.legend()
# plt.xscale('log')
plt.yscale('log')
#%%

#%%# =============================================================================
# Save windows pooling data
# =============================================================================

# load_wpool = np.load(folder_results + 'speed/windowspooling_' + hash_parameters+'.npz', allow_pickle=True)

#%%
# =============================================================================
# Compute velocity statistics
# =============================================================================


bin_centers, norm_counts_list = compute_normalized_histograms(
    np.concatenate(wp_wt), 
    np.concatenate(wp_mut)
)
#%%
# Plot
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, norm_counts_list[0], label="WT", linestyle='-', alpha=0.9)
plt.plot(bin_centers, norm_counts_list[1], label="Mut", linestyle='-', alpha=0.9)
plt.xlabel("dFC Speed")
plt.ylabel("Relative Frequency (normalized to global max)")
plt.title("Histogram of dFC Speeds (WT vs Mut, shared normalization)")
plt.legend()
plt.yscale('log')
# plt.grid(True)
plt.tight_layout()
plt.show()


#%%
# Save the data in optimal format .npz 

np.savez(SPEED_DIR / f'windows_pooling_{HASH_TAG}.npz',
         wpool_wt       = wp_wt, 
         wpool_mut      = wp_mut, 
         )


#%%



# =============================================================================
# Compute and plot quantile range
# =============================================================================

def quantiles_across_groups(group_list, q_range):
    """
    Compute quantile ranges across multiple groups and window pools.

    Parameters:
    - group_list: array-like of shape (n_groups, n_wpools), each entry is a 1D array of speed values
    - q_range: array-like of quantiles to compute (e.g., np.linspace(0.01, 0.99, 99))

    Returns:
    - np.ndarray of shape (n_groups, len(q_range), n_wpools)
    """
    group_list = np.asarray(group_list, dtype=object)
    n_groups, n_wpools = group_list.shape
    n_quantiles = len(q_range)

    # Allocate output: shape (n_groups, n_wpools, n_quantiles)
    output = np.empty((n_groups, n_wpools, n_quantiles))

    for g in range(n_groups):
        for w in range(n_wpools):
            output[g, w, :] = np.quantile(group_list[g, w], q_range)

    return output

 #%%
 #distribution difference
def qq_diff(qq_data):
    qq_data = np.transpose(qq_data, (1, 0, 2))  # Shape: (n_groups, n_wpools, n_quantiles) 
    num_group = qq_data.shape[0] # Number of groups
    qq_diff =[] # Initialize list to store slopes
    for qq_aux in qq_data:
        diff_qq = np.squeeze(np.diff(qq_aux, axis=0))
        # print(diff_qq.shape)
        qq_diff.append(diff_qq)
    return qq_diff

# Slope difference
def qq_slope(qq_data):
    qq_data = np.transpose(qq_data, (1, 0, 2))  # Shape: (n_groups, n_wpools, n_quantiles) 
    num_group = qq_data.shape[0] # Number of groups
    qq_slope =[] # Initialize list to store slopes
    for qq_aux in qq_data:
        
        slope = np.diff(qq_aux, axis=1)
        diff_slope = np.squeeze(np.diff(slope, axis=0))
        # print(np.shape(diff_slope))

        qq_slope.append(diff_slope)
    return qq_slope

#%%
# ============================ Quantile per phenotype ==========================

#Parameters
q_range = np.linspace(0.01, 0.99,99)
label_gen = ('wt','mut')
label_vel = ('short', 'mid', 'long')
#%%
#quantile per genotype
wp_genotype = (wp_wt, wp_mut)  #window pooling for genotype

# quantile per group
# qq_genotype = quantile_per_group(wp_genotype, q_range)
qq_genotype = quantiles_across_groups(wp_genotype, q_range)
#%%
# Slope per group
slope_qq_gen = qq_slope(qq_genotype)

# Diff per group
diff_qq_gen = qq_diff(qq_genotype)

# Plots 
# Plot slopes
plt.figure(2, figsize=(12,5))
for vv in range(np.shape(slope_qq_gen)[0]):
    plt.plot(np.arange(len(np.diff(q_range)))/len(np.diff(q_range)), slope_qq_gen[vv],'.-',label=label_vel[vv])
plt.title(r"$\Delta$ Slope of Percentiles (Genotype)")
plt.xlabel("Percentiles")
plt.ylabel(r"$\Delta$ Slope (Mut-WT)")
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()


# Plot differences
plt.figure(3, figsize=(12,5))
for vv in range(np.shape(diff_qq_gen)[0]):
    plt.plot(q_range, diff_qq_gen[vv],'.-',label=label_vel[vv])
plt.title(r"$\Delta$ Difference of Percentiles (Genotype)")
plt.xlabel("Percentiles")
plt.ylabel(r"$\Delta$ Difference (Mut-WT)")
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

#QQ plot
plt.figure(4, figsize=(12,5))
plt.clf()
for vv in range(np.shape(qq_genotype)[1]):
    print(vv)
    plt.subplot(1,3,vv+1)
    plt.title('Q-Q plot %s'%label_vel[vv])
    plt.scatter(qq_genotype[0, vv], qq_genotype[1, vv], label='%s'%label_vel[vv], facecolors='none', edgecolors='C%s'%vv, s=40)
    plt.xlabel("WT Quantiles")
    plt.ylabel("Mut Quantiles")
    # plot a diagonal line
    plt.plot([0.1, 1], [0.1, 1], color='k', linestyle='--', alpha=0.5)
    plt.legend()
    
#%% Compute Bootstrap permutation

#%%
# # ================== Bootstrap permutation function ==========================


#%%

def _bootstrap_job(data_true, q_values, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(data_true)
    indices = np.random.randint(0, n, n)
    sampled = np.take(data_true, indices)
    return np.quantile(sampled, q_values)

def bootstrap_permutation_joblib(data_true, q_range, replicas=1000, n_jobs=-1, verbose=1):
    start = time.time()
    q_values = np.asarray(q_range)

    seeds = np.random.randint(0, 1_000_000, size=replicas)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_bootstrap_job)(data_true, q_values, seed) for seed in tqdm(seeds)
    )

    # Stack results: shape (replicas, len(q_range)) → transpose
    quantiles = np.stack(results, axis=0).T  # shape: (len(q_range), replicas)

    # Compute CI across replicas
    low_q, high_q = np.quantile(quantiles, [0.025, 0.975], axis=1)

    stop = time.time()
    print(f'Joblib bootstrap time: {round(stop - start, 2)} seconds')

    return low_q, high_q


def handler_bootstrap_permutation(wp_type, q_range, replicas=10, n_jobs=-1, bootstrap_fn=bootstrap_permutation_joblib):
    n_type = np.array(wp_type).shape[0]
    aux_qq_data = []
    for wp_ in tqdm(wp_type):
        n = wp_.shape[0]
        wp_boot = np.array([
            bootstrap_fn(wp_[xx], q_range, replicas, n_jobs=n_jobs, verbose=0) 
            for xx in range(n)
            ])
        aux_qq_data.append(wp_boot)
    return aux_qq_data

#%%

# # ================== Bootstrap permutation ==========================
# # Parameters
n_replicas = 5000

qq_gen_bootstrap = np.array(handler_bootstrap_permutation(wp_genotype,q_range, replicas=n_replicas))

#%%

def plot_bootstrap_ci_comparison(q_range, qq_data, qq_bootstrap, group_labels=('WT', 'Mut'), vel_labels=('short', 'mid', 'long'), title='Bootstrap CI per Window'):
    """
    Plot quantile curves and their confidence intervals for two groups.

    Parameters:
    - q_range: array of quantile values
    - qq_data: shape (2, len(q_range), 3), median quantiles for each group
    - qq_bootstrap: shape (2, 3, 2, len(q_range)), low/high CIs: [group, window, low/high, quantiles]
    """
    n_windows = len(vel_labels)
    
    plt.figure(figsize=(5 * n_windows, 5))
    for i in range(n_windows):
        plt.subplot(1, n_windows, i + 1)
        for g, label in enumerate(group_labels):
            # Plot median quantile line
            plt.plot(q_range, qq_data[g, :, i], label=f"{label} median", color=f"C{g}")
            
            # Plot CI band
            low = qq_bootstrap[g, i, 0]
            high = qq_bootstrap[g, i, 1]
            plt.fill_between(q_range, low, high, color=f"C{g}", alpha=0.2, label=f"{label} CI")

        plt.title(f"{vel_labels[i].capitalize()} Window")
        plt.xlabel("Quantile")
        plt.ylabel("dFC Speed")
        plt.grid(True)
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
qq_data = np.transpose(qq_genotype,(0,2,1)) # Shape: (n_groups, n_wpools, n_quantiles)
plot_bootstrap_ci_comparison(q_range, qq_data, qq_gen_bootstrap, group_labels=('WT', 'Mut'))

#%%
# =============================================================================
# Save data quantiles and bootstrap
# =============================================================================
data_quantile = {}
data_quantile['q_range'] = q_range
#quartile
data_quantile['qq_gen_all'] = qq_genotype
#quartile slope
data_quantile['qq_slope_gen_all'] = slope_qq_gen
#quartile bootstrap
data_quantile['qq_boot_gen'] = qq_gen_bootstrap
#quartile bootstrap slope
data_quantile['qq_slope_boot_gen'] = qq_gen_bootstrap
#quartile difference
data_quantile['qq_diff_gen'] = diff_qq_gen
#quartile difference slope
data_quantile['qq_slope_diff_gen'] = slope_qq_gen
#Save the data in optimal format .npz
np.savez(SPEED_DIR / f'qq_genotype_boot={n_replicas}.npz', **data_quantile)
# savemat('results/statistics/qq_animals_gen_phe_boot=%s.mat'%n_replicas, data_quantile)

#%%
