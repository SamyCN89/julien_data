import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
sys.path.append('../shared_code')
from fun_optimization import fast_corrcoef

def ts2dfc_stream(ts, window_size, lag=None, format_data='2D', method='pearson'):
    """
    Compute dynamic functional connectivity (DFC) stream using a sliding window approach.

    Parameters:
        ts (np.ndarray): Time series data (timepoints x regions).
        window_size (int): Size of the sliding window.
        lag (int): Step size between windows.
        format_data (str): '2D' (vectorized FC) or '3D' (FC matrices).
        method (str): Correlation method; only 'pearson' currently supported.

    Returns:
        np.ndarray: DFC stream, either in 2D (n_pairs x frames) or 3D (n_regions x n_regions x frames).
    """
    t_total, n = ts.shape
    lag = lag or window_size
    frames = (t_total - window_size) // lag + 1
    n_pairs = n * (n - 1) // 2

    if format_data == '2D':
        dfc_stream = np.empty((n_pairs, frames))
        tril_idx = np.tril_indices(n, k=-1)
    elif format_data == '3D':
        dfc_stream = np.empty((n, n, frames))

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + window_size
        window = ts[wstart:wstop, :]
        # fc = np.corrcoef(window, rowvar=False)
        fc = fast_corr(window, method=method)


        if format_data == '2D':
            dfc_stream[:, k] = fc[tril_idx]
        else:
            dfc_stream[:, :, k] = fc

    return dfc_stream


def dfc_speed(dfc_stream, vstep=1):
    """
    Calculate the speed of change in the dynamic functional connectivity stream.

    Parameters:
        dfc_stream (np.ndarray): DFC stream (2D or 3D).
        vstep (int): Temporal step size to compare FC patterns.

    Returns:
        float: Median speed of FC variation.
        np.ndarray: Time series of speed values.
    """
    if dfc_stream.ndim == 3:
        fc_stream = dfc_stream.reshape(dfc_stream.shape[0]*dfc_stream.shape[1], dfc_stream.shape[2])
    elif dfc_stream.ndim == 2:
        fc_stream = dfc_stream
    else:
        raise ValueError("Provide a valid 2D or 3D DFC stream.")

    nslices = fc_stream.shape[1]
    speeds = np.empty(nslices - vstep)

    for sp in range(nslices - vstep):
        fc1 = fc_stream[:, sp]
        fc2 = fc_stream[:, sp + vstep]
        covariance = np.cov(fc1, fc2)
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        speeds[sp] = 1 - correlation

    speed_median = np.median(speeds)
    return speed_median, speeds


def parallel_dfc_speed_oversampled_series(ts, window_parameter, lag=1, tau=3,
                                           min_tau_zero=False, get_speed_dist=False, method='pearson', n_jobs=-1):
    """
    Compute dFC speed for multiple window sizes and tau values, in parallel.

    Parameters:
        ts (np.ndarray): Time series (timepoints x regions).
        window_parameter (tuple): (min_win, max_win, step)
        lag (int): Lag between windows.
        tau (int): Max tau for oversampling.
        min_tau_zero (bool): Start tau at 0 or -tau.
        get_speed_dist (bool): Return full speed distributions.
        method (str): Correlation method.

    Returns:
        np.ndarray: Median speeds per window (and tau).
        list: Flattened speed distributions (if get_speed_dist=True).
    """
    min_tau = 0 if min_tau_zero else -tau
    win_min, win_max, win_step = window_parameter
    time_windows_range = np.arange(win_min, win_max + 1, win_step)
    tau_array = np.append(np.arange(min_tau, tau), tau)

    def compute_speed_for_window(tt):
        aux_dfc_stream = ts2dfc_stream(ts, tt, lag, format_data='2D', method=method)
        height_stripe = aux_dfc_stream.shape[1] - tt - tau
        speed_oversampl = [dfc_speed(aux_dfc_stream, vstep=tt + sp)[1][:height_stripe] for sp in tau_array]
        return np.median(speed_oversampl, axis=1), speed_oversampl if get_speed_dist else None

    results = Parallel(n_jobs=n_jobs)(delayed(compute_speed_for_window)(tt) for tt in tqdm(time_windows_range))
    speed_medians, speed_dists = zip(*results) if get_speed_dist else (zip(*results), None)

    if get_speed_dist:
        speed_dists = [item for sublist in speed_dists for item in sublist]
        return np.array(speed_medians), speed_dists
    else:
        return np.array(speed_medians)
