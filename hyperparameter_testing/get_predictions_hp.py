'''
This file retrieves predictions for specified wnet_prior versions
and stores the predictions as csv files
'''


# ============ variables to set =================
versions = [] # which versions in hp_tests directory to predict for
use_CPU = True
num_times = -1 # default is -1, which returns all times
filepath = "." # insert filepath for hp_tests (predictions stored in {filepath}/hp_tests/hp_v{version})
wnet_prior_filepath = "./models/Wnet_prior.h5" # insert correct filepath for Wnet_prior



# ======== imports ===========
import sys
import time
import datetime
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import h5py
import xarray as xr
from xhistogram.xarray import histogram

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.offsetbox import AnchoredText

from keras.models import load_model
from wnet_prior import get_dataset
import tensorflow as tf


# =========== constants ===================
SITES = ['asi', 'cor', 'nsa', 'sgp_cirrus', 'sgp_pbl', 'ena', 'lei', 'lim', 'manus', 'mao', 'pgh', 'sgp', 'twp']
OBS_WSTD_FP = "/home/shared/data2/Wstd_asr_resampled_stdev30min_72lv_%s.nc"
MERRA_INPUT_FP = "/home/shared/data2/Merra_input_asr_72lv_%s.nc"
G5NR_NEW_FP = "/home/shared/data2/RIPS_Wnet_data.nc"
G5NR_OLD_FP = "/home/shared/data/nature_test_data.nc"


# ============ set up environment ============
if use_CPU:
    print("Using CPU")
    try:
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except Exception as e:
        print(f"Error disabling GPU: {e}")


# ============== configure plotting =================
plt.switch_backend('agg')

plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title

xlab = r'$\log_{10}(\sigma_w ~\rm{m~s}^{-1})$'
ylab = r'$P(\log_{10}(\sigma_w))$'




# =========== methods ==================
def log10_scaler(ds): 
    '''used on observation dataarray and prediction np.ndarray'''
    func = lambda x: np.log10(x)
    return xr.apply_ufunc(func, ds, dask='parallelized')

def plot_3_pdf(wstd, prior, my_prior, ax, print_legend=True):
    '''Expects xr.DataArray (because this is what histogram takes in)'''
    axn = 0
    lw = 1.5
    
    # get bins
    bins = np.linspace(-3.0,1.5, 300) # -3, -2.98, ..., 1.5 (len=300)
    dx = (bins[1:]- bins[0:-1]) # list of bin lengths [0.015]
    bx = (bins[1:]+ bins[0:-1])/2

    # obs plot
    LOGwstd = log10_scaler(wstd) # get log_{10}(wstd)
    hs = histogram(LOGwstd, bins=[bins], block_size=10) # create histogram
    hobs =  hs.values # list of number of values in each bin
    fobs = hobs/dx/np.sum(hobs) # list [for bin j -> (# vals in bin j)/((bin j length) * (total # val))] 
    lab = 'G5NR Wstd'  
    obs_l = ax[axn].plot(bx, fobs, color='r', linestyle='-', linewidth=lw, label=lab)
    ax[axn].set_title(f'Wstd {lab}')
    ax[-1].plot(bx, fobs, color='r', linestyle='-', linewidth=lw, label=lab) 
    curves = obs_l
    axn += 1

    # og_wnet_prior plot 
    LOGprior = log10_scaler(prior)
    hs = histogram(LOGprior, bins=[bins], block_size=10)
    h =  hs.values       
    fx = h/dx/np.sum(h) 
    lab = 'Wnet-prior'
    prior_l = ax[axn].plot(bx, fx, color='b', linestyle='-', linewidth=lw, label=lab)
    ax[axn].set_title(f'{lab} Predictions')
    ax[-1].plot(bx, fx, color='b', linestyle='-', linewidth=lw, label=lab) 
    curves += prior_l
    axn += 1

    # my_wnet_prior plot 
    LOGmyprior =  log10_scaler(my_prior)
    hs = histogram(LOGmyprior, bins=[bins], block_size=10)
    h =  hs.values
    fx = h/dx/np.sum(h) 
    lab  = 'My Wnet_prior'     
    myprior_l = ax[axn].plot(bx, fx, color='k', linestyle='-', linewidth=lw, label=lab)
    ax[axn].set_title(f'{lab} Predictions')
    ax[-1].plot(bx, fx, color='k', linestyle='-', linewidth=lw, label=lab) 
    curves += myprior_l
    axn += 1
    
    ax[-1].set_title('Combined')
    if print_legend:
        labels = [c.get_label() for c in curves]
        ax[-1].legend(curves, labels, loc="upper right", frameon=False, framealpha=0., fontsize='small', borderpad=0)








# ================================================== main ===============================================================

# load data
dataset_us = get_dataset(filepaths=[G5NR_OLD_FP], name="test_US", location="US")
print(f"dataset has times: {dataset_us.all_times}")

X, y = dataset_us.get_data_test(num_times)
og_preds = pd.read_csv(f"{filepath}/hp_tests/hp_v1/preds_v1_-1times.csv")['og_pred'] # saves time


# load wnet_prior
wnet_prior = load_model(wnet_prior_filepath, compile=False) # offline mode


# load new wnet_prior and predict
for version in versions:
    print(f"\n\nPredicting for hp_v{version}")
    
    mymodelpath = f"{filepath}/hp_tests/hp_v{version}/wnet_prior_v{version}.hdf5"
    my_wnet_prior = load_model(mymodelpath, compile=False)

    t1 = time.time()
    my_preds = my_wnet_prior.predict(X, batch_size=2048)
    t2 = time.time()
    
    print(f"Wnet-prior prediction time on {X.shape[0]} samples: {t2-t1}")
    
    df = pd.DataFrame(
        {
            'my_pred': my_preds.squeeze(),
            'og_pred': og_preds.squeeze(),
            'g5nr_sim': y.values.squeeze(),
            'lev': X['lev'].values.squeeze(),
            'time': X['time'].values.squeeze(),
            'lat': X['lat'].values.squeeze(),
            'lon': X['lon'].values.squeeze()
        }
    )
    df.to_csv(f"{filepath}/hp_tests/hp_v{version}/preds_v{version}_{num_times}times.csv")

    # ============== make plots ========================
    fig, axes = plt.subplots(nrows=4, ncols=1) # currently plotting 3 things + overlayed
    fig.set_size_inches(9,8) # (width, height)
    axes = axes.flatten()
    
    plot_3_pdf(y, xr.DataArray(og_preds, name='og_preds'), xr.DataArray(my_preds, name='my_preds'), axes)
    
    plt.subplots_adjust(hspace=0.5)
    fig.text(0.5, 0.04, xlab, ha="center", va="center")
    fig.text(0.05, 0.5, ylab, ha="center", va="center", rotation=90)
    fig.suptitle(f'Wnet-prior vs Wnet-prior-{version} on G5NR', fontsize=20)
    plt.legend()
    
    fig.savefig(f'{filepath}/hp_tests/hp_v{version}/preds_v{version}_{num_times}times.png')
