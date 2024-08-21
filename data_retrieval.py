#Importing the packages

from netCDF4 import Dataset
import xarray as xr
from xhistogram.xarray import histogram
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.cm as cm


import keras
from keras.models import load_model

from scipy.stats import ks_2samp

# Set of global variables

path_asr_GLOBAL = '/home/shared/data/Wstd_asr_resampled_stdev30min_72lv_'
path_merra_GLOBAL = "/home/shared/data/Merra_input_asr_72lv_"
path_site_GLOBAL = '/home/shared/sites/df_'
modelpath_wnet_GLOBAL = "/home/shared/models/Wnet.h5"
path_wnet_prediction_GLOBAL = '/home/shared/model_results/'


# This set of functions is responsible for data preprocessing of our dataset

# ============== DATA RETRIEVAL ===================

def outlier(x):
    return abs((x-x.mean(dim='time')) / x.std(dim='time')) # causes some RuntimeWarnings (division by 0/nan)

def standardize(ds):
    i = 0
    # ['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL'] in that order??
    m = [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6]  #hardcoded from G5NR
    s = [30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5]
    for v in  ds.data_vars:
        ds[v] = (ds[v] - m[i])/s[i]
        i = i+1
    return ds

def get_data(site='', path_asr = path_asr_GLOBAL, path_merra = path_merra_GLOBAL, chunk_size=512*72):
    """
    This function is intended to capture reconstruct the G5NR-similar data with MERRA-2 and observations
    
    Input:
    path_asr: a string that contains a path to sigma_W observations
    path_merra: a string that contains a path to MERRA-2 reanalysis data
    site: a specific site on which information has been collected
    
    Output:
    X: xarray that contains all the information except for sigma_W
    y: xarray of sigma_W
    
    """
    
    # ================= preprocess the links ====================
    path_asr = path_asr + site + '.nc'
    path_merra = path_merra + site + '.nc'
    
    
    data_obs = xr.open_mfdataset(path_asr, parallel=True)
    data_merra = xr.open_mfdataset(path_merra, parallel=True, chunks={"time": 2560})

    # ================= process obs ====================
    data_obs = data_obs.where((data_obs != -9999.) and (data_obs < 15.))
    # some conditionals for if site is manus, twp, mao, or ena (we don't have this data)
    data_std = (data_obs.where(data_obs > 0.001)).groupby('time.month').map(outlier) # abs(anomaly/std)
    data_obs = data_obs.where(data_std < 2.5) # no outliers beyond 2 st dev?
    data_obs = data_obs.dropna('time', how='all', thresh=2) # keep only timesteps w/ >= 2 non-nan
    data_obs = data_obs.fillna(0) # fill missing vals w/ 0 (WHY THIS CHOICE?)
    
    # ================= process merra ==================
    data_merra = data_merra.resample(time="5min").interpolate("linear") # turn 3-hr intervals into 5 min ones
    data_merra = data_merra[['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']] # all this does is reorder vars
    
    # ================= align merra w/ obs ===================
    data_merra, data_obs = xr.align(data_merra, data_obs, exclude = {'height', 'lev'}) # (inner join) time steps of merra, obs

    # ================= prep model input X (standardize, add 4 surface vars) =================
    X = xr.map_blocks(standardize, data_merra, template=data_merra) # standardize (7 features, 10 vars)
    
    levs = X.coords['lev'].values
    num_levs = len(levs) # should be 72
    surface_vars = ['AIRD', 'KM', 'RI', 'QV']
    for sv in surface_vars: # get entire surface_var column (all time) for lev=71, remove level dimension becomes row
        sv_row = X[sv].sel(lev=[71]).squeeze() # lev 71 is 1 level above surface (lev 72)
        
        X_sv = sv_row
        for i in range(num_levs - 1): # append sv_row below X_sv to get 72 rows of sv_row
            X_sv = xr.concat([X_sv, sv_row], dim='lev')

        X[sv + "_sfc"] = X_sv.assign_coords(lev=levs)
    
    # ==================== clean up input X ========================
    X = X.unify_chunks() # chunk becomes (all timesteps, one level)
    
    X = X.to_array() # turn DataSet into DataArray
    # for above, data variables stacked (become 1st axis of new array), variables become coords
    
    X = X.rename({'variable':'feature'}) # optional, just using feature instead of variable
    X = X.stack(s=('time', 'lev')) # time and lev coords stacked into single coord
    X = X.squeeze() # removes length-1 axes (none in this case)
    X = X.transpose()
    X = X.chunk({'s': 72*1024}) # not sure about this chunk choice - could be hardware specific?
    
    # ==================== clean up target Y =======================
    y = data_obs['W_asr_std'] # get DataArray from DataSet
    y = y.stack(s=('time', 'height')) # all 72 levels for one time, and then next time
    y = y.chunk({'s': 72*1024})

    # manually triggers loading from disk/remote source to memory
    # might be necessary when working w/ many files on disk?
    return X.load(), y.load()


def get_site_df(site: str, path_site = path_site_GLOBAL):
    """
    This function puts the xarray data into the readable pandas form, by adding time, level information and deleting 0 rows
    so we can work with it in the reanalysis.
    
    Input:
    path_site: a string that contains a path to site information in the csv format, which is quicker to process
    site: a specific site on which information has been collected
    
    Output:
    df_X: pandas dataframe, which doesn't contain empty sigma_W values and includes time and level in the dataframe
    
    """
     # ================= preprocess the links ====================
    path_site = path_site + site + '.csv'
    
    if os.path.isfile(path_site):
        #We check for existence, so we don't do the same preprocessing twice
        df_X = pd.read_csv(path_site)
    else:
        X, Wobs = get_data(site=site) #We get the data from each site

        #Preprocessing
        df_X = pd.DataFrame(X, columns = X.feature) #We form a dataframe
        df_X['W_obs'] = Wobs
        
        #time manipulation
        df_X['time'] = X.time
        df_X['time'] = pd.to_datetime(df_X['time']) #Conversion to pandas-friendly time format
        df_X['lev'] = X.lev
        df_X = df_X[df_X['W_obs'] != 0].reset_index(drop=True)
        
        #We save the file
        df_X.to_csv(path_site, index=False)

    # ===== Update to the terminal =====
    print(f'Successfully retrieved data for the site {site}')
    return df_X

# ============ WNET processing ===================

def construct_wnet(site: str, modelpath_wnet = modelpath_wnet_GLOBAL, save_path = path_wnet_prediction_GLOBAL):
    """
    This function reconstructs the prediction made by Wnet and puts them into a dataframe
    
    Input: 
    save_path: a path where we would like to store the csv of results produced by wnet
    site: a site for which we construct the predictions of wnet 
    
    Output: 
    result_df: a dataframe of wnet predictions for levels and times
    
    """
    
    # ============= This part of code could be commented out with a stronger GPU ====================
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #This sets to be operating on CPU
    
    # ===============================================================================================
    
    #That's an unstable function, so check whether it runs
    try:
        wnet = load_model(modelpath_wnet, compile=False)
    except:
        print('Failed to load Wnet model, due to weak computational power')
    
    #Getting the information
    X, obs = get_data(site = site)
    
    wnet_preds = wnet.predict(X, batch_size=32768) #retrieve the predictions
    wnet_preds = wnet_preds.reshape(-1)
    
    #get the time and lev
    time_coords = X.coords['time'].values
    lev_coords = X.coords['lev'].values
    
    #we reconstruct the dataframe from a dictionary
    result_df = {'time': time_coords, 'lev': lev_coords, 'obs': obs.values, 'wnet_pred': wnet_preds}
    result_df = pd.DataFrame(result_df)
    result_df = result_df[result_df['obs'] != 0]
    
    #we save the output
    result_df.to_csv(f'{save_path}/results_{site}_filtered.csv', index=False)
    return result_df

def get_wnet(site: str, levels: list, path_wnet = path_wnet_prediction_GLOBAL):
    """
    This function exists to construct Wnet prediction for set of levels on a certain site
    
    Input:
    site: the site for which we want to collect information
    levels: the levels for which we want to get information
    
    Output:
    df_wnet: dataframe which includes time and level
    """
    # getting the information out 
    link = path_wnet + f'results_{site}_filtered.csv'
    
    if os.path.isfile(link):
        df_wnet = pd.read_csv(link)
        df_wnet = df_wnet[df_wnet['lev'].isin(levels)]

        # a tiny bit of whimsical editing (average out by hour and linear interpolation)
        df_wnet['time'] = pd.to_datetime(df_wnet['time'])
        df_wnet.set_index('time', inplace=True)
        
        df_wnet = df_wnet.drop(['lev'], axis = 1)
    else:
        df_wnet = construct_wnet(site)
        
        #We repeat the same procedure
        df_wnet = df_wnet[df_wnet['lev'].isin(levels)]

        # average out by hour and linear interpolation
        df_wnet['time'] = pd.to_datetime(df_wnet['time'])
        df_wnet.set_index('time', inplace=True)
        df_wnet = df_wnet.drop(['lev'], axis = 1)
        
    # ====== Update to the terminal ========
    print('Got Wnet data!')
    return df_wnet

# ============ DATAFRAMES ===================

#This is a function to downsample a dataframe
def downsample_function(df_X, final_length):
    """
    This function downsamples the dataframes so they become a fixed length
    
    Input:
    df_X: dataframe with N entries
    final_length: the length of the dataframe we would like to achieve in the end
    
    Output:
    df: dataframe of the given length
    
    """
    
    #Find the number of levels
    levels = df_X['lev'].unique().tolist() #We find the levels present in the dataframe
    num_of_samples = final_length//len(levels) #this is how many samples should be
    new_df = []
    for level in levels:
        df_lev = df_X[df_X['lev'] == level]
        df_sample = df_lev.sample(n = num_of_samples, replace=True) #Random Sampling without Replacement
        new_df.append(df_sample)
    df = pd.concat(new_df, ignore_index=True)
    return df

#Gets W by level
def level_W(sites, levels, downsample = False):
    """
    We get the combined dataframes from different sites merged by level groups
    
    Input:
    sites: this is a list of sites we concatenate the level-wise information by
    levels: [level_set_1, level_set_2, ..., level_set_n] - list of level sets
    
    Output:
    level_W: 2-D array of all the sigma_w's collected by level sets
    """
    
    # We identify length of the sample
    length_total = len(get_site_df('mao'))*4
    
    level_W = [np.array([]) for _ in range(len(levels))]
    for site in sites:
        df_X = get_site_df(site)
        if downsample and site not in ['lei', 'manus', 'lim']:
            df_X = downsample_function(df_X, length_total) #We downsample the dataframe
            
        i = 0 #We construct a counter
        for level_subset in levels: #We iterate 
            df_sub = df_X[df_X['lev'].isin(level_subset)] #We take the levelset
            w_lev = np.array(df_sub['W_obs']) #W of the observations
            level_W[i] = np.append(level_W[i], w_lev)
            i+=1

    return level_W

def season_W(sites, downsample = False):
    """
    We get the combined dataframes from different sites merged by seasons (fall, winter, spring, summer)
    
    Input:
    sites: this is a list of sites we concatenate the level-wise information by
    
    Output:
    season_W: 2-D array of all the sigma_w's collected by seasons
    """
    
    # We identify length of the sample
    length_total = len(get_site_df(site = 'mao'))*4 
    
    seasons = [1,2,3,4]
    season_W = [np.array([]) for _ in range(len(seasons))]
    for site in sites:
        df_X = get_site_df(site = site)
        df_X['season'] = pd.to_datetime(df_X['time']).dt.month%12 // 3 + 1 #Here we select the 
        if downsample and site not in ['lei', 'manus', 'lim']:
            df_X = downsample_function(df_X, length_total)
    
        i = 0 #initialize the counter
        for season in seasons:
            df_sub = df_X[df_X['season'] == season] #We take the levelset
            w_ses = np.array(df_sub['W_obs'])
            season_W[i] = np.append(season_W[i], w_ses)
            i+=1
    return season_W

def time_of_day_W(sites, downsample = False):
    """
    We get the combined dataframes from different sites merged by time of the day 
    (night, morning, afternoon, evening)
    
    Input:
    sites: this is a list of sites we concatenate the level-wise information by
    
    Output:
    sigma_W: 2-D array of all the sigma_w's collected by seasons
    """
    
    # We identify length of the sample
    length_total = len(get_site_df(site = 'mao'))*4
    
    # Define time bins for different parts of the day
    time_bins = [0, 6, 12, 18, 24]  # Hours: [0-6) (6-12) (12-18) (18-24)
    time_labels = [1, 2, 3, 4]  # Labels for each time bin
    time_W = [np.array([]) for _ in range(len(time_labels))]
    
    for site in sites:
        df_X = get_site_df(site = site)
        
        # Convert 'time' column to datetime and extract hour
        df_X['hour'] = pd.to_datetime(df_X['time']).dt.hour
        
        # Bin the hours into categories
        df_X['time_bin'] = pd.cut(df_X['hour'], bins=time_bins, labels=time_labels, right=False)
        
        if downsample and site not in ['lei', 'manus', 'lim']:
            df_X = downsample_function(df_X, length_total)
        
        # Process each time bin
        for i, label in enumerate(time_labels):
            df_sub = df_X[df_X['time_bin'] == label]  # Filter by time bin
            w_bin = np.array(df_sub['W_obs'])
            time_W[i] = np.append(time_W[i], w_bin)
    
    return time_W

