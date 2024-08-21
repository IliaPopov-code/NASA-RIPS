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

from data_retrieval import get_site_df, level_W, season_W, time_of_day_W

#Here we define globals
all_sites = ['asi', 'cor', 'lei', 'manus', 'mao', 'lim', 'pgh', 'sgp_pbl', 'sgp_cirrus',  'twp', 'ena', 'nsa']

def make_pdf(array, bins, dx):
    """
    This is a constructor of a piecewise pdf, using the histogram architecture
    
    Input:
    array: a list of values we want to make a pdf out of
    bins: bins of the histogram
    dx: the scaling constant
    
    Output:
    pdf: the points on the pdf
    """
    W = np.log10(np.array(array))
    W = W[~np.isnan(W)]
    hs, _ = np.histogram(W, bins=bins)
    pdf = hs / (np.sum(hs) * dx)
    return pdf


def create_sampling_plot(statistic_array, lower_bound, higher_bound, sampling_times, y_label, title):
    plt.figure(figsize=(15, 10))
    
    #We transpose the information
    df_plot = pd.DataFrame(statistic_array, columns=np.arange(lower_bound, higher_bound), index = sampling_times)
    df_plot = df_plot.T

    palette = sns.color_palette("husl", n_colors=len(df_plot.columns)) #We define a palette
    i = 0 #we define the counter
    for site in sampling_times:
        plt.plot(df_plot.index, df_plot[site], label=site, color = palette[i], linestyle='-', linewidth=6, zorder=1)
        plt.scatter(df_plot.index, df_plot[site],color = palette[i], edgecolor='black', s=75, alpha=0.75, zorder=2)
        i+=1
    plt.xlabel('Level', fontsize = 40)
    plt.ylabel(y_label, fontsize = 40) #Wasserstein distance values
    plt.title(title, fontsize = 45)
    plt.legend(fontsize = 25)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True,linestyle='--', linewidth=0.7)
    return None

def plot_sampling_statistic(site: str):
    """
    This function exists to reconstruct the sampling statistic for a site and then plot the results
    Input:
    site: the site from which we collect information
    
    Output:
    KS_statistic_array: Array of KS-statistics for each level
    percentage_missing_array: Array of missing values per level
    """
    
    bins = np.linspace(-3, 1.5, 300)  # pdf bounds
    dx = (bins[1:] - bins[:-1])
    bx = (bins[1:] + bins[:-1]) / 2
    
    df_X = get_site_df(site) #we get the dataframe corresponding to the site

    #We create bounds for the levels we investigate
    lower_bound = int(min(df_X['lev']))
    higher_bound = int(max(df_X['lev']))

    sampling_times = [30, 60, 90, 120] #Those are the times we will be sampling at
    
    # We define arrays, which 
    KS_statistic_array = []
    percentage_missing_array = []
    
    for sampling_time in sampling_times:
        KS = []
        percentage = []
        for level in range(lower_bound, higher_bound):
            df_lev = df_X[df_X['lev'] == level] # we keep only one level
            df_lev = df_lev.drop(['lev'], axis = 1) # we drop the level column
            w = np.array(df_lev.W_obs) #
            df_lev['time'] = pd.to_datetime(df_lev['time'])

            # Set the 'time' column as the index
            df_lev.set_index('time', inplace=True)
            
            #We resample by the median
            df_lev = df_lev.resample(f'{sampling_time}T').median()
            
            #We find which rows have missing values
            rows_with_missing = df_lev.isnull().any(axis=1)
            num_rows_with_missing = rows_with_missing.sum()
            
            #We calculate the missing percentage
            percentage_missing = (num_rows_with_missing / len(df_lev)) * 100
            percentage.append(percentage_missing)

            #We drop the NaNs in order to not run into the errors
            df_lev = df_lev.dropna()
            
            #We reconstruct the pdfs
            pdf_obs = make_pdf(w, bins, dx)
            pdf_new = make_pdf(np.array(df_lev.W_obs), bins, dx)

            ks = ks_2samp(pdf_new, pdf_obs)[0]
            KS.append(ks)

        KS_statistic_array.append(KS)
        percentage_missing_array.append(percentage)

    #Here we begin plotting
    create_sampling_plot(KS_statistic_array, lower_bound, higher_bound, sampling_times, "KS statistic",
                         f"KS statistic on {site} site for sampling times")
    create_sampling_plot(percentage_missing_array, lower_bound, higher_bound, sampling_times, "Percentage missing",
                         f" Percentage missing for {site} site for sampling times")
    return None

def generate_blue_colors(N: int):
    """
    This function generates an array of blue colors
    
    Input:
    N: number of colors we would like to generate
    Output:
    hex_colors: list of HEX colors we wanted to generate
    
    """
    # Define the start and end colors (dark blue to light blue)
    start_color = np.array([0, 0, 0.5])  # Dark blue
    end_color = np.array([0.678, 0.847, 0.902])  # Light blue (Alice Blue)
    
    # Generate an array of N evenly spaced numbers between 0 and 1
    gradient = np.linspace(0, 1, N)
    
    # Interpolate between start_color and end_color
    colors = [start_color + (end_color - start_color) * g for g in gradient]
    
    # Convert the list of colors to hex
    hex_colors = ['#' + ''.join(f'{int(c*255):02X}' for c in color) for color in colors]
    
    return hex_colors


def histogram(sites, split_const = 0.8, save_file = False):
    """
    This function takes in the list of sites for which we want to construct the histogram
    
    Input:
    sites: list of sites, which we want to include into the histogram
    split_const: the train-test split according to the paper
    
    Output:
    None
    """
    lengths = [] #an array of lengths of training datasets
    
    for site in sites:
        df_X = get_site_df(site) 
        if site in ['mao', 'lei', 'man', 'lim']:
            lengths.append(int(split_const*len(df_X))*4)
        else:
            lengths.append(int(split_const*len(df_X)))
    colors = generate_blue_colors(len(sites))
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(sites, lengths, color= colors, edgecolor='black')
    plt.xlabel('Site', fontsize = 16)
    plt.ylabel('Length of DataFrame', fontsize = 16)
    plt.title('Training data on sites (adjusted accordingly to the paper)', fontsize = 16)
    plt.xticks(rotation=45, fontsize = 16)  # Rotate site names for better readability
    plt.yticks(fontsize = 16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    #If we want to save the histogram plot
    if save_file:
        plt.savefig('./HISTOGRAM.png', dpi = 300)
    plt.show()

def make_pdf_plots(array_list, labels, title = r'PDFs of $\sigma_W$ distributions on Observational data', 
               bound_low = -3, bound_high = 1.5, save = False, palette = 'jet'):
    
    """
    This function is intended for construction of pdf-plots of sigma_W.
    
    Input: 
    array_list: list of different arrays we want to turn into pdfs [array_1, array_2, ..., array_n]
    labels: for each of the array lists, we use a corresponding label
    
    Output:
    None
    """
    
    plt.figure(figsize=(12, 8))
    bins = np.linspace(bound_low, bound_high, 300)  # pdf bounds
    dx = (bins[1:] - bins[:-1])
    bx = (bins[1:] + bins[:-1]) / 2
    
    #Different pallets - viridis, husl, turbo, Spectral
    if palette != 'jet':
        pallete = sns.color_palette(palette, n_colors=len(array_list))
    
    #Specifically for jet color pallete
    else:
        values = np.linspace(0, 1, len(array_list))
        pallete = cm.jet(values)
    
    for i in range(len(array_list)):
        pdf = make_pdf(array_list[i], bins, dx)
        plt.plot(bx, pdf, label=r'$\sigma_W$ ' + labels[i], color = pallete[i], linewidth=3)
    plt.title(title, fontsize=20)
    plt.xlabel(r'$log_{10}(\sigma_W \ m \ s^{-1})$', fontsize=25)
    plt.ylabel(r'$\frac{dP(\sigma_W)}{d\log(\sigma_W)}$', fontsize=30)
    plt.legend(fontsize = 15)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    #Saving
    if save:
        plt.savefig('./pdfs.png', dpi = 300)
    plt.show()
    
def plot_sites(sites: list):
    """
    This function is used to create a pdf-plot of all the sites in the same place
    
    Input:
    sites: list of sites, which we want to include into the plot
    
    Output:
    None
    
    """
    W_list = []
    #We iterate through sites
    for site in sites:
        W = np.array(get_site_df(site).W_obs)
        W_list.append(W)
    make_pdf_plots(W_list, sites)
    

def EDA_plots_wrapper(key: str, sites: list):
    """
    This function exists for quick plotting of already existing plots for Obs data across space and time
    
    Input:
    key: There are several keys - level, season, day
    level - plot is by level groups
    season - plot is by season
    day - plot is by the time of day
    sites: the sites for which we are constructing the plot
    
    """
    
    levels = [[53,54,55,56], [57,58,59,60], [61,62,63,64], [65,66,67,68], [69,70,71,72]]
    labels_lev = ['Levels 53-56', 'Levels 57-60', 'Levels 61-64', 'Levels 65-68', 'Levels 69-72']
    labels_season = ['Winter', 'Spring', 'Summer', 'Fall']
    labels_time = ['Night', 'Morning', 'Afternoon', 'Evening']
    
    if key == 'level':
        W = level_W(sites, levels, downsample = False)
        make_pdf_plots(W, labels_lev, palette = 'viridis')
    elif key == 'season':
        W_season = season_W(sites, downsample = False)
        make_pdf_plots(W_season, labels_season, palette = 'viridis')
    elif key == 'day':
        W_time = time_of_day_W(sites, downsample = False)
        make_pdf_plots(W_time, labels_time, palette = 'viridis') #This is the proof that temporal is important
    else:
        print('Input a proper key')
    return None