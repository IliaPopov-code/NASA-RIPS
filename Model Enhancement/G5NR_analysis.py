from netCDF4 import Dataset
import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.offsetbox import AnchoredText

#from keras.models import load_model
from scipy.stats import ks_2samp
from xclim.analog import kolmogorov_smirnov
from PIL import Image

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from skimage.metrics import structural_similarity as ssim
import cartopy.crs as ccrs

def normalized_cross_correlation(image1, image2):
    """
    Compute the Normalized Cross-Correlation (NCC) between two images.
    
    :param image1: First image (grayscale).
    :param image2: Second image (grayscale).
    :return: Normalized cross-correlation coefficient.
    
    """

    # Mean and standard deviation of the images
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    std1 = np.std(image1)
    std2 = np.std(image2)

    # Subtract the mean from the images
    image1_normalized = (image1 - mean1) / std1
    image2_normalized = (image2 - mean2) / std2

    # Compute the normalized cross-correlation
    ncc = np.mean(image1_normalized * image2_normalized)
    
    return ncc

def array_to_image(array):
    """
    Converts a 2D array into a grayscale PIL image.
    :param array: 2D numpy array.
    :return: PIL Image.
    """
    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    image = Image.fromarray(array)
    
    # Convert to grayscale if necessary
    if image.mode != 'L':
        image = image.convert('L')
        
    return image

#path to dataset currently - '/home/shared/data/nature_test_data.nc'

def get_G5NR(path:str):
    """
    Collects the G5NR information.
    
    :path: path directing us to G5NR dataset.
    :return: tensor of wstd observations
    """
    dataset = Dataset(path, 'r')
    wstd_obs = dataset.variables['Wstd'][:]
    wstd_obs = np.array(wstd_obs[:, 1:, :, :])

    #We transpose the array in order to work with it
    wstd_obs = np.transpose(wstd_obs, (0, 2, 3, 1))
    return wstd_obs
def get_wnet_prior(path: str):
    """
    Collects the Wnet_prior prediction on G5NR.
    
    :path: path directing us to prediction results dataset.
    :return: tensor of wnet_prior predictions
    """
    #setting up the axis
    time = np.arange(5) #THIS COULD BE ALTERED, TIME IS NOT CONST
    latitude = np.arange(-90,90.5,0.5)
    longitude = np.arange(-180,180,0.5)
    level = np.arange(1,73)
    
    shape = (len(time), len(latitude), len(longitude), len(level))
    
    #loading the data
    df = pd.read_csv(path)
    tensor_prior = np.array(df.w_net_prior).reshape(shape)
    return tensor_prior

def get_wnet(path: str):
    """
    Collects the Wnet_prior prediction on G5NR.
    
    :path: path directing us to prediction results dataset.
    :return: tensor of wnet_prior predictions
    """
    #setting up the axis
    time = np.arange(5) #THIS COULD BE ALTERED, TIME IS NOT CONST
    latitude = np.arange(-90,90.5,0.5)
    longitude = np.arange(-180,180,0.5)
    level = np.arange(1,73)
    
    shape = (len(time), len(latitude), len(longitude), len(level))
    
    #loading the data
    df = pd.read_csv(path)
    tensor = np.array(df.w_net).reshape(shape)
    return tensor

#Useful bounds for plotting the heatmap:

#North America: -140, -40, 15, 65
#South America: -35, -80, 12, -55
#Africa: -20, 60, -35, 38
#Australia: 113, 154, -44, -10
#Asia: 65, 155, 0, 70 
#Europe: -25, 65, 35, 72

def heatmap_plot(w_array, level, bounds, noise_level = 0, res = '110m'):
    """
    This function constructs a plot for sigma_w on the global map
    
    :w_array: this is a four dimensional array (time, latitude, longitude, level)
    :level: the specific slice (level) we would like to visualize
    :bounds: a list of values which is a rectangular bound [lonW, lonE, latS, latN]
    """
    
    # We define the center of the box here as well its hyperparameters
    cLat = (bounds[2] + bounds[3]) / 2
    cLon = (bounds[0] + bounds[1]) / 2
    latitude = np.arange(-90,90.5,0.5)
    longitude = np.arange(-180,180,0.5)
    
    # Bounds of the colorbar
    vmin, vmax = 0, 0.5  # Limiting values
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Average out the tensor values
    w_array_ = np.mean(w_array, axis=0)
    df_obs = pd.DataFrame(w_array_[:,:,level], columns=longitude, index=latitude)
    
    #This part is used to create a visibility of noise
    df_obs += np.random.normal(0, noise_level, df_obs.shape)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    img = ax.imshow(df_obs, extent=(-180, 180, -90, 90), origin='lower', norm=norm, cmap='turbo', transform=ccrs.PlateCarree())
    
    #We add the coastlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=1, color='black', alpha=1, linestyle='--')
    ax.set_title(r'$\bar{\sigma}_{W}$ Heatmap Observed' + f' in {region} at level {level}', fontsize = 20)
    ax.set_extent(bounds, crs=projPC)  # Set plot extent

    ax.set_xlabel('Longitude', fontsize = 14)
    ax.set_ylabel('Latitude', fontsize = 14)

    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label(r'Average Standard Deviation $\sigma_{W}$ [$\frac{m}{s}$]', fontsize= 20)

    plt.tight_layout()
    plt.show()