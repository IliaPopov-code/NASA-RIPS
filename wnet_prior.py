'''
This file contains the get_dataset and DaskGenerator classes,
which streamlines the data and batch retrieving processes for wnet_prior training. 
'''

import os
import glob
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import dask as da
import xesmf as xe
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import load_model
from keras.utils import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



    
# data preprocessing functions
def standardize(ds, s, m):
    assert len(list(ds.data_vars)) == len(m)

    for i, var in  enumerate(ds.data_vars):  
        ds[var] = (ds[var] - m[i])/s[i]

    return ds



# model building functions
def set_seed(seed=42):
    # sets numpy, backend, and python random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) # globally define mse

def polyMSE(ytrue, ypred):
    x = tf.where(ypred > 1e-6, ypred, 0) # only consider loss for valid values
    y = tf.where(ytrue > 1e-6, ytrue, 0)
    
    m1 =  0.
    m2 = 0.
    for n in range(2, 16, 2): # 2, 4, ..., 14
        k = tf.constant(n*0.1)
        m1 = tf.pow(x, k) + m1
        m2 = tf.pow(y, k) + m2
    
    return tf.reduce_mean(mse(m1, m2))

def abs_polyMSE(ytrue, ypred):
    ypred = abs(ypred)
    
    x = tf.where(ypred > 1e-6, ypred, 0) # only consider loss for valid values
    y = tf.where(ytrue > 1e-6, ytrue, 0)
    
    m1 =  0.
    m2 = 0.
    for n in range(2, 16, 2): # 2, 4, ..., 14
        k = tf.constant(n*0.1)
        m1 = tf.pow(x, k) + m1
        m2 = tf.pow(y, k) + m2
    
    return tf.reduce_mean(mse(m1, m2))
    

def build_wnet(hp, add_ReLU=False, use_abs_loss=False, seed=0):
    '''Builds and returns the initialized wnet_prior model

    Parameters
    ----------
    hp : dict
        The dictionary containing hyperparameter settings
    add_ReLU : bool
        Boolean that enables adding a ReLU output layer
    use_abs_loss : bool
        Boolean that enables use of abs_polyMSE rather than polyMSE
    seed : int
        The seed to initialize weights with (default 0)

    Returns
    -------
    model:
        The initialized keras model
    
    '''
    num_feats =  hp['num_features']
    input_data = keras.Input(shape=(num_feats,))
    initializer = tf.keras.initializers.HeUniform(seed=seed) # seed for reproducibility of initial weights
  
    x = input_data
    for hidden_layer_size in hp['hidden_layer_sizes']:
        x = layers.Dense(hidden_layer_size, kernel_initializer=initializer)(x) 
        x = layers.LeakyReLU(alpha=0.2)(x)       
	
    output = layers.Dense(1)(x)
    if add_ReLU:
        output = layers.ReLU()(output)
    
    model = keras.Model(input_data, output)
    opt = tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'], amsgrad=True)

    if use_abs_loss:
        model.compile(loss=abs_polyMSE, optimizer=opt)
    else:
        model.compile(loss=polyMSE, optimizer=opt)

    return model 

def set_callbacks(name = "Wnet_prior", patience=20):
    '''Returns callbacks to be used for Wnet_prior model training

    Parameters
    ----------
        name (str): The name for the csv logger (default is Wnet_prior)
        patience (int): The number of epochs of no loss improvement before early stopping (default 20)

    Returns
    -------
        callbacks (list): A list of callback functions
    '''
    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.000000001,  
                       patience=patience, 
                       verbose=1)

    csv_logger = CSVLogger(name + '.csv', append=True)

    model_checkpoint = ModelCheckpoint(name + '.hdf5',
                               monitor='val_loss', 
                               verbose=1,
                               save_best_only=True, 
                               mode='min')
    
    callbacks = [csv_logger, model_checkpoint, early_stop]   
    return callbacks  
 

# classes below

class get_dataset():
    '''
    A class used to represent a dataset and customize batch retrieval
    
    ...
    
    Attributes
    ----------
    name : (str) 
        Name of the dataset
    filepaths : list
        List of filepaths of .nc files
    location : str
        Location to subset data at (default None)
    g5nr_data : DataSet
        Xarray DataSet containing g5nr data from correct files and location
    all_times : list
        List of all available timesteps in g5nr_data
    times_seen : set
        Set of timesteps that have been randomly selected for batches
    batch_times : list
        List of the randomly selected timesteps for the current batch
    data_in : DataSet
        Xarray DataSet containing 14 input variables processed from g5nr_data
    data_out : DataArray
        Xarray DataArray containing Wstd values taken from g5nr_data
        
    Methods
    -------
    preprocess_g5nr(ds)
        Ensures that g5nr data being read in is between levels 1 and 72 by dropping all other levels.

    get_data(times_per_batch)
        Randomly selects times_per_batch number of timesteps and selects g5nr_data at those timesteps,
        which forms the current data batch. Also standardizes the input variables and creates data_in and data_out.

    get_data_test(num_times=-1)
        Mimics get_data, but for testing purposes.
        Selects the first num_times timesteps of g5nr data, or if num_times is -1 (default), selects all timesteps.

    get_Xy(batch_size=2048)
        Processes current data batch (data_in and data_out) to be used for training and predicting.
        Concatenates 4 surface variables, stacks coordinates, and chunks by batch_size.
    '''

    
    # class constants
    LEV1 = 1
    LEV2 = 72
    VARS_IN = ['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']
    MEANS = [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6] # hard-coded means for vars in VARS_IN
    STDS = [30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5] # hard-coded std for vars in VARS_IN
    SURF_VARS = ['AIRD', 'KM', 'RI', 'QV']
    FEATS = len(VARS_IN) + len(SURF_VARS) # this should be 14
    N_FEATS_IN = len(VARS_IN) * (LEV2 - LEV1 + 1) # this should be 10 * 72, and is used for chunking
    CHUNK = {"lat": -1, "lon": -1, "lev": -1, "time": 1} # used in reading in the data
    COORDS = {
        "US": {"lat1": 25, 
               "lat2": 50,
               "lon1": -150,
               "lon2": -50}
    }
    G5NR_FP_NEW = "/home/shared/data2/RIPS_Wnet_data.nc"
    G5NR_FP_OLD = "/home/shared/data/nature_test_data.nc"
    

    def __init__(self, filepaths=None, name="", location=None):
        self.name = name

        if (filepaths is None):
            filepaths = [self.G5NR_FP_NEW, self.G5NR_FP_OLD]
            
        self.g5nr_data = xr.open_mfdataset(filepaths, chunks=self.CHUNK,
                                      concat_dim="time", combine="nested",
                                      preprocess=self.preprocess_g5nr, parallel=True)

        if location:
            self.g5nr_data = self.g5nr_data.sel(lat=slice(self.COORDS[location]["lat1"], self.COORDS[location]["lat2"]), 
                                                    lon=slice(self.COORDS[location]["lon1"], self.COORDS[location]["lon2"]))

        self.all_times = list(self.g5nr_data['time'].values)

        self.times_seen = set()

    
    def preprocess_g5nr(self, ds):
        tmp = ds.sel(lev=slice(self.LEV1, self.LEV2))
        return tmp.where(tmp['lev'] != 0, drop=True)
        
    
    def get_data(self, times_per_batch):
        self.batch_times = random.sample(self.all_times, times_per_batch) # gets random timestamps for a single batch
        # self.batch_times = ['2006-01-08T10:30:00.000000000'] # for initial testing
        
        g5nr_batch = self.g5nr_data.sel(time=self.batch_times)
        self.times_seen.update(self.batch_times)

        data_in = g5nr_batch[self.VARS_IN]
        data_in = xr.map_blocks(standardize, data_in, kwargs = {"m":self.MEANS, "s": self.STDS}, template = data_in)
        
        self.data_in = data_in # this is a DataSet
        self.data_out = g5nr_batch['Wstd'] # this is a DataArray

    def get_data_test(self, num_times=-1):
        if (num_times > 0) and (num_times < len(self.all_times)):
            times = self.all_times[0:num_times]
        else:
            times = self.all_times
        
        g5nr_batch = self.g5nr_data.sel(time=times)

        data_in = g5nr_batch[self.VARS_IN]
        data_in = xr.map_blocks(standardize, data_in, kwargs = {"m":self.MEANS, "s": self.STDS}, template = data_in)
        
        self.data_in = data_in
        self.data_out = g5nr_batch['Wstd']

        return self.get_Xy()
        
    
    def get_Xy(self, batch_size=2048):
        Xall = self.data_in
        yall = self.data_out

        levs = Xall.coords['lev'].values
        for var in self.SURF_VARS:
            Xs = Xall[var].sel(lev = [71]) # 1 level above surface
            Xsfc = Xs
            
            for lev in range(len(levs)-1):
                Xsfc = xr.concat([Xsfc, Xs], dim='lev')
                
            Xsfc = Xsfc.assign_coords(lev=levs)
            Xall[f"{var}_sfc"] = Xsfc

        Xall =  Xall.unify_chunks()
        Xall = Xall.to_array()
        Xall = Xall.stack( s = ('time', 'lat', 'lon', 'lev')) 
        Xall = Xall.rename({"variable": "feature"})                       
        Xall = Xall.squeeze()
        Xall = Xall.transpose()
        Xall = Xall.chunk({"feature": self.N_FEATS_IN, "s": batch_size})

        yall = yall.stack(s = ('time', 'lat', 'lon', 'lev' ))
        yall =  yall.squeeze()
        yall =  yall.transpose()   
        yall =  yall.chunk({"s": batch_size})

        return Xall, yall



'''
This class provides a way to stream the data batches iteratively.
DaskGenerator inherits from tf.keras.utils.PyDataset, which lets us do multiprocessing in a safer way.

'''
class DaskGenerator(Sequence):
    '''
    A class inheriting from keras Sequence (PyDataset) used to streamline the data batch retrieval process

    ...
    
    Attributes
    ----------
    dataset : dataset
        A dataset instance containing functionality to load in new data batches
    epochs_per_batch : int
        Number of epochs before loading in a new data batch
    times_per_batch : int
        Number of timesteps in a data batch
    batch_size : int
        Batch size (default 2048) for each gradient update (note this is different from a data batch)
    enable_on_epoch_end : bool
        Boolean that enables new data batch retrieval after a certain number of epochs
    num_samples : int
        Size of current data batch
    sample_batches : list
        List of dask.delayed objects partitioning the data batch into batches of sample inputs
    class_batches : list
        List of dask.delayed objects partitioning the data batch into batches of sample outputs
    curr_epoch : int
        Counter that tracks the number of epochs spent with the current data batch
    
    Methods
    -------
    on_epoch_end()
        Prints information on current data batch, epoch, and total timesteps seen in training.
        Also periodically restarts with a new data batch when enable_on_epoch_end is True
    '''

    
    def __init__(self, dataset, epochs_per_batch, times_per_batch, batch_size=2048, enable_on_epoch_end=True):
        self.dataset = dataset
        self.epochs_per_batch = epochs_per_batch
        self.times_per_batch = times_per_batch
        self.batch_size = batch_size
        self.enable_on_epoch_end = enable_on_epoch_end

        # Load first round of data
        self.dataset.get_data(self.times_per_batch)
        X, y = self.dataset.get_Xy(self.batch_size)
        X = X.persist()
        y = y.persist()
        
        self.num_samples = len(y['s'].values) # size of a data batch
        
        self.sample_batches = X.data.to_delayed()
        self.class_batches = y.data.to_delayed()

        self.curr_epoch = 1

        assert len(self.sample_batches) == len(self.class_batches), 'lengths of samples and classes do not match'
        assert self.sample_batches.shape[1] == 1, 'all columns should be in each chunk'

    def __len__(self):
        '''Total number of batches, equivalent to Dask chunks in 0th dimension. This is the number of minibatches in one data batch.'''
        return len(self.sample_batches)


    def __getitem__(self, idx):
        '''Extract and compute a single chunk returned as (X, y). This is also a minibatch of size self.batch_size'''
        X_mini, y_mini = da.compute(self.sample_batches[idx, 0], self.class_batches[idx]) # compute minibatch at given index
        X_mini = np.asarray(X_mini).squeeze() # convert to numpy arrays, remove extra dimensions
        y_mini = np.asarray(y_mini).squeeze()

        return X_mini, y_mini

    def on_epoch_end(self):
        print(f"_{self.dataset.name}_epoch_{self.curr_epoch}/{self.epochs_per_batch}_ \t times {self.dataset.batch_times}")
        print(f"{self.dataset.name} has seen {len(self.dataset.times_seen)} times: \t {self.dataset.times_seen}")
        
        self.curr_epoch += 1

        if self.enable_on_epoch_end and (self.curr_epoch > self.epochs_per_batch): # get new batch, restart
            # ===== restart ============
            self.curr_epoch = 1

            # ===== getting new batch ========
            self.dataset.get_data(self.times_per_batch)
            print(f"NEW DATA BATCH ({self.dataset.name}): \t {self.dataset.batch_times}")

            # ===== processing data ==========
            X, y = self.dataset.get_Xy(batch_size=self.batch_size)
            X = X.persist() # persist data in memory
            y = y.persist()
            self.num_samples = len(y['s'].values)
            
            self.sample_batches = X.data.to_delayed() # convert new data to delayed objects for lazy computation
            self.class_batches = y.data.to_delayed()

        if (self.dataset.name == 'train'):
            print('\n\n') # just for better readability in log file
