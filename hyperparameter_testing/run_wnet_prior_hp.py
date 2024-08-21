'''
This file trains a version of wnet_prior in order to perform hyperparameter testing.

To run in command line:
- nohup python wnet_prior_hp.py version learning_rate batch_size seed > prior_v{version}.log 2>&1 &
- ./run_wnet_prior_hp.sh learning_rate batch_size seed
- ./run_multiple_hp.sh learning_rate batch_size seed
'''

import sys
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable debugging logs for info, warnings

import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import load_model
from keras.utils import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# make sure wnet_prior is in the same directory
from wnet_prior import get_dataset, DaskGenerator
from wnet_prior import build_wnet, polyMSE, set_callbacks


# ======== access command line arguments ==============
if (len(sys.argv) != 5): # first argument is filename
    print(f"Usage: {sys.argv[0]} version learning_rate batch_size seed")
    sys.exit(1)

# 1: version
version = int(sys.argv[1])

# 2: learning rate
try:
    learning_rate = float(sys.argv[2])
except:
    print("Error: Could not read learning_rate_arg, so defaulted to 0.0001.")
    learning_rate = 0.0001

# 3: batch size (minibatch)
try:
    batch_size = int(sys.argv[3])
except:
    print("Error: Could not read batch_size_arg, so defaulted to 2048.")
    batch_size = 2048

# 4: seed
try:
    seed = int(sys.argv[4])
except:
    print("Error: Could not read seed_arg, so defaulted to 0.")
    seed = 0




# ========= define hyperparameters ==============
model_name = f"wnet_prior_v{version}"

hp = {
    'num_layers': 5,
    'num_nodes': 128,
    'num_features': [],
    'hidden_layer_sizes': [],
}
hp['hidden_layer_sizes'] = (hp['num_nodes'],) * hp['num_layers']
hp['learning_rate'] = learning_rate

# size of batches
times_per_batch_train = 2 # 2-3 avoids overfitting
times_per_batch_val = 1

# epochs (per batch, total)
epochs_per_batch_train = 5
epochs_per_batch_val = 10
num_epochs = 1500

# training architecture
enable_on_epoch_end = True
patience = 100

# environment settings
use_CPU = True



# ============ set up environment ============
if use_CPU: # in the case where no GPU is available
    try:
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except Exception as e:
        print(f"Error disabling GPU: {e}")

strategy = tf.distribute.MirroredStrategy()



# ========== get data and final parameters ==============
# these instances train with the 10 new timesteps so that we can test with the old 5 timesteps (all random)
train_data = get_dataset(filepaths="/home/shared/data2/RIPS_Wnet_data.nc", name="train", location="US")
val_data = get_dataset(filepaths="/home/shared/data2/RIPS_Wnet_data.nc", name="val", location="US")

train_gen = DaskGenerator(train_data, epochs_per_batch_train, times_per_batch_train, batch_size, enable_on_epoch_end)
val_gen = DaskGenerator(val_data, epochs_per_batch_val, times_per_batch_val, batch_size, enable_on_epoch_end)

hp['num_features'] = train_data.FEATS # should be 14
steps = int(0.99 * train_gen.num_samples / batch_size)


# ========= save hyperparams and training info ==========
print(f"version {version} \t learning rate {hp['learning_rate']} \t batch size {batch_size} \n" +
      f"{num_epochs} epochs \t enable_on_epoch_end {enable_on_epoch_end} \t patience {patience} \t seed {seed} \n" +
     f"times_per_batch_train {times_per_batch_train} \t times_per_batch_val {times_per_batch_val} \n" +
     f"epochs_per_batch_train {epochs_per_batch_train} \t epochs_per_batch_val {epochs_per_batch_val}")



# ========== build and train model ===================
with strategy.scope():
    model =  build_wnet(hp, seed)

start = time.time()
history = model.fit(train_gen,
                    validation_data=val_gen,
                    steps_per_epoch=steps,
                    epochs=num_epochs,
                    verbose=2,
                    callbacks=set_callbacks(model_name, patience)
                   )
end = time.time()


# ========= save hyperparams and training info ==========
print(f"version {version} \t learning rate {hp['learning_rate']} \t batch size {batch_size} \t training time {end-start} \n" +
      f"{num_epochs} epochs \t enable_on_epoch_end {enable_on_epoch_end} \t patience {patience} \t seed {seed} \n" +
     f"times_per_batch_train {times_per_batch_train} \t times_per_batch_val {times_per_batch_val} \n" +
     f"epochs_per_batch_train {epochs_per_batch_train} \t epochs_per_batch_val {epochs_per_batch_val}")



# ========= plot loss ==================================
plt.switch_backend('agg')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_name + '_loss.png')

