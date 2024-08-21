import numpy as np
from tqdm import tqdm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
from joblib import dump, load

import xarray as xr
from xhistogram.xarray import histogram
import pandas as pd

from data_retrieval import get_data, get_site_df


class ProgressRandomForestRegressor(RandomForestRegressor):
    def fit(self, X, y):
        self.n_estimators = getattr(self, 'n_estimators', 100)
        self.estimators_ = []
        self._validate_y_class_weight(y)
        
        # Initialize the base estimator
        self.base_estimator_ = DecisionTreeRegressor(random_state=self.random_state)
        
        # Progress bar
        with tqdm(total=self.n_estimators, desc="Fitting Random Forest") as pbar:
            for i in range(self.n_estimators):
                tree = clone(self.base_estimator_)
                if self.bootstrap:
                    n_samples = X.shape[0]
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    X_sample = X[indices]
                    y_sample = y[indices]
                    tree.fit(X_sample, y_sample, sample_weight=None, check_input=False)
                else:
                    tree.fit(X, y, sample_weight=None, check_input=False)
                self.estimators_.append(tree)
                pbar.update(1)  # Update progress bar for each fitted tree

        # Call the parent class's fit method to ensure all necessary attributes are set
        super().fit(X, y)

        return self
    
    # ======== LOSS FUNCTION HERE ==========
    def score(self, X, y):
        # Calculate MAE as the validation loss
        y_pred = self.predict(X)
        return -mean_absolute_error(y, y_pred)
    # =======================================

def augment_data(df, num_repeats=4, perturbation_factor=0.01):
    augmented_data = []
    
    for _ in range(num_repeats):
        # Apply random perturbation
        perturbation = np.random.normal(loc=0, scale=perturbation_factor, size=df.shape)
        perturbed_df = df.copy()
        perturbed_df.iloc[:, :] += perturbation  # Apply perturbation to all columns
        augmented_data.append(perturbed_df)
    
    return pd.concat(augmented_data, ignore_index=True)

def get_dataframe(training_sites, levels):
    split_const = 0.8
    
    dataframes = []
    for site in training_sites:
        df_X = get_site_df(site)

        df_X = df_X[df_X['lev'].isin(levels)]
        df_X = df_X.drop(['lev', 'time'], axis = 1)
        #Just as they say in the paper
        if site in ['mao', 'lei', 'man', 'lim']:
            df_X = df_X[:int(split_const*len(df_X))]
            df_X = augment_data(df_X)
        else:
            df_X = df_X[:int(split_const*len(df_X))]
        dataframes.append(df_X)

    # Concatenate all DataFrames into one big training DataFrame
    training_df = pd.concat(dataframes, ignore_index=True)
    return training_df

#Setting up globals
training_sites = ['asi', 'cor', 'ena', 'lei', 'manus', 'mao', 'lim', 'nsa', 'pgh', 'sgp_pbl', 'twp']
site = ['asi']
mid_troposphere = [61, 62, 63, 64, 65, 66]
high_troposphere = [54, 55, 56, 57, 58, 59, 60]
low_troposphere = [67, 68, 69, 70, 71]
all_troposphere = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
target_var = 'W_obs'


#The training happens by each slice of the troposphere, and n_estimators is the number of trees we train

#High troposphere

training_df = get_dataframe(site, high_troposphere)
X, y = training_df.drop([target_var], axis = 1), np.array(training_df[target_var])

rf = ProgressRandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(np.array(X), np.array(y))

dump(rf, './RF/RF_high_asi.joblib', compress=('gzip', 4))

#Mid troposphere

training_df = get_dataframe(site, mid_troposphere)
X, y = training_df.drop([target_var], axis = 1), np.array(training_df[target_var])

rf = ProgressRandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(np.array(X), np.array(y))

dump(rf, './RF/RF_mid_asi.joblib', compress=('gzip', 4))

# Low troposphere

training_df = get_dataframe(site, low_troposphere)
X, y = training_df.drop([target_var], axis = 1), np.array(training_df[target_var])

rf = ProgressRandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(np.array(X), np.array(y))

dump(rf, './RF/RF_low_asi.joblib', compress=('gzip', 4))


