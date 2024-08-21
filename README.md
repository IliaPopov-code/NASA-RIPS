

# RIPS - NASA GSFC Summer 2024 Work

## Overview

This repository contains four different Python scripts for calculating and plotting the standard deviation in vertical wind velocity (sigmaW), and to demonstrate the Constraining Adversarial Trainining method following Barahona et al. "Deep Learning Parameterization of Vertical Wind Velocity Variability via Constrained Adversarial Training (doi:https://doi.org/10.1175/AIES-D-23-0025.1)". These scripts are designed to work with gridded, netcdf data from observations, high resolution simulated output, and reanalysis data. Observartional and reanalysis data should represent time series of sigmaW and atmospheric state  at different vertical positions, respectively, while the simulated data is taken from the NAS GEOS-5 Nature Run (G5NR), https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/. Other scripts provide various methods for computing and visualizing sigmaW.

## Environment

First, to use the code available in this repository, set up a Conda virtual environment. After setting up your Conda environment, run the jupyter notebooks within this repository. Make sure to install all the required dependencies and libraries into the Conda environment from the txt.

## Ensemble model

This notebook utilizes G5NR data to enhance the training of the Wnet Prior neural network model by employing a robust ensemble learning approach, specifically k-fold cross-validation. The process is designed to refine the model's generalization capabilities by dividing the dataset into 5 distinct folds, where each fold is used once as a validation set while the remaining folds form the training set. This method not only helps in mitigating overfitting but also provides a more comprehensive evaluation of the model's performance across different subsets of the data. Within the folder, you'll find the saved model weights corresponding to each of the 5 trained folds. After training, inference is performed for both the Wnet Prior Ensemble model, which aggregates the predictions from all folds, and the standard Wnet Prior model, serving as a baseline. The inference process is conducted across various levels, allowing for a detailed comparison of prediction accuracy and model performance at different stratifications. These comparisons are crucial in assessing the benefits of the ensemble approach over the traditional single-model inference, particularly in terms of capturing variability across levels.
