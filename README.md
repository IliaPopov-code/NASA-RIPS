# RIPS - NASA GSFC Summer 2024 Work

## Overview

This repository contains the code used by the RIPS2024 NASA Goddard team for their summer research project, Testing and Enhancement of Machine Learning Based Parameterizations in a Cloud Based Environment. We have included five directories containing our work on model comparison metrics and preliminary enhancements. Furthermore, we include a Python script used for retraining the Wnet_prior model, and a text file containing package dependencies for our conda environment.

Our work extends on the findings from Barahona et al. "Deep Learning Parameterization of Vertical Wind Velocity Variability via Constrained Adversarial Training (doi:https://doi.org/10.1175/AIES-D-23-0025.1)". 

These scripts are designed to work with gridded, netcdf data from observations, high resolution simulated output, and reanalysis data. Observartional and reanalysis data should represent time series of sigmaW and atmospheric state  at different vertical positions, respectively, while the simulated data is taken from the NAS GEOS-5 Nature Run (G5NR), https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/. Other scripts provide various methods for computing and visualizing sigmaW.

## Environment

First, to use the code available in this repository, set up a Conda virtual environment. We've included the text file conda_environment.txt in this repository, which can be used to create an environment by running:
<br>
```
conda create --name <env> --file <this file>
```

After setting up your Conda environment, run the jupyter notebooks within this repository. Make sure to install all the required dependencies and libraries into the Conda environment from the txt.

## Confidence Intervals

All code related to confidence intervals is found in the folder `confidence_intervals`. This code requires the datasets G5NR as well as observational datasets in order to be run. After saving the appropriate datasets, pass through the appropriate file paths throughout the notebooks. 

### (1) `residual_wild_bootstrapping.ipynb`

This notebook produces confidence intervals based on the residual and wild bootstrapping technique as outlined by https://faculty.washington.edu/yenchic/17Sp_403/Lec6-bootstrap_reg.pdf. The code provided includes processing of provided data files, the computation of confidence intervals, calculation of the coverage rate, and code for plotting confidence intervals.

Required inputs include G5NR data (accessible here: https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/) that has already been preprocessed to have input variables `['T', 'PL', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL’]` according to the method outlined by Katherine H. Breen in the repository https://github.com/katherbreen/Wnet/tree/main. 

### (2) `pairs_bootstrapping_splitting_data.ipynb`

This notebook provides code to split data that is necessary to run pairs bootstrapping for the model Wnet-prior. In this code, we provide methods to split the data into appropriate train, validation, and test sets using a 80/10/10 split. We then also provide code to resample with replacement from the training dataset based on set seeds. The resampled training datasets then can be used to train various Wnet-prior models that will be used as an ensemble to compute a confidence interval.

### (3) `pairs_bootstrapping.ipynb`

This notebook provides code to compute and plot confidence intervals based on the pairs bootstrapping technique. This code assumes that there are 10-11 input models that have been trained on the previously resampled datasets. This code also assumes the availability of G5NR data. There are options to modify the appropriate file path in order to run the code. Then, it runs the models on provided test data to compute a confidence or prediction interval and save its plots.

## Ensemble model

This notebook utilizes G5NR data to enhance the training of the Wnet Prior neural network model by employing a robust ensemble learning approach, specifically k-fold cross-validation. The process is designed to refine the model's generalization capabilities by dividing the dataset into 5 distinct folds, where each fold is used once as a validation set while the remaining folds form the training set. This method not only helps in mitigating overfitting but also provides a more comprehensive evaluation of the model's performance across different subsets of the data.

Within the folder, you'll find the saved model weights corresponding to each of the 5 trained folds. After training, inference is performed for both the Wnet Prior Ensemble model, which aggregates the predictions from all folds, and the standard Wnet Prior model, serving as a baseline. The inference process is conducted across various levels, allowing for a detailed comparison of prediction accuracy and model performance at different stratifications. These comparisons are crucial in assessing the benefits of the ensemble approach over the traditional single-model inference, particularly in terms of capturing variability across levels.

## Hyperparameter Testing

This directory contains the code used to run and compare wnet_prior models under different hyperparameters for sensitivity testing.

The files are organized as follows:

```
hyperparameter_testing/
|
|-- run_wnet_prior_hp.py
|-- run_wnet_prior_hp.sh
|-- run_multiple_hp.sh
|-- get_predictions_hp.py
|-- make_plots_hp.ipynb
```

First, run_wnet_prior_hp.py is a Python file that utilizes the methods and classes in wnet_prior.py to train a model. Hyperparameters like learning rate and batch size are expected arguments for this script.

To automate the training process, run_wnet_prior_hp.sh and run_multiple_hp.sh are shell scripts that train multiple models under user specified hyperparameters. An argument for seed was included to allow for reproducibility of the model weights.

Finally, get_predictions_hp.py generates predictions (as a .csv file) and plots for the model versions specified by the user. The make_plots_hp.ipynb notebook then produces additional plots used to compare model versions across various hyperparameters.

To use these files, minor changes will be need, such as ensuring all files are in appropriate directories and updating the filepath variable used in the code. Additionally, a text file containing the next model version number is required, as this will be read by and edited by the shell scripts.

## Model Development

This directory contains our investigation into preliminary model enhancements. The organization of the directory follows this pattern:

```
model_enhancement/
|
|-- random_forest_inference.py
|-- random_forest_training.py
```

In order to run the code properly the editing should follow this sequence:
### Update the paths in data_retrieval.py
As all files import the functions from this file, you should update the paths which go to MERRA and G5NR data, Wnet model, Wnet-prior model, and set up saving directories for the code to properly run.
### Update the training of Random Forest in random_forest_training.py
In order to retrain the Random Forest properly, the intrinsic information within the file needs to be updated. You can follow the instructions within the file to achieve this.

After that is done you can run the code within G5NR_analysis.py to construct all the plots related to G5NR and MERRA_analysis.py, which perform Exploratory Data Analysis on the MERRA-2 dataset.

## Comparison Metrics

This directory contains the code used to produce the comparison metrics in our final report.

It is organized as follows
```
comparison_metrics/
|
|-- data_retrieval.py
|-- G5NR_analysis.py
|-- MERRA_analysis.py
```


## `wnet_prior.py`

This Python script uses data from G5NR to train the neural network "Wnet-prior," which predicts sigmaW values. Our code is adapted from the Wnet_prior.py file used in https://github.com/katherbreen/Wnet, with alterations to customize our specific training process.



## Acknowledgements

Our project was hosted by the Institue of Pure and Applied Math (IPAM) at UCLA and sponsored by the NASA Goddard Space and Flight Center.

Our work and code expand on the developments in the following paper and github repository:
- https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0025.1.xml?tab_body=fulltext-display#d12260778e658
- https://github.com/katherbreen/Wnet
