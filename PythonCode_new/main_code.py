import GaitRecognitionFunctions as GR
import data_augmentation as DA
import numpy as np
import pandas as pd
import openpyxl
import os
import scipy.io as spio
from keras.models import load_model
from keras.utils import plot_model



# General settings
repeats = 1  # repeat times of training the model
wk_label = 1
window_size = 200
overlap_rate = 0.5
Nfolds = 5            # train:test = 4:1
Nfolds_val = 3        # train:validation = 2:1
input_axis = 6        # 3-> only acc, 6-> acc&gyr
aim_label = 1         # label of walking: 1
del_labels = [0, 11]  # activities "0=none" and "11=undefined" will be deleted to save memory
augmentation_methods = ['rotation']  # can for aim 1 or aim 2
fs_trainingdata = 100    # raw sampling frequency for data of training models
fs_validationdata = 104  # raw sampling frequency of external data used for validating the existing models

InputDataDir = './Example Data/for model training_Aim1'
ModelsSaveDir = "./CNN_models_save"      # from AIM 1 / self defined
ModelsInfoDir = "./CNN_models_info"      # from AIM 1
ModelResultDir = "./CNN_models_results_from AIM1"  
ExValDataTxtDir = "./Example Data/for external validation_Aim2" # for AIM 2
ExValScoresDir = "./CNN_models_results_from AIM2" # for AIM 2
Unsupervised_data_path = './Example Data/for unsupervised prediction_Aim3/PID927.mat'  
BestModelDir = "./best_cnn_models" # for AIM 3

if not os.path.exists(ModelsSaveDir):
    os.makedirs(ModelsSaveDir)
if not os.path.exists(ModelsInfoDir):
    os.makedirs(ModelsInfoDir)
if not os.path.exists(ModelResultDir):
    os.makedirs(ModelResultDir)
if not os.path.exists(ExValScoresDir):
    os.makedirs(ExValScoresDir)

#
# # 1) AIM 1: train the model
# # groups: no. subject of each sampling points
# # all_subject: unique no. of all subjects
DataX, DataY_binary, groups, filenames, all_subjects = GR.load_matfiles(InputDataDir, aim_label, input_axis, del_labels)

GR.train(repeats, DataX, DataY_binary, groups, Nfolds, augmentation_methods, input_axis, Nfolds_val, fs_trainingdata,
          window_size, overlap_rate, ModelsInfoDir, ModelsSaveDir, ModelResultDir)

# # 2) AIM 2: validate the existing models
# put the '.h5' model into a folder, here we use ModelsSaveDir
plotsingal = True
DataX, DataY, groups = GR.load_txtdata(ExValDataTxtDir, input_axis)
model_path = ModelsSaveDir
GR.validation(fs_validationdata, window_size, overlap_rate, input_axis, model_path, DataX, DataY, groups,
               augmentation_methods, plotsingal, ExValScoresDir)

# 3) AIM 3: predict unsupervised data
item = spio.loadmat(Unsupervised_data_path)
X = item['ACC'][:,1:4]

model_path = f'{BestModelDir}/CNNmodel_3axis.h5'
# cnn_model = load_model(model_path)             # plot the model structure

overlap_rate = 0.995
window_size = 200
GR.predict_data_unsupervised(X, model_path, window_size, overlap_rate)

