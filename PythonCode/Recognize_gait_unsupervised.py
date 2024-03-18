# This code is to predict activities (walk and non-walk) by using the existing best CNN model,
# so the dataset doesn't have activity label y
import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GaitRecognitionFunctions_general as GR
import data_augmentation_general as DA
from tensorflow.keras.models import load_model
from keras.utils import plot_model


# Some settings
fs = 104  # raw sampling frequency
window_size = 200
percentage = 0.5
input_axis = 6     # 3-> only acc, 6-> acc&gyr
ExValDataTxtDIr = "/Users/yugezi/Desktop/1.1_ProjectVIBE_MP/3_ADAPT/5_MichielPunt_ExternalValitaionData/UsedInModel_txtFiles"
model_folder = '/Users/yugezi/PycharmProjects/ADAPT-project/Acc only/TrainedModel_ROT90_Results/CNN_models_save'
ExValPredictPNGDir = "/Users/yugezi/PycharmProjects/ADAPT-project/Acc only/TrainedModel_ROT90_Results/CNN_models_use_byExValData/png"
ExValPredictSVGDir = "/Users/yugezi/PycharmProjects/ADAPT-project/Acc only/TrainedModel_ROT90_Results/CNN_models_use_byExValData/svg"
ExValScoresDir = "/Users/yugezi/PycharmProjects/ADAPT-project/Acc only/TrainedModel_ROT90_Results/CNN_models_result"

if not os.path.exists(ExValPredictPNGDir):
    os.makedirs(ExValPredictPNGDir)
if not os.path.exists(ExValPredictSVGDir):
    os.makedirs(ExValPredictSVGDir)

print(model_folder)
print(ExValDataTxtDIr)
# save the best model structure picture
best_model = f'{model_folder}/CNNmodel_8.h5'
cnn_model = load_model(best_model)
plot_model(cnn_model, to_file=f'{ExValScoresDir}/Model Structure.png', show_shapes=True, show_layer_names=True)

# load the data
signals_sta = np.loadtxt(f'{ExValDataTxtDIr}/signals_sta.txt')
signals_wk = np.loadtxt(f'{ExValDataTxtDIr}/signals_wk.txt')
group_sta = np.loadtxt(f'{ExValDataTxtDIr}/group_sta.txt')
group_wk = np.loadtxt(f'{ExValDataTxtDIr}/group_wk.txt')
with open(f'{ExValDataTxtDIr}/filenames_balance','rb') as fp:
    filenames_sta = pickle.load(fp)

with open(f'{ExValDataTxtDIr}/filenames_wk', 'rb') as fp:
    filenames_wk = pickle.load(fp)


# data prepare
DataX = np.vstack((signals_wk,signals_sta))
if input_axis == 3:
    DataX = DataX[:,0:3]
elif input_axis == 6:
    DataX = DataX[:,0:6]
groups_tmp = np.vstack((group_wk.reshape(-1,1),group_sta.reshape(-1,1)))
groups = groups_tmp.reshape(-1,)
# segment data into windows and categorical y_data
X_win, groups_win = GR.segment_signal_unsupervised(DataX, groups, window_size, percentage)

# load and use existing models to predict label y
for filename in os.listdir(model_folder):
    if filename.endswith('.h5'):
        file_path = os.path.join(model_folder, filename)
        numbers = re.search(r'(\d+)\.', filename)
        print(numbers.groups())

        cnn_model = load_model(file_path)
        y_predict = cnn_model.predict(X_win).round()

        # remove the overlapping, although it does not influence the results
        X_rm_repeat = X_win[:, :100, :]
        X = X_rm_repeat.reshape(-1, 3)
        y_predict_final = np.repeat(y_predict, 100, axis=0)
        groups_final = np.repeat(groups_win, 100, axis=0)

        time_seconds = np.arange(len(X)) / fs
        plt.figure(figsize=(19, 10))
        plt.plot(time_seconds, X[:, 0])
        plt.plot(time_seconds, y_predict_final[:, 1])
        plt.title('Predicted and True Vertical Acceleration')  #accX
        plt.legend(['Signal', 'Predicted Label (wk=1)', 'True Label (wk=-1)'])
        plt.xlabel("Seconds", fontsize=20)
        plt.ylabel("Gravity Acceleration (9.8 m/s²)", fontsize=16)
        plt.tick_params(axis='x', labelsize=15, labelcolor='black', pad=10)
        plt.tick_params(axis='y', labelsize=15, labelcolor='black', pad=10)

        plt.savefig(f'{ExValPredictPNGDir}/signal_AccxModel{numbers.group()}png') # AccX is vertical acc
        plt.savefig(f'{ExValPredictSVGDir}/signal_AccxModel{numbers.group()}svg')
