# This code is to predict activities (walk and non-walk) by using the existing best CNN model,
# so the dataset doesn't have activity label y
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import GaitRecognitionFunctions_general as GR
from tensorflow.keras.models import load_model
from keras.utils import plot_model


# Some settings
fs = XXX            # sampling frequency of the sensor
window_size = 200   # the same value during training process
percentage = 0.5    # the same value during training process
input_axis = 6      # optional, 6 -> acceleration& gyroscope, 3 -> only acc

InputDataDir = 'XXXXX'
PredictPNGDir = './github_rwk_predict/png'
PredictSVGDir = './github_rwk_predict/svg'

if not os.path.exists(PredictPNGDir):
    os.makedirs(PredictPNGDir)
if not os.path.exists(PredictSVGDir):
    os.makedirs(PredictSVGDir)


# load the best model 
if input_axis == 3:
   best_model = './best_cnn_models/CNNmodel_3acc.h5'
elif input_axis == 6:
   best_model = './best_cnn_models/CNNmodel_6acc.h5'
else:
    raise ValueError("Invalid value for data_axis. It should be either 3 or 6.")
    
cnn_model = load_model(best_model)
plot_model(cnn_model, to_file=f'{PredictPNGDir}/Model Structure.png', show_shapes=True, show_layer_names=True) # save the structure


# load the data
signals_sta = np.loadtxt(f'{InputDataDir}/signals_sta.txt')
signals_wk = np.loadtxt(f'{InputDataDir}/signals_wk.txt')
group_sta = np.loadtxt(f'{InputDataDir}/group_sta.txt')
group_wk = np.loadtxt(f'{InputDataDir}/group_wk.txt')
with open(f'{InputDataDir}/filenames_balance','rb') as fp:
    filenames_sta = pickle.load(fp)
with open(f'{InputDataDir}/filenames_wk', 'rb') as fp:
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

# predict ylabel 
y_predict = cnn_model.predict(X_win).round()

# remove the overlapping, although it does not influence the results
X_rm_repeat = X_win[:, :100, :]
X = X_rm_repeat.reshape(-1, 3)
y_predict_final = np.repeat(y_predict, 100, axis=0)
groups_final = np.repeat(groups_win, 100, axis=0)

# plot
time_seconds = np.arange(len(X)) / fs
plt.figure(figsize=(19, 10))
plt.plot(time_seconds, X[:, 0])
plt.plot(time_seconds, y_predict_final[:, 1])
plt.legend(['Signal', 'Predicted Label (wk=1)'])
plt.xlabel("Seconds", fontsize=20)
plt.ylabel("Gravity Acceleration (9.8 m/s²)", fontsize=16)
plt.tick_params(axis='x', labelsize=15, labelcolor='black', pad=10)
plt.tick_params(axis='y', labelsize=15, labelcolor='black', pad=10)
plt.title('Vertical acceleration with predictied ylabel') # here,AccX is vertical acc

plt.savefig(f'{PredictPNGDir}/VTacc_pred_y.png') 
plt.savefig(f'{PredictSVGDir}/VTacc_pred_y.svg')
