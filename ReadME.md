This algorithm was developed by Y. Zhang on 18/03/2024 in collaboration with the Norwegian University of Science and Technology, Utrecht University of Applied Sciences, and Vrije Universiteit Amsterdam.



# 1. Recognize real-world gait episodes based on deep learning methods

We developed a convolutional neural network (CNN) to recognize real-world gait or not **(binary classification)** based on inertial measurement units (IMU) data from **lower back** and the CNN model worked perfectly on older adults (mean age 76.4(5.6) years) and stroke patients(mean age 72.4(12.7) year) who can walked without aids. Therefore, our developed CNN model are suitable for older people who walk slowly, as presented in paper **XXX (paper link).**

![Model Structure](images/Model%20Structure.png)
**Figure1. CNN Model structure (The input IMU data can be 3-axis or 6-axis, W is the window size, 200 is 2 seconds with sampling frequency 100 Hz)**
(Hyperparameters: Epochs = 30,Batch_size = 32, Filters = 64, Kernel_size = 3)

![Model performance_external dataset](/images/Model%20performance_external%20dataset.png)
**Figure2. CNN Model performance on external dataset (stroke patients)(Note:DA, the abbreviation for data augmentation, here we use rotation 90° on xyz-axis, separately）**



## 2. How to use the algorithm
For different aims, you can choose different main Python code. 

**1) train a CNN model, see 2.1**
```
Model_training.py
```
**2) validate externally the existing model, see 2.2**
```
External_validate_model.py
```
**3) predict the unknown activities, see 2.3**
```
Recognize_gait_unsupervised.py
```

In addition, the repository contains **subfunction code for training process (see 3.1) and data augmentation (see 3.2)**.
```
GaitRecognitionFunctions_general.py
data_augmentation_general.py
```

You can select the code according to your need.
|Aim |Data with true activity labels| Main code | Subfunction | Use existing models|
| ---|----------------------- | -----------| ----------|----------|
|Trian a model| Yes | Model_training.py | GaitRecognitionFunctions_general.py <br>data_augmentation_general.py| No |
|Validate externally| Yes | External_validate_model.py | GaitRecognitionFunctions_general.py <br>data_augmentation_general.py| Yes |
|Predict the unknown data| No | Recognize_gait_unsupervised.py | GaitRecognitionFunctions_general.py | Yes |


## 2.1. Aim 1: To train a CNN model
There are 1 main code and 2 subfunction code for this aim. Put all these code into the same folder, for example ./github_rwk/", so that we can call subfuntions in the main functions.

The main code
```
Model_training.py
```
The subfunciton code
```
GaitRecognitionFunctions_general.py # required
data_augmentation_general.py        # optional
```

### 2.1.1. Install the necessary packages
Before running the code, make sure install all necessary packages "numpy, pandas, openpyxl, os" in your python environment with python version > 3.6.

To **check** which packages are installed in your Python environment, you can type the following command on the Python console or terminal:
```
pip list
```
If you're using Anaconda or Miniconda, you can use the conda list command:
```
conda list
```

If not, you can **install** these packages, by below code
```
pip install package_name
or
conda install package_name
```

Finally, **import** necessary packages in the main code "Model_training.py"
```
import GaitRecognitionFunctions_general as GR
import data_augmentation_general as DA
import numpy as np
import pandas as pd
import openpyxl
import os
```

### 2.1.2. Setting folders
At the beginning of the main code "Model_training.py", we set the location of the input data and output. For folder of input data, it should contain data files, eg. mat files. For the folders of output, just set the location and the code will **automatically generate** the folder.
```
# The folder of input
InputDataDir = './github_rwk/InputData/'

# The folders of output, set the location and it can be automatically generated later
# a) store the model performance of testing and validating datasets
ModelResultDir = "./github_rwk/CNN_models_results"
# b) store the '.h5' model files 
ModelsSaveDir = "./github_rwk/CNN_models_save"
# c) will generate 3 subfolders, i.e., "InitialWeights", "loss plot" and "number of windows", storing the related paramters during the training
ModelsInfoDir = "./github_rwk/CNN_models_info"

if not os.path.exists(ModelResultDir):
    os.makedirs(ModelResultDir)
if not os.path.exists(ModelsSaveDir):
    os.makedirs(ModelsSaveDir)
if not os.path.exists(ModelsInfoDir):
    os.makedirs(ModelsInfoDir)
```

### 2.1.3. Setting parameters
There are some parameters that need to be set in the main code. Below are the default values, you can modify them by your own.
```
repeats = 2                          # repeat times of the model
fs = 100                             # sampling frequency of the IMU
wk_label = 1                         # the y-label of walking
window_size = 200                    # window size for model training, here is 2 seconds
percentage = 0.5                     # the overlapping of windows
Nfolds = 5                           # the number of fold splitting the training and test sets
Nfolds_val = 3                       # the number of fold splitting the training and validating sets
input_axis = 6                       # 3-> only acc, 6-> acc&gyr
augmentation_methods = ['rotation']  # type the methods you will use, if no, type "None" instead
```

### 2.1.4. Load data
**What you need to do is to prepare the IMU data in a folder with "mat" files. The signals can be 6 axes [3-axis acceleration, 3-axis gyroscope] or only 3 axes [3-axis acceleration] (random directions), which you can set in 2.1.3 "input_axis".
Each ".mat" file represents each subject and the signals are stored in variable "signal" of the mat file.**
If the data is in .txt files, you can reference 2.2.4.

We load the data by using the function "load_matfiles" in the Python code "GaitRecognitionFunctions_general.py". After loading, we will get DataX, DataY, DataY_binary, and groups.

In our paper, **the ADAPT dataset was used to train the model**, collected by Bourke et al, including semi-structured supervised and free-living unsupervised situations both with manually annotated labels based on video data. DataX has 6 axes, i.e., 3-axis acceleration, 3-axis gyroscope. And columns XYZ axes of acceleration and gyroscope respond to vertical (up, positive values), medial-lateral (right +), and anteroposterior (anter +), respectively. DataY includes all activity labels. DataY_binary includes walking and non-walking labels. Groups are the number of subjects, for further splitting datasets.

The code loading data in the main, shown as below
```
DataX, DataY, DataY_binary, groups, filenames, subject_number = GR.load_matfiles(InputDataDir, wk_label, input_axis)
DataX_new, DataY_binary_new, groups_new = delete_useless_label(DataX, DataY, DataY_binary, groups)    # Optional. To delete the activities that don't make sense but will affect the training results. Here, we delete "undefined" and "none".
```

### 2.1.5. Model training
Then there is nothing you need to modify. After splitting training, validating and testing datasets by subjects, the code will automatically augment the training and validating datasets according to your set in **2.1.3**. Finally, put all datasets into the model running, where includes segmenting windows, balancing data, and fitting model, then we can get the results "score_val", "score_ test", and model into the responding folders.

Since each time, the splitting datasets contains different subjects' data, leading to different model results, so you can run it several times to select the model with best results.

```
Scores_val = list()
Scores_test = list()

for i in range(1,repeats):
    print("===========" * 6, end="\n\n\n")
    print('Repeat ', i)
    X_train_val, X_test, y_train_val, y_test,groups_train_val,groups_test = GR.split_GroupNfolds(DataX_new,
                                                                                                 DataY_binary_new,
                                                                                                 groups_new,
                                                                                                 Nfolds)

    print(f'## 2) DA = {augmentation_methods}')
    print("## 3) split train and validate data")
    if augmentation_methods is not None:
        X_train_val_all, y_train_val_all, groups_train_val_all = use_data_augmentation(X_train_val, y_train_val,
                                                                                       groups_train_val, fs,
                                                                                       augmentation_methods,
                                                                                       input_axis)
        X_train, X_val, y_train, y_val, groups_train, groups_val = GR.split_GroupNfolds(X_train_val_all,
                                                                                        y_train_val_all,
                                                                                        groups_train_val_all,
                                                                                        Nfolds_val)
    elif augmentation_methods is None:
        X_train, X_val, y_train, y_val, groups_train, groups_val = GR.split_GroupNfolds(X_train_val,
                                                                                        y_train_val,
                                                                                        groups_train_val,
                                                                                        Nfolds_val)


    print("## 4) training the model")
    CNN_model, model_history, final_epoch, score_val_i, score_test_i = GR.run_model(X_train,X_val,
                                                                                    y_train,y_val,
                                                                                    groups_train, groups_val,
                                                                                    X_test, y_test, groups_test,
                                                                                    window_size, percentage,
                                                                                    i, ModelsInfoDir)

    # combine results
    score_val_i['No.'] = i
    score_test_i['No.'] = i
    score_val_i['ModelEpoch'] = final_epoch
    score_test_i['ModelEpoch'] = final_epoch
    Scores_val.append(score_val_i)
    Scores_test.append(score_test_i)

    # save
    CNN_model.save(f'{ModelsSaveDir}/CNNmodel_{i}.h5')
    save_split_datasets_info(ModelResultDir)
    score_val_i.to_csv(f'{ModelResultDir}/scores_val.txt', mode='a', sep='\t', index=False, header=not i)
    score_test_i.to_csv(f'{ModelResultDir}/scores_test.txt', mode='a', sep='\t', index=False, header=not i)

save_dataframe(Scores_val, ModelResultDir, 'scores_val.xlsx')
save_dataframe(Scores_test,ModelResultDir, 'scores_test.xlsx')
```


The pipeline of above process is shown as the below

![Flow Chart_ADAPT](images/flow%20chart_ADAPT.png)


## 2.2. Aim 2: To externally validate the existing model
This aim is to validate an existing CNN externally by using a dataset with true activity labels (also walking and non-walking binary labels). **Following are the steps in main code "External_validate_model.py"**

### 2.2.1. Install and import the necessary packages
Import these packages, if not existing, install them accoding to the stpes 2.1.1.

```
import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.utils import plot_model
import GaitRecognitionFunctions_general as GR
import data_augmentation_general as DA
```

### 2.2.2. Setting folders
Put the dataset into the input foler and set the path on the code. 
```
ExValDataTxtDIr = './github_rwk_ExVal/InputData'
```

Set the path of the existing models, which can be the results from 2.1.
```
model_folder = "./github_rwk/CNN_models_save"
```

Set the path of output folders. If they aren't existing, they can be automatically generated.
```
ExValPredictPNGDir = "./github_rwk_ExVal/png"  # the figures of vertical acceleration with true and predicted labels
ExValPredictSVGDir = "./github_rwk_ExVal/svg"
ExValScoresDir = "./github_rwk_ExVal/CNN_models_result"

if not os.path.exists(ExValPredictPNGDir):
    os.makedirs(ExValPredictPNGDir)
if not os.path.exists(ExValPredictSVGDir):
    os.makedirs(ExValPredictSVGDir)
if not os.path.exists(ExValScoresDir):
    os.makedirs(ExValScoresDir)
```

### 2.2.3. Setting parameters
Set the parameters on the begining of the main code. 
```
fs = XX            # sampling frequency of the sensor, can be different with the one in the model training
window_size = 200  # the same size with the model training
percentage = 0.5   # the overlapping rate of windows
input_axis = 6     # 3-> only acc, 6-> acc&gyr
model_folder = "./github_rwk/CNN_models_save" # models from training process 2.2.1, or the existing models you want to validate externally
```

### 2.2.4. Load and segment data 
**We put all the IMU signals of walking and non-walking into separate ".txt" files under the input folder.** Also, the signals can be 6 axes [3-axis acceleration, 3-axis gyroscope] or only 3 axes [3-axis acceleration] (random directions). All subjects's signals are spliced vertically. The responding activity labels and subjects' number are spliced vertically and in different files.

Anyway, there are 5 '.txt' files under the input folder. They are 
- signals_wk.txt
- signals_nonwk.txt
- y_wk.txt
- y_nonwk.txt
- groups_wk.txt
- group_sta.txt
- filenames_wk      # filenames of each subject's IMU data before combined into signals_wk.txt
- filenames_nonwk

If the data is in .mat files, you can reference 2.1.4.

The code of loading data are as below
```
signals_sta = np.loadtxt(f'{ExValDataTxtDIr}/signals_sta.txt')
signals_wk = np.loadtxt(f'{ExValDataTxtDIr}/signals_wk.txt')
y_sta = np.loadtxt(f'{ExValDataTxtDIr}/y_sta.txt')
y_wk = np.loadtxt(f'{ExValDataTxtDIr}/y_wk.txt')
group_sta = np.loadtxt(f'{ExValDataTxtDIr}/group_sta.txt')
group_wk = np.loadtxt(f'{ExValDataTxtDIr}/group_wk.txt')

# load the filenames just for checking
with open(f'{ExValDataTxtDIr}/filenames_balance','rb') as fp:
    filenames_sta = pickle.load(fp)

with open(f'{ExValDataTxtDIr}/filenames_wk', 'rb') as fp:
    filenames_wk = pickle.load(fp)
```

Then, combine the data of walking and non-walking together into DAtaX and DataY
```
DataY_tmp = np.vstack((y_wk.reshape(-1,1),y_sta.reshape(-1,1)))
DataY = DataY_tmp.reshape(-1,)
groups_tmp = np.vstack((group_wk.reshape(-1,1),group_sta.reshape(-1,1)))
groups = groups_tmp.reshape(-1,)
```

The data augmentation is also optional after loading the raw data. **You can directly call the augmentaion functions and combined all the data.**
Example:
```
DataX_rot, DataY_rot, groups_rot = DA.rotation_acc(DataX, DataY, groups_clean, [90])  # add data of rotating 90° on xyz axis 
DataX_new = np.concatenate((DataX_clean, DataX_rot), axis=0)
DataY_new = np.concatenate((DataY_clean, DataY_rot), axis=0)
groups_new = np.concatenate((groups_clean, groups_rot), axis=0)
```

Before put the data into models, we segment data into windows
```
X_win, y_win, groups_win = GR.segment_signal(DataX_new, DataY_new, groups_new, window_size, percentage)
```

### 2.2.5. Validating the existing Model
There is nothing to modify for you. We use the existing models from the folder  and plot the vertical accelerations with true and predicted y labels.

```
# load existing models from the folders and predict y labels in a loop
scores_all = []
model_folder = "./github_rwk/CNN_models_save"
for filename in os.listdir(model_folder): 
    if filename.endswith('.h5'):
        file_path = os.path.join(model_path)
        numbers = re.search(r'(\d+)\.', filename)
        print(numbers.groups())

        cnn_model = load_model(file_path)
        y_predict = cnn_model.predict(X_win).round()

        # remove the overlapping, although it does not influence the results
        X_rm_repeat = X_win[:, :100, :]
        X = X_rm_repeat.reshape(-1, 3)
        y_predict_final = np.repeat(y_predict, 100, axis=0)
        y_true_final = np.repeat(y_win, 100, axis=0)
        groups_final = np.repeat(groups_win, 100, axis=0)
        score = GR.model_performance(y_true_final, y_predict_final)
        score = score.assign(loop_time=numbers.group())
        scores_all.append(score)

        time_seconds = np.arange(len(X)) / fs
        plt.figure(figsize=(19, 10))
        plt.plot(time_seconds, X[:, 0])
        plt.plot(time_seconds, y_predict_final[:, 1])
        plt.plot(time_seconds, -y_true_final[:, 1])  # true labels
        plt.title('Predicted and True Vertical Acceleration')  #accX
        plt.legend(['Signal', 'Predicted Label (wk=1)', 'True Label (wk=-1)'])
        plt.xlabel("Seconds", fontsize=20)
        plt.ylabel("Gravity Acceleration (9.8 m/s²)", fontsize=16)
        plt.tick_params(axis='x', labelsize=15, labelcolor='black', pad=10)
        plt.tick_params(axis='y', labelsize=15, labelcolor='black', pad=10)

        plt.savefig(f'{ExValPredictPNGDir}/signal_AccxModel{numbers.group()}png') # AccX is vertical acc
        plt.savefig(f'{ExValPredictSVGDir}/signal_AccxModel{numbers.group()}svg')

flattened_scores = [np.ravel(arr) for arr in scores_all]
scores = pd.DataFrame(flattened_scores, columns=["Accuracy","Precision","Sensitivity","F1-score","Specificity","loop_time"])
print('CNN model results of external dataset.\n', scores)
scores.to_excel(f"{ExValScoresDir}/scores_external_validation.xlsx",index=False)
```

The pipeline of above process is shown as the below

![flow chart_external data](images/flow%20chart_external%20data.png)

## 2.3. Aim 3: To predict unknown activities

For the prediction, the columns of input signals can contain 3-axis acceleration, 3-axis gyroscope, 3-axis magnitude data, and activity labels.

You can choose either 6 axes or only 3-axis acceleration to predict the activity labels.

The main code we need to use is 
```
PythonCode /Recognize_gait_unsupervised.py
```

### 2.3.1. Import the necessary packages
```
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import GaitRecognitionFunctions_general as GR
from tensorflow.keras.models import load_model
from keras.utils import plot_model
```

### 2.3.2. Setting folders
Set the folders of input and ouput. Input data should be put into the input folder.

```
InputDataDir = "XXXX"
PredictPNGDir = "./github_rwk_predict/png"
PredictSVGDir = "./github_rwk_predict/svg"

if not os.path.exists(PredictPNGDir):
    os.makedirs(PredictPNGDir)
if not os.path.exists(PredictSVGDir):
    os.makedirs(PredictSVGDir)

```

### 2.3.3. Setting parameters
```
fs = XXX            # sampling frequency of the sensor
window_size = 200   # the same value during training process
percentage = 0.5    # the same value during training process
input_axis = 6      # optional, 6 -> acceleration& gyroscope, 3 -> only acc
```

### 2.3.4. Load and segment data
If data are in .txt files, to load the data, please reference 2.2.1. If data are in .mat files,please reference 2.1.1.
The code of segmentation is the same,
```
X_win, groups_win = GR.segment_signal_unsupervised(DataX, groups, window_size, percentage)
```

### 2.3.5. Predict y label
Here, we got the best CNN models from 6 axes and 3 axis separately in the gituhub folder. So we can directly load the model, predict the y labels and plot them with signals.

```
model_path = '/Best CNN models/CNNmodel_6axes.h5'
OR
model_path = '/Best CNN models/CNNmodel_3axes.h5'

cnn_model = load_model(file_path)
y_predict = cnn_model.predict(X_win).round()

X_rm_repeat = X_win[:, :100, :]
X = X_rm_repeat.reshape(-1, 3)
y_predict_final = np.repeat(y_predict, 100, axis=0)
groups_final = np.repeat(groups_win, 100, axis=0)
time_seconds = np.arange(len(X)) / fs

plt.figure(figsize=(19, 10))
plt.plot(time_seconds, X[:, 0])
plt.plot(time_seconds, y_predict_final[:, 1])
plt.title('Predicted Vertical Acceleration')  #accX
plt.legend(['Signal', 'Predicted Label (wk=1)'])
plt.xlabel("Seconds", fontsize=20)
plt.ylabel("Gravity Acceleration (9.8 m/s²)", fontsize=16)
plt.tick_params(axis='x', labelsize=15, labelcolor='black', pad=10)
plt.tick_params(axis='y', labelsize=15, labelcolor='black', pad=10)

plt.savefig(f'{PredictPNGDir}/VTacc_pred_y.png') # AccX is vertical acc
plt.savefig(f'{PredictSVGDir}/VTacc_pred_y.svg')
```

## 3. Description of Subfunctions
### 3.1. Functions in "GaitRecognitionFunctions_general.py"
Packages needed are 
```
import os
import pandas as pd
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential,load_model
from keras.layers import Dense,Flatten,Dropout,Conv1D,MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import re
import pickle
```

The functions contain following subfuntions, and you can call them by
```
import GaitRecognitionFunctions_general.py as GR
GR.subfuntion_names
```

Subfuntions contain below
```
DataX, DataY, DataY_binary, groups, filenames, subject_number = load_matfiles(dataDir, aim_label, input_axis)
y_dataDichotoom = dichotomy_labels(y_data, aim_label)
X_final, y_final, groups_final = segment_signal(X, y, groups, window_size, percentage)
X_final, groups_final = segment_signal_unsupervised(X, groups, window_size, percentage)
DataX_blc, DataY_blc, DataGroups_blc = down_sampling_win(X, y, groups, i, output_path, data_type)
X_train, X_test, y_train, y_test,groups_train,groups_test = split_GroupNfolds(X_data, y_data, Groups, Nfolds)
X_train, X_val, y_train, y_val, groups_train, groups_val = split_LeaveOneOut(X_t_v, y_t_v, groups_t_v)
plot_loss(history,i,output_path)
model, early_stopping = cnn_model(x_t,y_t,i,output_path)
model, history, score_val,score_test = fit_and_evaluate(x_t, x_val, y_t, y_val,x_test,y_test, i, output_path)
score = odel_performance(y_true, y_pred)
model, model_history, final_epoch, score_val, score_test = run_model(X_train, X_val, y_train, y_val, groups_train, groups_val, X_test, y_test, groups_test, window_size, percentage, i,output_path)
```

### 3.2. To augment the dataset
The functions "data_augmentation_general.py" contain following many methods, and you can call individual method according to your aim.

```
import data_augmentation_general.py as DA
DA.method_name() 
```

The methods and hyperparameters of them are show as below and you can modify them by your own: 
| augmentation methods       | recommended scale |
| -------------------------- | ----------------- |
| Jitter                     | [0.01, 0.02]      |
| scaling                    | [0.3,0.4]         |
| Resampling (interpolation) | [0.9,1.1]         |
| Rotation                   | [90°]             |


## 4. References

[1] Bourke AK, Ihlen EAF, Bergquist R, Wik PB, Vereijken B, Helbostad JL. A physical activity reference data-set recorded from older adults using body-worn inertial sensors and video technology—The ADAPT study data-set. Sensors. 2017;17(3):559.
[2] Our ADAPT paper after publishing


## Help

For questions about the algorithm and the implementations please contact: [y6.zhang@vu.nl](mailto:y6.zhang@vu.nl)
