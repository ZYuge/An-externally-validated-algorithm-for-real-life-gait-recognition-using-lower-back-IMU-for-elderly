This algorithm was developed by Y. Zhang on 18/03/2024 in collaboration with the Norwegian University of Science and Technology, Utrecht University of Applied Sciences, and Vrije Universiteit Amsterdam.  



# 1. Recognize real-life gait episodes using lower back IMU for older adults

We developed a convolutional neural network (CNN) based on inertial sensor data of the lower back (L5) to classify real-life activities in two categories, gait and non-gait, as presented in paper **[1] XXX (paper link)**.  

This model is suitable for older people who walk slowly. The data for model training came from healthy and gait-impaired older adults with mean age 76.4(5.6) years old [2]. The data for externally validation came from stroke survivors who could walk independently at a mean age of 72.4 (12.7) years [3]. 

![Model Structure](images/Model%20Structure.png)
**Figure1. CNN Model structure (The input IMU data can be 3-axis or 6-axis, W is the window size, 200 is 2 seconds with sampling frequency 100 Hz)**  
(Hyperparameters: Epochs = 30, Batch_size = 32, Filters = 64, Kernel_size = 3)  

![Model performance_external dataset](/images/Model%20performance_external%20dataset.png)
**Figure2. CNN Model performance on external dataset (stroke patients)(Note:DA, the abbreviation for data augmentation, here we use rotation 90° on xyz-axis, separately）**



## 2. How to use the algorithm
Before running the code, you need to install the necessary packages in python environment (versions 3.7 and above), including: os, re, pickle, math, openpyxl,h5py,numpy,pandas,scipy,matplotlib,tensorflow, keras, and sklearn.  

We provide the code for the whole process, including **data preprocessing** (data reading, balancing, augmentation and segmentation), **model training (Aim 1)** (model evaluation, fit,  and overfitting prevention), and **external validation process (Aim 2)** (data reading, model prediction, model performance evaluation).  

We also provide the best-performing model that we have obtained, into which you can put your sensor data to get the binary results **(Aim 3)**.   
**Need to do: add the performance of the best model(acc,precision,sensitivity on training, testing and external validation)**  

In **"main_code.py"**, you can call the functions in **"GaitRecognitionFunctions.py"** for following different aims.  
**Need to do: add a picture in the end, introducing functions contained in GR with the process direction**  

```
import GaitRecognitionFunctions as GR
```

**1) train a CNN model**  
```
DataX, DataY_binary, groups, filenames, all_subjects = GR.load_matfiles(InputDataDir, aim_label, input_axis, del_labels)

GR.train(repeats, DataX, DataY_binary, groups, Nfolds, augmentation_methods, input_axis, Nfolds_val, fs_trainingdata,
          window_size, overlap_rate, ModelsInfoDir, ModelsSaveDir, ModelResultDir)
```

The `GR.train()` is responsible for data preprocessing, repeated holdouts-validation, data augmentation, model training, and model saving. It controls the model training process and saves the results through a series of input parameters.  

`repeats`: number of training repetitions. In each traing, the dataset split and kernel weight in the model are random.   
`DataX`: the sensor data for model training (3 channels/6 channels).    
`DataY_binary`: the activity label corresponding to each sampling point in DataX.    
`groups`: the subject number corresponding to each sampling point.  
Other function input can be seen in below 4)

**2) externally validate the existing model**
```
DataX, DataY, groups = GR.load_txtdata(ExValDataTxtDir, input_axis)

GR.validation(fs_validationdata, window_size, overlap_rate, input_axis, model_path, DataX, DataY, groups,
               augmentation_methods, plotsingal, ExValScoresDir)
```
This `GR.validation()` is responsible for loading an existing pre-trained model, visualizes its structure, loading external validation data, and performing model using the specified settings, such as window size, overlap rate, and augmentation methods. The results and plots are stored in the directory `ExValScoresDir`.

`DataX`, `DataY`,`groups` are based on external validation data.

**3) predict the unknown activities**
```
GR.predict_data_unsupervised(X, model_path, window_size, overlap_rate)
```

**4) the general settings in the begining**  
`wk_label`: the defined label of walking.  
`aim_label`: the responding label of our targeted activity.   
`window_size`: the winidow size of data in model training.  
`overlap_rate`: the overlap rate in each window
`Nfolds`: the number of partitions (folds) you want to divide the dataset into training and testing datasets by subjects for repeated holdout-validation.  
`augmentation_methods`: choose the methods you want to use for augmentation, it can be none, one method or multiple methods. Here, we provide methods of 'noise','scaling', 'interpolation', 'rotation'.  
`input_axis`: the input channels, can be 3 for acceleration data only, and can be 6 for aceeleration & gyroscope data.  
`Nfolds_val`: the number of partitions (folds) you want to divide the training dataset into training and validating datasets.  
`fs_trainingdata`: the sampling frequency of model training data. 
`fs_validationdata`: the sampling frequency of externally validation data.   
`InputDataDir`: the folder storing the input data for model training.   
`ModelsInfoDir`: store subject numbers of splitted datasets and kernel weights in each training repeat.    
`ModelsSaveDir`: store model with '.h5' in each training repeat.  
`ModelsResultDir`: store the model performance in each training repeat. 
`ExValDataTxtDir`: store the input data for external validation. Here, the data was stored in '.txt'.
`ExValScoreDir`: store the model performance on external validation, picture of true and predicted labels on signals. 


## 3. Format of data
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



The pipeline of above process is shown as the below

![flow chart_external data](images/flow%20chart_external%20data.png)

## 2.3. Aim 3: To predict unknown activities

For the prediction, the columns of input signals can contain 3-axis acceleration, 3-axis gyroscope, 3-axis magnitude data, and activity labels.

You can choose either 6 axes or only 3-axis acceleration to predict the activity labels.

The main code we need to use is 
```
PythonCode /Recognize_gait_unsupervised.py
```

### 2.3.4. Load the best model
```
if input_axis == 3:
   best_model = './best_cnn_models/CNNmodel_3acc.h5'
elif input_axis == 6:
   best_model = './best_cnn_models/CNNmodel_6acc.h5'
else:
    raise ValueError("Invalid value for data_axis. It should be either 3 or 6.")
    
cnn_model = load_model(best_model)
plot_model(cnn_model, to_file=f'{PredictPNGDir}/Model Structure.png', show_shapes=True, show_layer_names=True) # save the structure
```

### 2.3.5. Load and segment data
If data are in .txt files, to load the data, please reference 2.2.1. If data are in .mat files,please reference 2.1.1.
The code of segmentation is the same,
```
X_win, groups_win = GR.segment_signal_unsupervised(DataX, groups, window_size, percentage)
```

### 2.3.6. Predict y label
Here, we got the best CNN models from 6 axes and 3 axis separately in the gituhub folder. So we can directly load the model, predict the y labels and plot them with signals.

```
y_predict = cnn_model.predict(X_win).round()

# remove the overlapping in windows, although it does not influence the results
X_rm_repeat = X_win[:, :100, :]
X = X_rm_repeat.reshape(-1, 3)
y_predict_final = np.repeat(y_predict, 100, axis=0)
groups_final = np.repeat(groups_win, 100, axis=0)
time_seconds = np.arange(len(X)) / fs

# plot
plt.figure(figsize=(19, 10))
plt.plot(time_seconds, X[:, 0])
plt.plot(time_seconds, y_predict_final[:, 1])
plt.title('Predicted Vertical Acceleration')  #accX
plt.legend(['Signal', 'Predicted Label (wk=1)'])
plt.xlabel("Seconds", fontsize=20)
plt.ylabel("Gravity Acceleration (9.8 m/s²)", fontsize=16)
plt.tick_params(axis='x', labelsize=15, labelcolor='black', pad=10)
plt.tick_params(axis='y', labelsize=15, labelcolor='black', pad=10)

plt.savefig(f'{PredictPNGDir}/VTacc_pred_y.png') # here,AccX is vertical acc
plt.savefig(f'{PredictSVGDir}/VTacc_pred_y.svg')
```

## 3. Description of Subfunctions
### 3.1. Functions in "GaitRecognitionFunctions_general.py"

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

[1] Our ADAPT paper after publishing  
[2] Bourke AK, Ihlen EAF, Bergquist R, Wik PB, Vereijken B, Helbostad JL. A physical activity reference data-set recorded from older adults using body-worn inertial sensors and video technology—The ADAPT study data-set. Sensors. 2017;17(3):559.  
[3] Felius RAW, Geerars M, Bruijn SM, et al. Reliability of IMU-Based Gait Assessment in Clinical Stroke Rehabilitation. Sensors (Basel) 2022;22(3).  

 


## Help

For questions about the algorithm and the implementations please contact: [y6.zhang@vu.nl](mailto:y6.zhang@vu.nl)
