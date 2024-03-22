This algorithm was developed by Y. Zhang on 18/03/2024 in collaboration with the Norwegian University of Science and Technology, Utrecht University of Applied Sciences, and Vrije Universiteit Amsterdam.

# Recognize real-world gait episodes based on deep learning methods

We developed a convolutional neural network (CNN) to recognize real-world gait based on inertial measurement units (IMU) data and the CNN model worked perfectly on older adults (≥ 65 years old) and stroke patients who can walked without aids. Therefore, our developed CNN model are suitable for older people who walk slowly, as presented in paper **XXX (paper link).**


![Model Structure](images/Model%20Structure.png)
** Figure1. CNN Model structure **

## CNN Model performance

## How to use the algorithm
This repository contains the python code for below aims. 

1) train a CNN model,
```
Model_training.py
```
2) validat externally the model
```
External_validate_model.py
```
3) predict the unknown activities.
```
Recognize_gait_unsupervised.py
```

In addition, the repository contains function code for training and data augmentation.
```
GaitRecognitionFunctions_general.py
data_augmentation_general.py
```

You can select the code according to your need.
|Aim |with true activity labels| code | Use existing models|
| ---|----------------------- | -----------| ----------|
|Trian a model| Yes | Model_training.py | No |
|Validate externally| Yes | External_validate_model.py | Yes |
|predict unknown data| No | Recognize_gait_unsupervised.py | Yes |


## Aim1: train a CNN model

 Run the Main.py file. The results of model performance will appear in the folder "CNN_models_results"[SMB: should we make the folder where results are stored an input variable?]. The initial weights of models and the number of windows for split datasets will be stored in the folder "CNN_models_info". The trained model in ".h5" will be saved in the "CNN_models_save" folder [SMB: but this is only if you train the model, right? Otherwise, it will not be?].

### 1) Install the necessary packages
First step, make sure install all necessary packages in your python environment with python version > 3.6.

### 2) Folders and data preparation
Input forlder
Output folder
Secondly, place the data from the IMU files (low back) in the data folder
 [SMB: I see in the code now that everything is pointing to something like "/yuge/etc etc etc". make sure that this is not the case by using relative or system paths].

### 3）Usage

The columns of input "DataX" are [3-axis acceleration, 3-axis gyroscope] or only [3-axis acceleration].[SMB; specify colums; does it matter which one is AP, ML, VT? or not?]

For training and externally validating models, "DataY" and "groups" have the corresponding activity labels and subject numbers on each sampling point of DataX, respectively.

- ##### To train the model

We used the ADAPT dataset to train the model. The ADAPT dataset is a IMU dataset collected on older adults by Bourke et al, which includes semi-structured supervised and free-living unsupervised situations both with manually annotated labels based on video data.  

The code for the training model is shown below:

```
PythonCode /Main.py [SMB: but you say that this is the code to run the model on your own data, but here you say that this iss the code used to train? Shouldnt there be two seperate main files?]
```

The pipeline of this code is shown as the below

![Flow Chart_ADAPT](images/flow%20chart_ADAPT.png)



- ##### To externally validate the model

```
PythonCode /External_validate_model.py
```

![flow chart_external data](images/flow%20chart_external%20data.png)

- ##### To predict unknown activities

For model training [SMB: but this piece is about prediction, not trainnig?, the columns of input signals are 3-axis acceleration, 3-axis gyroscope, 3-axis magnitude data, and activity labels.

```
PythonCode /Recognize_gait_unsupervised.py
```

- ##### To augment the dataset

```
PythonCode /data_augmentation_general.py
```

the hyperparameters of data augmentation are show as below: --> table

| augmentation methods       | recommended scale |
| -------------------------- | ----------------- |
| Jitter                     | [0.01, 0.02]      |
| scaling                    | [0.3,0.4]         |
| Resampling (interpolation) | [0.9,1.1]         |
| Rotation                   | [90°]             |



### Parameters of our model

window size=200 

sampling points of the ADAPT data =100, external data = 104

Epochs = 30

Batch_size = 32

Filters = 64 

Kernel_size = 3




### Results

DA is data augmentation. [SMB: I would not use abbreviations; really no need for that]

Below results are the average results after running 30 times for non DA version and 10 times for DA version.

![Model performance_external dataset](/images/Model%20performance_external%20dataset.png)



![Model performance_testing dataset](images/Model%20performance_testing%20dataset.png)

![Model performance_validation dataset](images/Model%20performance_validation%20dataset.png)



### References

[1] Bourke AK, Ihlen EAF, Bergquist R, Wik PB, Vereijken B, Helbostad JL. A physical activity reference data-set recorded from older adults using body-worn inertial sensors and video technology—The ADAPT study data-set. Sensors. 2017;17(3):559.



## Help

For questions about the algorithm and the implementations please contact: [y6.zhang@vu.nl](mailto:y6.zhang@vu.nl)
