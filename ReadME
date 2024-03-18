This algorithm was developed by Y. Zhang on 18/03/2024 in collaboration with the Norwegian University of Science and Technology, Utrecht University of Applied Sciences, and Vrije Universiteit Amsterdam.

# Recognize real-world gait episodes based on deep learning methods

This script is written to recognize real-world gait episodes of healthy and stroked old adults (OA) based on inertial measurement units (IMU) data by using a convolutional neural network (CNN). The stroke OA has a very slow and poor walking pattern. This repository contains the Python code for training and using the  CNN model as presented in **XXX (paper link).**

## How to use the algorithm

Make sure all dependencies are correctly installed. python version > 3.6. Place the data from the IMU files (low back) in the data folder. Run the Main.py file. The results of model performance will appear in the folder "CNN_models_results". The initial weights of models and the number of windows for split datasets will be stored in the folder "CNN_models_info". The trained model in ".h5" will be saved in the "CNN_models_save" folder.

### Usage

The columns of input "DataX" are [3-axis acceleration, 3-axis gyroscope] or only [3-axis acceleration].

For datasets of training and externally validating models, "DataY" and "groups" has the corresponding activity labels and subject numbers on each sampling point of DataX, respectively.

- ##### To train the model

We use the ADAPT dataset to train the model. The ADAPT dataset is a thoroughly collected IMU dataset on OA by Bourke et al, which includes semi-structured supervised and free-living unsupervised situations both with manually annotated labels based on video data.  

The code for the training model is shown below:

```
python /Main.py 
```

The pipeline of this code is shown as the below

![flow chart_ADAPT](/Users/yugezi/Desktop/1.1_ProjectVIBE_MP/3_ADAPT/2_Draft/Figures/Flow chart/flow chart_ADAPT.png)



- ##### To externally validate the model

  ```
  python /External_validate_model.py
  ```

![flow chart_external data](/Users/yugezi/Desktop/1.1_ProjectVIBE_MP/3_ADAPT/2_Draft/Figures/Flow chart/flow chart_external data.png)

- ##### To predict unknown activities

For model training, the columns of input signals are 3-axis acceleration, 3-axis gyroscope, 3-axis magnitude data, and activity labels.

```
python /Recognize_gait_unsupervised.py
```

- ##### To augment the dataset

```
python /data_augmentation_general.py
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



### CNN Model structure

![image-20240318212852516](/Users/yugezi/Library/Application Support/typora-user-images/image-20240318212852516.png)

### Results

DA is data augmentation.

Below results are the average results after running 30 times for non DA version and 10 times for DA version.

![image-20240318214133800](/Users/yugezi/Library/Application Support/typora-user-images/image-20240318214133800.png)



![image-20240318214327438](/Users/yugezi/Library/Application Support/typora-user-images/image-20240318214327438.png)

![image-20240318214333577](/Users/yugezi/Library/Application Support/typora-user-images/image-20240318214333577.png)



### References

[1] Bourke AK, Ihlen EAF, Bergquist R, Wik PB, Vereijken B, Helbostad JL. A physical activity reference data-set recorded from older adults using body-worn inertial sensors and video technology—The ADAPT study data-set. Sensors. 2017;17(3):559.



## Help

For questions about the algorithm and the implementations please contact: [y6.zhang@vu.nl](mailto:y6.zhang@vu.nl)
