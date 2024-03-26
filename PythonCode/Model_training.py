# load variable 'signal' in each .mat file 
# For the ADAPT dataset, 1-11 columns in each signal: 1-3 ACC xyz;4-6 GYR xyz; 7-9 MAG xyz; 10 activity Labels; 11 Time
import matplotlib.pyplot as plt
import GaitRecognitionFunctions_general as GR
import data_augmentation_general as DA
import numpy as np
import pandas as pd
import openpyxl
import os


# %% 0.1. subfunctions
def delete_useless_label(DataX,DataY,DataY_binary,groups):
    """
    delete activities "0=none" and "11=undefined" to save memory
    """
    loc_un = np.concatenate(np.array(np.where(DataY == 11)))
    loc_no = np.concatenate(np.array(np.where(DataY == 0)))
    row_del = np.sort(np.hstack([loc_un,loc_no]))
    DataX_new = np.delete(DataX,row_del,0)
    DataY_binary_new = np.delete(DataY_binary,row_del,0)
    groups_new = np.delete(groups,row_del,0)
    return DataX_new, DataY_binary_new, groups_new


def use_data_augmentation(x, y, groups, fs, augmentation_methods=None, input_axis=None):
    """ Determine whether use the Data Augmentation methods and which methods
        The magnitude number of DA can be adjusted
    :return: The final data before putting model [raw data +/- DA]

    """
    augmented_data = []

    # Apply selected augmentation methods
    if 'noise' in augmentation_methods:
        x_noise, y_noise, groups_noise = DA.data_with_noise(x, y, groups, [0.01, 0.02])
        augmented_data.extend([x_noise, y_noise, groups_noise])

    if 'scaling' in augmentation_methods:
        x_scaling, y_scaling, groups_scaling = DA.data_with_scaling(x, y, groups, [0.3, 0.4])
        augmented_data.extend([x_scaling, y_scaling, groups_scaling])

    if 'interpolation' in augmentation_methods:
        x_intrp, y_intrp, groups_intrp = DA.data_with_intrp(x, y, groups, fs, [0.9, 1.1])
        augmented_data.extend([x_intrp, y_intrp, groups_intrp])

    if 'rotation' in augmentation_methods:
        if input_axis == 3:
            x_rot, y_rot, groups_rot = DA.rotation_acc(x, y, groups, [90])
        elif input_axis == 6:
            x_rot, y_rot, groups_rot = DA.rotation(x, y, groups, [90])
        else:
            print("Undefined Input Axis")
        augmented_data.extend([x_rot, y_rot, groups_rot])

    # Combine all augmented data
    X_all = np.concatenate([x] + augmented_data[::3], axis=0)
    y_all = np.concatenate([y] + augmented_data[1::3], axis=0)
    groups_all = np.concatenate([groups] + augmented_data[2::3], axis=0)
    return X_all, y_all, groups_all


def save_split_datasets_info(OutputDir):
    filepath = os.path.join(OutputDir,'file_groups.txt')
    with open(filepath, 'a') as file:
        print(f'Repeat{i}',file=file)
        print(f"Train_group = {np.unique(groups_train)}", file=file)
        print(f"Test_group ={np.unique(groups_test)}", file=file)
        print(f"Validate_group ={np.unique(groups_val)}", file=file)


def save_dataframe(data, output_path, filename):
    data = pd.concat(data, ignore_index=True)  # remain only one header
    output_xlsx_path = os.path.join(output_path, filename)
    if os.path.isdir(output_xlsx_path):
        output_xlsx_path = os.path.join(output_xlsx_path)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    workbook.save(output_xlsx_path)
    data.to_excel(output_xlsx_path, index=False)


# %% 0.2. Some settings
repeats = 2  # repeat times of the model
fs = 100  # raw sampling frequency
wk_label = 1
window_size = 200
percentage = 0.5
Nfolds = 5         # train:test = 4:1
Nfolds_val = 3     # train:validation = 2:1
input_axis = 6     # 3-> only acc, 6-> acc&gyr
augmentation_methods = ['rotation']
# augmentation_methods = None

InputDataDir = './github_rwk/InputData/'
ModelResultDir = "./github_rwk/CNN_models_results"
ModelsSaveDir = "./github_rwk/CNN_models_save"
ModelsInfoDir = "./github_rwk/CNN_models_info"

if not os.path.exists(ModelResultDir):
    os.makedirs(ModelResultDir)
if not os.path.exists(ModelsSaveDir):
    os.makedirs(ModelsSaveDir)
if not os.path.exists(ModelsInfoDir):
    os.makedirs(ModelsInfoDir)

# %% 1. load the data (input_axis3->only acc, 6-> acc&gyr) and binary activity labels (walking1,non-walking0)
DataX, DataY, DataY_binary, groups, filenames, subject_number = GR.load_matfiles(InputDataDir, wk_label, input_axis)
DataX_new, DataY_binary_new, groups_new = delete_useless_label(DataX, DataY, DataY_binary, groups)


# %% 2. Run the model
# %%    Steps: split train_val&test, split train&val, +/-DA, run the model(segment, down-sample, fit model,evaluate)
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
