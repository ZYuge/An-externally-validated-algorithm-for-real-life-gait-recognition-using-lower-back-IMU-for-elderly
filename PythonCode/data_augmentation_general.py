"""Unit code for data augmentation.
- Reference: Jongkuk Lim，lim.jeikei@gmail.com
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# In[2]: jittering


def jitter(data: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Add a random jittering.
    Args:
        x: sensor data, including many subjects' (3-dimensional，like cell in matlab)
        sigma: amount of the noise.
    Returns:
        x + normal random with sigma STD.
    """
    jitter_noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    data_noise = data + jitter_noise
    return data_noise


# In[2]: scaling
def scaling(data, sigma):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, data.shape[1]))
    scaling_values = np.matmul(np.ones((data.shape[0], 1)), scalingFactor)
    data_scaling = data * scaling_values
    return data_scaling


# [3] resampling walking frequency or sensor collecting frequency
# changing resampling frequency as fs*sigma, using interpolation method
# equal to in the same sampling time and sampling frequency, the smaller sigma has les sampling points but the same
# numbers of steps, so it looks like walking faster with less sigma under the same sampling frequency
# increasing sigma means lower walking speed or increasing sampling frequency
def resampling(data, y_binary, groups, fs, sigma):
    """ fs: the raw sampling frequency;
        sigma: the multiplier for data length scaling
        the one after interpolate is loc_wks * sigma.
    """
    t = np.arange(0, data.shape[0] / fs, 1 / fs)
    t_new = np.arange(0, data.shape[0] / fs, 1 / (fs * sigma))
    data_intrp = np.zeros((t_new.shape[0], data.shape[1]))
    for icol in np.arange(data.shape[1]):
        data_intrp[:, icol] = np.interp(t_new, t, data[:, icol])

    DataY_binary_intrp = find_new_loc_of_wk(data_intrp, y_binary, sigma)
    groups_intrp = find_new_loc_of_sub(data_intrp, groups, sigma)

    return data_intrp, DataY_binary_intrp, groups_intrp


def mark_wk(y_binary):
    # mark the location of walking in the original list
    loc_wk = np.array(np.where(y_binary == 1))
    diffloc = np.array(np.diff(loc_wk))
    ll = np.array(np.where(diffloc[0] > 1))
    loc_wke = loc_wk[0][ll]  # the end of walking episode
    loc_wks = loc_wk[0][ll + 1]  # the start of walking episode

    loc_wks2 = loc_wks.tolist()
    loc_wks2.insert(0, [loc_wk[0][0]])  # loc_wk[0][0] must be the list
    loc_wks = np.concatenate(loc_wks2[:])

    s = loc_wks[-1]
    e = len(y_binary)
    loc_wk_eind_episode = np.array(np.where(y_binary[s:e]==0))
    loc_wk_eind_episode = np.concatenate(loc_wk_eind_episode)

    if np.size(loc_wk_eind_episode) == 0:
        loc_wke_eind = e  # if not np.any(): np.any() =False, means empty; here not False, means empty and run this line
    else:
        loc_wke_eind = loc_wk_eind_episode[0] + s - 1
    e2 = loc_wke.shape[1]
    loc_wke = np.concatenate(loc_wke)
    loc_wke2 = loc_wke.tolist()
    loc_wke2.insert(e2, loc_wke_eind)
    loc_wke_final = np.array(loc_wke2)

    # loc_wke3 = loc_wke2 + [[loc_end]]  #
    # loc_wke = np.concatenate(loc_wke3[:])
    return loc_wks, loc_wke_final


def find_new_loc_of_wk(data_intrp, y_binary, sigma):
    loc_wks, loc_wke_final = mark_wk(y_binary)  # mark the start and end location of walking in original sequence
    loc_wks_new = [math.ceil(num) for num in (loc_wks * sigma)]  # Rounded to the maximum value as the start location
    loc_wke_new = [math.floor(num) for num in (loc_wke_final * sigma)]  # rounded to the minimum as the end location
    DataY_binary_intrp = np.zeros(data_intrp.shape[0])
    for irow in np.arange(len(loc_wks_new)):
        s = loc_wks_new[irow]
        e = loc_wke_new[irow]
        DataY_binary_intrp[s:e + 1] = np.ones(len(np.arange(s, e + 1)))  # includes the number "e" in wk label
    return DataY_binary_intrp


def find_new_loc_of_sub(data_intrp, groups, sigma):
    groups_intrp = np.zeros(data_intrp.shape[0])
    loc_sub_end = []
    for indx in np.unique(groups):
        ind = np.array(np.where(groups == indx))  # the location of each group
        loc_sub_end.append(ind[0][-1])
    loc_sub_end = np.array(loc_sub_end)
    loc_sub_end_new = [math.ceil(num) for num in
                       (loc_sub_end * sigma)]  # Rounded to the maximum value as the start location
    loc_sub_end_new = np.sort(loc_sub_end_new)
    loc_sub_end_new[-1] = data_intrp.shape[0]

    loc_sub_start = loc_sub_end_new[0:-1] + 1
    loc_sub_start_new = np.array([np.zeros(1), loc_sub_start], dtype=object)  # loc_wk[0][0] must be the list
    loc_sub_start_new = np.concatenate(loc_sub_start_new[:])

    subjects = np.unique(groups)

    for isub in np.arange(len(loc_sub_start_new)):
        s = int(loc_sub_start_new[isub])
        e = loc_sub_end_new[isub]
        # print(subjects[isub])
        groups_intrp[s:e] = np.repeat(int(subjects[isub]), len(np.arange(s, e)))  # includes e

    loc_zero = np.where(groups_intrp==0)
    loc_zero = np.array(loc_zero)
    if loc_zero.size != 0:
        for i in np.arange(len(loc_zero)):
            groups_intrp[loc_zero[i]] = groups_intrp[loc_zero[i]-1]

    return groups_intrp


def rotation(data, y, group, deg):
    """ rotation 6-axis data (3acc,3gr)
    data: the data is already subtracted the mean for each subject
    rotate 3 degrees on each axis respectively (3*3) in both acc and gyr:
        rotate x y z by 30 degrees, respectively;
        rotate x y z by 60 degrees, respectively;
        rotate x y z by 90 degrees respectively
    """

    data_rotation_final = []
    n = 0
    def inner_func(r, data):
        acc_rotation = r.apply(data[:, 0:3])
        gyr_rotation = r.apply(data[:, 3:6])
        data_rotation = np.hstack((acc_rotation, gyr_rotation))
        return data_rotation

    for indx in np.arange(len(deg)):
        rx = R.from_euler('x',deg[indx],degrees=True)
        ry = R.from_euler('y',deg[indx],degrees=True)
        rz = R.from_euler('z',deg[indx],degrees=True)
        # rzyx = R.from_euler('zyx', [deg[indx],deg[indx],deg[indx]], degrees=True)
        data_rot_x = inner_func(rx, data)
        data_rot_y = inner_func(ry, data)
        data_rot_z = inner_func(rz, data)
        # data_rot_zyx = inner_func(rzyx, data)
        data_rotation_final.append(data_rot_x)
        data_rotation_final.append(data_rot_y)
        data_rotation_final.append(data_rot_z)
        # data_rotation_final.append(data_rot_zyx)
        n = n+1

    data_rotation_final = np.array(data_rotation_final)
    data_rotation_final = np.concatenate(data_rotation_final)
    y_rot = n * 3 * list(y)  # here, 4 is three rotation methods x,y,z,zyx
    grp_rot = n * 3 * list(group)  # duplicate y_lables and no.subjects
    y_rot = np.array(y_rot)
    grp_rot = np.array(grp_rot)
    return data_rotation_final, y_rot, grp_rot


def rotation_acc(acc, y, group, deg):
    """ rotation 3acc
    data: the data is already subtracted the mean for each subject
    rotate 3 degrees on each axis respectively (3*3) in both acc and gyr:
        rotate x y z by 30 degrees, respectively;
        rotate x y z by 60 degrees, respectively;
        rotate x y z by 90 degrees respectively
    """

    data_rotation_final = []
    n = 0

    for indx in np.arange(len(deg)):
        rx = R.from_euler('x',deg[indx],degrees=True)
        ry = R.from_euler('y',deg[indx],degrees=True)
        rz = R.from_euler('z',deg[indx],degrees=True)
        acc_rot_x = rx.apply(acc[:, 0:3])
        acc_rot_y = ry.apply(acc[:, 0:3])
        acc_rot_z = rz.apply(acc[:, 0:3])
        data_rotation_final.append(acc_rot_x)
        data_rotation_final.append(acc_rot_y)
        data_rotation_final.append(acc_rot_z)
        n = n+1

    data_rotation_final = np.array(data_rotation_final)
    data_rotation_final = np.concatenate(data_rotation_final)
    y_rot = n * 3 * list(y)  # here, 3 is xyz
    grp_rot = n * 3 * list(group)  # duplicate y_lables and no.subjects
    y_rot = np.array(y_rot)
    grp_rot = np.array(grp_rot)
    return data_rotation_final,y_rot,grp_rot


# 3. Data Augmentation
# [1] adding noise
def data_with_noise(DataX, DataY_binary, groups,sigmas):
    DataX_noise = []
    DataY_binary_noise = []
    groups_noise = []
    for sigma in sigmas:
        """ loop sigma in 0.01 and 0.02, splice data with different noise sigma 
        """
        data_noise = jitter(DataX, sigma)
        DataX_noise.append(data_noise)
        DataY_binary_noise.append(DataY_binary)
        groups_noise.append(groups)
    DataX_noise = np.array(DataX_noise)
    DataY_binary_noise = np.array(DataY_binary_noise)
    groups_noise = np.array(groups_noise)

    DataX_noise_final = np.concatenate(DataX_noise[:])
    DataY_binary_noise_final = np.concatenate(DataY_binary_noise[:])
    groups_noise_final = np.concatenate(groups_noise[:])
    return DataX_noise_final, DataY_binary_noise_final, groups_noise_final


# [2]: scaling the data
def data_with_scaling(DataX, DataY_binary, groups, sigmas):
    DataX_scaling = []
    DataY_binary_scaling = []
    groups_scaling = []
    for sigma in sigmas:
        data_scaling = scaling(DataX, sigma)
        DataX_scaling.append(data_scaling)
        DataY_binary_scaling.append(DataY_binary)
        groups_scaling.append(groups)
    DataX_scaling = np.array(DataX_scaling)
    DataY_binary_scaling = np.array(DataY_binary_scaling)
    groups_scaling = np.array(groups_scaling)

    DataX_scaling_final = np.concatenate(DataX_scaling[:])
    DataY_binary_scaling_final = np.concatenate(DataY_binary_scaling[:])
    groups_scaling_final = np.concatenate(groups_scaling[:])
    return DataX_scaling_final, DataY_binary_scaling_final, groups_scaling_final


# [3]: resampling walking frequency or sensor collecting frequency
#     “Normal” walking speeds for community-dwelling healthy older adults range from 0.90 to 1.30 m/s,
#     speeds ≤0.60 to 0.70 m/s are strong risk factors for poor health outcomes, from doi: 10.2522/ptj.20100018
def data_with_intrp(DataX, DataY_binary, groups, fs, sigmas):
    DataX_intrp = []
    DataY_binary_intrp = []
    Groups_intrp = []
    for sigma in sigmas:
        data_intrp, act_labels, groups_intrp = resampling(DataX, DataY_binary, groups, fs, sigma)
        DataX_intrp.append(data_intrp)
        DataY_binary_intrp.append(act_labels)
        Groups_intrp.append(groups_intrp)
    DataX_intrp = np.array(DataX_intrp, dtype=object)
    DataY_binary_intrp = np.array(DataY_binary_intrp, dtype=object)
    Groups_intrp = np.array(Groups_intrp, dtype=object)

    DataX_intrp_final = np.concatenate(DataX_intrp[:])
    DataY_binary_intrp_fianl = np.concatenate(DataY_binary_intrp[:])
    groups_intrp_final = np.concatenate(Groups_intrp[:])
    return DataX_intrp_final, DataY_binary_intrp_fianl, groups_intrp_final
