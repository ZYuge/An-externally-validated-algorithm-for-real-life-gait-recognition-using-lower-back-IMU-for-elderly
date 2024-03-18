# This code is for preparing the external validating the existing models
# Here, we use the stroke patients datasets from walking and balance tests by Michiel Punt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


def load_data(filename, stt_del, end_del):
    """ To only reamin the data of movement,
        delete the looptest data of the first 30 and last 20 seconds(f=104)
        delete the balance data of the first 5 and last 10 seconds(f=104)
    """
    f = 104
    data = pd.read_table(filename)
    if data.empty:
        signal = []
    else:
        # Split String Column into several Columns
        nm_col = data.columns
        df = data[nm_col[0]].str.split(',',expand = True)
        if df.shape[1] == 1:
            df = data[nm_col[0]].str.split(';', expand=True)

        start = np.int(df[1][df[1]=='X'].index.values)+2 # the location of the start of signal
        data_selected = df.loc[start:,1:6].to_numpy() # dataframe to ndarrary, column 0 is timestamp
        if data_selected.shape[0] > (stt_del+end_del)*f:
            signal = np.zeros([data_selected.shape[0]-f*stt_del-f*end_del,6])
            for indx in np.arange(f*stt_del, data_selected.shape[0]-f*end_del):
                """the original values are string, so change string into float"""
                if len(data_selected[indx][0])==0 or data_selected[indx][0] == 'ERROR':
                    """skip rows which are empty or start with 'ERROR' """
                    pass
                else:
                    for col in np.arange(6):
                        signal[indx-f*stt_del][col] = np.float(data_selected[indx][col])
        else:
            signal = []
    return signal


def get_input(filenames,stt_del,end_del,type):
    signals = []
    y = []
    group = []
    filenames_use = []
    for indx in np.arange(len(filenames)):
        filename = filenames[indx]
        signal = load_data(filename,stt_del,end_del)
        if len(signal) != 0:
            filenames_use.append(filename)
            signals.append(signal)
            if type == 'loop':
                y.append(np.ones(signal.shape[0]))
            else:
                y.append(np.zeros(signal.shape[0]))

            # extract subjects'id from filename, put it into group
            l1 = filename.find('S')
            num_sub = filename[l1+1:l1+5]
            num_sub = np.int(num_sub)
            group.append(num_sub * np.ones(signal.shape[0]))

    signals = np.array(signals)
    signals = np.concatenate(signals)
    y = np.array(y)
    y = np.concatenate(y)
    group = np.array(group)
    group = np.concatenate(group)
    return signals, y, group

# loop
path = '/Users/yugezi/Desktop/1.1_ProjectVIBE_MP/3_ADAPT/5_MichielPunt_ExternalValitaionData/Raw files/looptest/'
filenames_wk = []
for root,dirs,files in os.walk(path):
    if root.find('Nee') != -1:
        """choose the data without using walkers"""
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                if os.path.splitext(file)[0].find('Onderrug') != -1:
                    filenames_wk.append(root+'/'+file)

signals_wk, y_wk, group_wk = get_input(filenames_wk,stt_del=30, end_del=20, type='loop')


# balance
path = '/Users/yugezi/Desktop/1.1_ProjectVIBE_MP/3_ADAPT/5_MichielPunt_ExternalValitaionData/Raw files/balans/'
filenames_sta = []
for root,dirs,files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
                filenames_sta.append(root+'/'+file)

signals_sta, y_sta, group_sta = get_input(filenames_sta,stt_del=5,end_del=10,type='balance')

plt.figure(1)
plt.plot(signals_wk[:,0:3])
plt.plot(group_wk)
plt.title('looptest')
plt.legend(['accx','accy','accz','groups'])
plt.savefig('looptest data & groups.png')

plt.figure(2)
plt.plot(signals_sta[:,0:3])
plt.plot(group_sta)
plt.legend(['accx','accy','accz','groups'])
plt.title('balance_test')
plt.savefig('balance_data & groups.png')


plt.figure(3)
plt.plot(signals_sta[:,0])
plt.title('balance (non-walking) signals_accX')
plt.savefig('balance_data.png')

plt.figure(4)
plt.plot(signals_wk[:,0])
plt.title('walking signals_accX')
plt.savefig('walking_data.png')


# save looptest & balance data as .py files
# default folder: /Users/yugezi/PycharmProjects/ADAPT-project/augmentation
np.savetxt('signals_sta.txt',signals_sta,fmt='%f3')
np.savetxt('signals_wk.txt',signals_wk,fmt='%f3')
np.savetxt('y_sta.txt',y_sta,fmt='%d')
np.savetxt('y_wk.txt',y_wk,fmt='%d')
np.savetxt('group_sta.txt',group_sta,fmt='%d')
np.savetxt('group_wk.txt',group_wk,fmt='%d')

with open('filenames_balance','wb') as fp:
    pickle.dump(filenames_sta, fp)

with open('filenames_wk', 'wb') as fp:
    pickle.dump(filenames_wk, fp)