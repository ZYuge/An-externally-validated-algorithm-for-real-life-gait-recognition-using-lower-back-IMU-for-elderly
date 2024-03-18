import matplotlib.pyplot as plt

def plot_data_all(data_all, i):
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(data_all[i][:, 0:3])
    plt.ylabel('acc')
    plt.legend(['x', 'y', 'z'])
    plt.suptitle('signals for data_all' + str(i))

    plt.subplot(3, 1, 2)
    plt.plot(data_all[i][:, 3:6])
    plt.ylabel('gyro')
    plt.legend(['x', 'y', 'z'])

    plt.subplot(3, 1, 3)
    plt.plot(data_all[i][:, 6:9])
    plt.ylabel('mag')
    plt.legend(['x', 'y', 'z'])
    plt.show()

    plt.figure(2)
    plt.plot(data_all[i][:, 0:3])
    plt.ylabel('acc')
    plt.plot(data_all[i][:, 9])
    plt.suptitle('activity labels for acc P' + str(i))


def plot_signal(DataX, DataY, groups, isub):
    """ data is DataX, with only 6 axes: 3acc and 3gry
        groups is the subject labels for all sample points
        specific subject: isub"""
    loc = np.array(np.where(groups == isub))
    loc = loc.transpose()
    start_point = int(loc[0])
    end_point = int(loc[-1])

    plt.subplot(2, 1, 1)
    plt.plot(DataX[start_point:end_point, 0:3])
    plt.plot(DataY[start_point:end_point])
    plt.ylabel('acc')
    plt.legend(['x', 'y', 'z'])

    plt.subplot(2, 1, 2)
    plt.plot(DataX[start_point:end_point, 3:6])
    plt.plot(DataY[start_point:end_point])
    plt.ylabel('gyro')
    plt.legend(['x', 'y', 'z'])
    plt.suptitle('signals for subject P' + str(isub))


