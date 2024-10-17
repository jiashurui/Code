import glob
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils.slidewindow import slide_window2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(-1, 1))

def get_data_1d_cha():
    file_pocket = '../data/CHA/smartphoneatpocket.csv'
    file_wrist = '../data/CHA/smartphoneatwrist.csv'

    data_pocket = pd.read_csv(file_pocket, header=None)
    data_wrist = pd.read_csv(file_wrist, header=None)

    data_pocket = data_pocket[data_pocket[13] == 11112]

    acc_x = np.array(data_pocket[4])
    acc_y = np.array(data_pocket[5])
    acc_z = np.array(data_pocket[6])

    combined_array = np.column_stack((acc_x, acc_y, acc_z))
    data_sliced_list = slide_window2(combined_array, 200, 0.5)
    random.shuffle(data_sliced_list)

    test_data = torch.tensor(np.array(data_sliced_list), dtype=torch.float32).transpose(1, 2)
    labels = torch.zeros(test_data.shape[0], dtype=torch.long)

    fig, ax = plt.subplots()
    plt.plot(data_pocket.iloc[0:100, 1], label='x')
    plt.plot(data_pocket.iloc[0:100, 2], label='y')
    plt.plot(data_pocket.iloc[0:100, 3], label='z')
    #设置图例
    ax.legend()
    plt.show()
    return test_data,labels


def get_cha_data_for_abnormal(slide_window , features_num):
    file_pocket = '../data/CHA/smartphoneatpocket.csv'
    # file_wrist = '../data/CHA/smartphoneatwrist.csv'

    data_pocket = pd.read_csv(file_pocket, header=None)
    # data_wrist = pd.read_csv(file_wrist, header=None)

    # 归一化
    # data_pocket.iloc[:, 1:13] = scaler.fit_transform(data_pocket.iloc[:, 1:13])
    #
    # acc_x = np.array(data_pocket[1])
    # acc_y = np.array(data_pocket[2])
    # acc_z = np.array(data_pocket[3])
    # label = np.array(data_pocket[13])
    #
    # combined_array = np.column_stack((acc_x, acc_y, acc_z, label))

    data_sliced_list = slide_window2(data_pocket, slide_window, 0.5)
    random.shuffle(data_sliced_list)

    data_tensor = torch.tensor(np.array(data_sliced_list), dtype=torch.float32).to(device)

    # 按照标签,分割出来数据
    condition_walk = data_tensor[:, :, 13] != 11116  # walk
    condition_stand = data_tensor[:, :, 13] == 11116  # stand

    tensor_walk = data_tensor[condition_walk[:, 0]]
    tensor_not_walk = data_tensor[condition_stand[:, 0]]

    return tensor_walk[:, :, :features_num], tensor_not_walk[:, :, :features_num]

if __name__ == '__main__':
    t , f = get_cha_data_for_abnormal(128, 6)
    print()
    # normal_data, stand_data = get_cha_data_for_abnormal(128)
    # print(normal_data.shape)