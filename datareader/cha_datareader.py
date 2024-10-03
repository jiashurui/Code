import glob
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from utils.slidewindow import slide_window2


def get_data_1d_cha():
    file_pocket = '../data/CHA/smartphoneatpocket.csv'
    file_wrist = '../data/CHA/smartphoneatwrist.csv'

    data_pocket = pd.read_csv(file_pocket, header=None)
    data_wrist = pd.read_csv(file_wrist, header=None)

    acc_x = np.array(data_pocket[4])
    acc_y = np.array(data_pocket[5])
    acc_z = np.array(data_pocket[6])

    combined_array = np.column_stack((acc_x, acc_y, acc_z))
    data_sliced_list = slide_window2(combined_array, 200, 0.5)
    random.shuffle(data_sliced_list)

    test_data = torch.tensor(np.array(data_sliced_list), dtype=torch.float32).transpose(1, 2)
    labels = torch.zeros(test_data.shape[0], dtype=torch.long)
    return test_data,labels

    # fig, ax = plt.subplots()
    # plt.plot(data_wrist.iloc[0:100, 1], label='x')
    # plt.plot(data_wrist.iloc[0:100, 2], label='y')
    # plt.plot(data_wrist.iloc[0:100, 3], label='z')
    # 设置图例
    # ax.legend()
    # plt.show()

get_data_1d_cha()
