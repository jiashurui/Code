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


    fig, ax = plt.subplots()
    plt.plot(data_wrist.iloc[0:100, 1], label='x')
    plt.plot(data_wrist.iloc[0:100, 2], label='y')
    plt.plot(data_wrist.iloc[0:100, 3], label='z')
    # 设置图例
    ax.legend()
    plt.show()

get_data_1d_cha()
