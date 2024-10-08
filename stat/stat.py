import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prototype.constant import Constant


def show_mh_hist_stat():
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name,sep='\t',header=None)
        data.columns = Constant.mHealth.data_columns
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    # forearm data
    big_df = big_df.iloc[:, 14:24]
    # 创建 1 行 3 列的子图，图像大小为 18x5
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 绘制每个子图的直方图
    data_list = [big_df['arm_x'], big_df['arm_y'], big_df['arm_z']]
    titles = ['X', 'Y', 'Z']

    for i, ax in enumerate(axes):
        ax.hist(data_list[i], bins=30, color='blue', edgecolor='black', alpha=0.7)
        ax.set_title(f'mHealth arm acceleration {titles[i]}')
        ax.set_xlabel('Value(m/s2)')  # 设置每个子图的 X 轴标签
        ax.set_ylabel('Number')  # 设置每个子图的 Y 轴标签

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_mh_hist_stat()