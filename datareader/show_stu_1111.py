# データ表示
from time import sleep

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def show_data(file_name='../data/student/1111_lab/accelerometers.csv'):
    df = pd.read_csv(file_name)
    df = df[(df['UNIX_time(milli)'] >= 1731305400000)]

    df = df.iloc[:, 1:]

    all_data = np.zeros((10, 3), np.float32)

    # 遍历 dataframe，每次读取 10 行
    chunk_size = 10
    show_length = 100
    for start_row in range(0, len(df), chunk_size):
        data = df.iloc[start_row:start_row + chunk_size].values
        all_data = np.vstack([all_data, data[:, :3]])[-show_length:, :]

        real_time_show_raw_data(all_data)

        sleep(1)


def real_time_show_raw_data(float_matrix):
    plt.ion()  # interactive
    # data
    x_data = np.arange(float_matrix.shape[0])
    y1_data = float_matrix[:, 0]  # 第一列
    y2_data = float_matrix[:, 1]  # 第二列
    y3_data = float_matrix[:, 2]  # 第三列

    # Init plt
    if not hasattr(real_time_show_raw_data, 'initialized'):
        real_time_show_raw_data.fig, real_time_show_raw_data.ax = plt.subplots()
        plt.title('acc_data')
        real_time_show_raw_data.line1, = real_time_show_raw_data.ax.plot(x_data, y1_data, label='acc_x', color='red')
        real_time_show_raw_data.line2, = real_time_show_raw_data.ax.plot(x_data, y2_data, label='acc_y', color='green')
        real_time_show_raw_data.line3, = real_time_show_raw_data.ax.plot(x_data, y3_data, label='acc_z', color='blue')

        real_time_show_raw_data.ax.set_xlim(0, float_matrix.shape[0])
        real_time_show_raw_data.ax.set_ylim(np.min(float_matrix[:, :3]), np.max(float_matrix[:, :3]))
        real_time_show_raw_data.ax.legend()

        real_time_show_raw_data.initialized = True
    else:
        # repaint
        real_time_show_raw_data.line1.set_xdata(x_data)
        real_time_show_raw_data.line1.set_ydata(y1_data)
        real_time_show_raw_data.line2.set_xdata(x_data)
        real_time_show_raw_data.line2.set_ydata(y2_data)
        real_time_show_raw_data.line3.set_xdata(x_data)
        real_time_show_raw_data.line3.set_ydata(y3_data)

        # adjust limitation
        show_range_percent = 1.1  # 110%
        real_time_show_raw_data.ax.set_xlim(0, float_matrix.shape[0] * show_range_percent)

        real_time_show_raw_data.ax.set_title(f'acc_data,')

    plt.draw()  # paint
    plt.pause(0.01)  # short pause for data async


if __name__ == '__main__':
    show_data()
