# 读取文件数据,不断更新图表
import glob
from time import sleep

import numpy as np
import pandas as pd

from prototype import constant, global_tramsform2
from train import train_conv_lstm
from utils.show import real_time_show_phone_data


def real_time_show_file_data(file_name='../data/realworld/*/forearm_merged.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    all_data = np.zeros((128, 3), np.float32)
    all_transformed_data = np.zeros((128, 3), np.float32)

    # 遍历 dataframe，每次读取 128 行
    chunk_size = 128
    show_length = 512
    for start_row in range(0, len(big_df), chunk_size):
        data = big_df.iloc[start_row:start_row + chunk_size].values
        all_data = np.vstack([all_data, data[:, :3]])[-show_length:, :]
        transformed, rpy = global_tramsform2.transform_sensor_data_to_np2(data)

        pred = train_conv_lstm.apply_conv_lstm(transformed)

        # 4 基础分类(共通)
        pred_label = constant.Constant.realworld_x_uStudent.action_map_en_reverse.get(pred.item())

        # 8分类(realworld)
        # pred_label = constant.Constant.RealWorld.action_map_reverse.get(pred.item())
        ground_truth = constant.Constant.RealWorld.action_map_reverse.get(data[1, 9])
        all_transformed_data = np.vstack([all_transformed_data, transformed[:, :3]])[-show_length:, :]
        real_time_show_phone_data(all_data, all_transformed_data, pred_label, rpy, ground_truth= ground_truth)

        sleep(0.5)


if __name__ == '__main__':
    real_time_show_file_data()