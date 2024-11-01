# 读取文件数据,不断更新图表
import glob
from time import sleep

import numpy as np
import pandas as pd

from prototype import global_tramsform, constant
from train import train_conv_lstm
from utils.show import real_time_show_phone_data

# 实时展示长冈科技大学, 大学生的行为数据图表
def real_time_show_file_data(file_name = '../data/student/0726_lab/merge_labeled.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    big_df = big_df.iloc[:, 1:]

    big_df = big_df[big_df['label'] != 3]  # 去除蹲下
    big_df = big_df[big_df['label'] != -1]  # 去除None

    all_data = np.zeros((128, 3), np.float32)
    all_transformed_data = np.zeros((128, 3), np.float32)

    # 遍历 dataframe，每次读取 128 行
    chunk_size = 128
    show_length = 512
    for start_row in range(0, len(big_df), chunk_size):
        data = big_df.iloc[start_row:start_row + chunk_size].values
        all_data = np.vstack([all_data, data[:, :3]])[-show_length:, :]
        transformed, rpy = global_tramsform.transform_sensor_data_to_np(data)

        pred = train_conv_lstm.apply_conv_lstm(transformed)

        # 4 基础分类(共通)
        pred_label = constant.Constant.realworld_x_uStudent.action_map_en_reverse.get(pred.item())

        # pred_label = constant.Constant.RealWorld.action_map_reverse.get(pred.item())
        ground_truth = constant.Constant.uStudent.action_map_en_reverse.get(data[1, 9])
        all_transformed_data = np.vstack([all_transformed_data, transformed[:, :3]])[-show_length:, :]
        real_time_show_phone_data(all_data, all_transformed_data, pred_label, rpy, ground_truth= ground_truth)

        sleep(0.5)

if __name__ == '__main__':
    real_time_show_file_data()