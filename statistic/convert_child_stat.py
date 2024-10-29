#  本文件的作用是将儿童行走的数据转换为特征量
import glob

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from anormal.t_SNE import plot_data_pca, plot_data_tSNE
from prototype.constant import Constant
from utils.slidewindow import slide_window2


def convert_data(file_name = '../data/child/2023_03/merged_data/*.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)

        groups = list(data.groupby('X'))

        for group in groups:
            label = Constant.ChildWalk.action_map.get(group[0])
            data = group[1]

            # 10hz
            data_list = slide_window2(data, 50, 0.5)

            feature_of_this_file = []
            for frame in data_list:
                # 50个时间步(5s) 的统计特征
                feature = calc_df_features(frame.iloc[:,1:10])
                feature['label'] = label

                feature_of_this_file.append(feature)
            appended_data.extend(feature_of_this_file)

    big_df = pd.concat(appended_data, ignore_index=True)
    big_df.to_csv('../data/child/2023_03/features/merged_data_features.csv', index=False)

    return big_df


# 返回: XYZ(轴) * 3种信号数据(加速度,角速度,磁力) * 10维度特征
# Series(90)
def calc_df_features(df):

    features_name_list = ['mean','min','max','median','std','cv','skew','kurtosis','signal_power','rms']
    singal_name_list = ['accx','accy','accz','gyrox','gyroy','gyroz','magx','magy','magz']
    features_list = []

    df_mean = df.mean()
    features_list.extend([item + '_' + features_name_list[0] for item in singal_name_list])
    df_min = df.min()
    features_list.extend([item + '_' + features_name_list[1] for item in singal_name_list])
    df_max = df.max()
    features_list.extend([item + '_' + features_name_list[2] for item in singal_name_list])
    df_median = df.median()
    features_list.extend([item + '_' + features_name_list[3] for item in singal_name_list])
    df_std = df.std()
    features_list.extend([item + '_' + features_name_list[4] for item in singal_name_list])
    # 计算变异系数 (CV)
    cv = df.std() / df.mean()
    features_list.extend([item + '_' + features_name_list[5] for item in singal_name_list])
    # 偏度(Skewness)
    skewness = df.apply(lambda x: skew(x))
    features_list.extend([item + '_' + features_name_list[6] for item in singal_name_list])
    # 峰度(Kurtosis)
    kurt = df.apply(lambda x: kurtosis(x))
    features_list.extend([item + '_' + features_name_list[7] for item in singal_name_list])
    # 信号功率 (Signal Power)
    signal_power = df.apply(lambda x: np.mean(x ** 2))
    features_list.extend([item + '_' + features_name_list[8] for item in singal_name_list])
    # 二乘平方根 (Root Mean Square, RMS)
    rms = df.apply(lambda x: np.sqrt(np.mean(x ** 2)))
    features_list.extend([item + '_' + features_name_list[9] for item in singal_name_list])

    df_stat = pd.concat([df_mean, df_min, df_max, df_median, df_std, cv, skewness, kurt, signal_power, rms], axis=0)
    df = pd.DataFrame([df_stat.values], columns=features_list)

    return df
def show_origin():
    file = '../data/child/2023_03/merged_data/*.csv'

    file_list = glob.glob(file)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    data = big_df.iloc[:, 1:10].values
    labels = big_df['X'].values

    plot_data_pca(data, labels, Constant.ChildWalk.action_map_en)
    plot_data_tSNE(data, labels, Constant.ChildWalk.action_map_en)


def show_pca():
    file = glob.glob('../data/child/2023_03/features/merged_data_features.csv')
    df = pd.read_csv(file[0])

    data = df.iloc[:, :90].values
    labels = df.iloc[:, 90].values

    plot_data_pca(data, labels, Constant.ChildWalk.action_map_en_reverse)
    plot_data_tSNE(data, labels, Constant.ChildWalk.action_map_en_reverse)

if __name__ == '__main__':
    show_origin()