import glob

import pandas as pd
from hmmlearn import hmm
import numpy as np

from datareader.realworld_datareader import get_realworld_raw_for_abnormal
from prototype.global_tramsform import transform_sensor_data_to_np
from prototype.global_tramsform2 import transform_sensor_data_to_np2
from utils.uci_datareader import get_data_1d_uci_all_data


# 加载数据


def read_data_2023(file_name='../data/child/2023_03/merged_data/*.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    big_df = big_df.iloc[:, 1:]
    big_df, _ = transform_sensor_data_to_np(big_df.values)

    return big_df[:, :6]

def read_data_2024(file_name='../data/child/2024_04/toyota_202404_crossing/*/*/*.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    big_df = big_df.iloc[:, 1:10]
    big_df, _ = transform_sensor_data_to_np(big_df.values)
    return big_df[:,:6]
def read_data_stu(file_name = '../data/student/0726_lab/merge_labeled.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    # 过滤掉 -1 标签(未参与实验的数据)
    big_df = big_df[big_df['label'] == 5]
    big_df = big_df.iloc[:, 1:]

    big_df, _ = transform_sensor_data_to_np(big_df.values)

    return big_df[:,:6]

def read_data_real(file_name='../data/realworld/*/chest_merged.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    big_df = big_df[big_df['label'] == 7]
    big_df = big_df.iloc[:, 1:]

    big_df, _ = transform_sensor_data_to_np2(big_df.values)

    return big_df[:,:6]

train_normal = read_data_2023()
real_normal = read_data_2024()
real_stu = read_data_stu()
real_data = read_data_real()
# 定义 HMM 模型
n_components = 10 # 假设有6个隐状态
model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=2000)


# 展平数据并训练 HMM
def train_hmm(train_data, model):
    # 训练 HMM 模型
    model.fit(train_data)


# 计算对数似然
def calculate_average_log_likelihood(data, model):
    total_log_likelihood = 0
    log_likelihood = model.score(data)
    total_log_likelihood += log_likelihood
    return total_log_likelihood / len(data)


# 训练 HMM 模型
train_hmm(train_normal, model)

# 对 test_normal 和 test_abnormal 数据集进行预测并计算平均似然
train_normal_result = calculate_average_log_likelihood(train_normal, model)

real_normal_result = calculate_average_log_likelihood(real_normal, model)
stu_normal_result = calculate_average_log_likelihood(real_stu, model)

realworld_normal_result = calculate_average_log_likelihood(real_data, model)

print(f"Train Normal 平均似然: {train_normal_result}")
print(f"Real Normal 平均似然: {real_normal_result}")
print(f"Stu Normal 平均似然: {stu_normal_result}")
print(f"Realworld Normal 平均似然: {realworld_normal_result}")
