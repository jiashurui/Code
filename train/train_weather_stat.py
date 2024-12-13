import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch

from train.conv_lstm import ConvLSTM

# 共通の緯度・経度グリッドを定義します
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)

file_path = '../data/weather/Data.csv'
df = pd.read_csv(file_path, index_col=0)
df.head()

# 沖縄周辺
selected_lat = common_lat[(common_lat >= 23.5) & (common_lat <= 29.5)]
selected_lon = common_lon[(common_lon >= 124.5) & (common_lon <= 130.5)]

# 日本列島
# selected_lat = common_lat[(common_lat >= 30.0) & (common_lat < 46.0)]
# selected_lon = common_lon[(common_lon >= 128.0) & (common_lon < 144.0)]


# build select index
selected_index = ['沖縄の天気', '沖縄の降水量']
for lat in selected_lat:
    for lon in selected_lon:
        selected_index.append(f"{lat}_{lon}")

df = df[selected_index]

Y1 = (df.iloc[:, 0] / 100).astype(int)
Y2 = df.iloc[:, 1]
X = df.iloc[:, 2:]

df[df.iloc[:, 0] >= 500].iloc[:, :2]
Y1[Y1 == 5] = 3
Y1[Y1 == 7] = 3

y1_train = Y1.iloc[:365]
y1_test = Y1.iloc[365:]
y2_train = Y2.iloc[:365]
y2_test = Y2.iloc[365:]
x_train = X.iloc[:365]
x_test = X.iloc[365:]

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def slide_window(input_list, window_size, over_lap):
    all_data = []
    stat_point = 0
    end_point = stat_point + window_size
    stride = int(window_size * over_lap)
    while True:
        if end_point >= len(input_list) - 1:
            break
        all_data.append(input_list[stat_point: end_point])
        stat_point += stride
        end_point += stride
    return all_data

x_train_window = slide_window(x_train, window_size=8, over_lap=0.5)
x_test_window = slide_window(x_test, window_size=8, over_lap=0.5)



def total_differences_by_layer(pressures):
    d, m, n = pressures.shape
    total_differences = np.zeros(d)  # 创建一个一维数组，用于存储每层的差值总和

    # 遍历每个二维层面
    for k in range(d):
        layer_difference = 0  # 用于存储当前层的差值总和
        for i in range(m):
            for j in range(n):
                current_value = pressures[k, i, j]

                # 检查上方邻居（在同一层面）
                if i > 0:
                    layer_difference += abs(current_value - pressures[k, i - 1, j])
                # 检查下方邻居（在同一层面）
                if i < m - 1:
                    layer_difference += abs(current_value - pressures[k, i + 1, j])
                # 检查左方邻居（在同一层面）
                if j > 0:
                    layer_difference += abs(current_value - pressures[k, i, j - 1])
                # 检查右方邻居（在同一层面）
                if j < n - 1:
                    layer_difference += abs(current_value - pressures[k, i, j + 1])

        total_differences[k] = layer_difference  # 将计算得到的当前层差值总和存储到数组中

    return total_differences

def calculate_daily_mean_gradient(data):
    num_days, m, n = data.shape
    daily_gradients_mean = np.zeros(num_days)  # 存储每天梯度平均值的数组

    for day in range(num_days):
        dx = np.zeros((m, n))
        dy = np.zeros((m, n))

        # 计算x方向的梯度
        dx[:, 1:-1] = (data[day, :, 2:] - data[day, :, :-2]) / 2

        # 计算y方向的梯度
        dy[1:-1, :] = (data[day, 2:, :] - data[day, :-2, :]) / 2

        # 计算梯度的模长并取平均
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        daily_gradients_mean[day] = np.mean(gradient_magnitude)

    return daily_gradients_mean

def calc_feature(data):
    day_mean = np.mean(x_train, axis=1)
    day_min = np.min(x_train, axis=1)
    day_max = np.max(x_train, axis=1)
    day_median = np.median(x_train, axis=1)
    day_std = np.std(x_train, axis=1)
    day_var = np.var(x_train, axis=1)

    # 计算空间平均气压差值
    average_differences = total_differences_by_layer(x_train.reshape(365, 13, 13))
    average_gradient = calculate_daily_mean_gradient(x_train.reshape(365, 13, 13))


    features = np.stack([day_mean,day_min,day_max,day_median,day_median,day_std,day_var,average_differences,average_gradient], axis=1)

    return features


X_train = calc_feature(x_train)
X_test = calc_feature(x_test)

# 定义 SVM 模型和参数搜索范围，包含多种核函数
param_grid = {
    'C': [0.1, 1, 10, 50],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],  # 多项式核的阶数
    'coef0': [0.0, 0.1, 0.5]  # sigmoid 和 poly 核函数的独立项系数
}

svc = SVC()
clf = GridSearchCV(svc, param_grid, scoring='accuracy', cv=5)

# 执行网格搜索以找到最佳参数
clf.fit(X_train, y1_train)

# 输出最佳参数和最佳得分
print("Best Param:", clf.best_params_)
print("Best Score (Cross Validation):", clf.best_score_)

# 用最佳模型预测训练集和测试集
best_model = clf.best_estimator_

# 训练集评估
y_pred_train = best_model.predict(X_train)
train_accuracy = accuracy_score(y1_train, y_pred_train)
train_cm = confusion_matrix(y1_train, y_pred_train)
print("\nTrain Confusion Matrix:\n", train_cm)
print("Train Accuracy:", train_accuracy)