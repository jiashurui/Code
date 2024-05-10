import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from utils.slidewindow import slide_window2
import random

class Simple1DCNN(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(Simple1DCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)  # 使用最大池化层进行下采样
        self.fc = torch.nn.Linear(32 * 50, 7)  # 输出大小调整为与标签相匹配
    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 示例数据
channel = 1 # 输入通道数
slide_window_length = 100  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 20
batch_size = 64  # 或其他合适的批次大小

# 创建示例输入数据 TODO
file_list = glob.glob('../data/realworld/1/acc_walking_*.csv')
final_data = []

# make label by fileName (walking)
# chest 1 forearm 2 head 3 shin 4 thigh 5 upperarm 6 waist 7
label_map = {
             'waist' :  0,
             'chest' :  1,
             'forearm': 2,
             'head':    3,
             'shin':    4,
             'thigh':   5,
             'upperarm':6
             }
for file_name in file_list:
    data = pd.read_csv(file_name)
    # 对于每一个dataframe , 按照文件名给其打上标签
    matched_substrings = [label for label in label_map.keys() if label in file_name]

    if not matched_substrings or len(matched_substrings) != 1:
        raise KeyError("无法生成标签")
    else:
        data['label'] = label_map.get(matched_substrings[0])
    ########################################################
    # 按照行处理数据
    # 数据处理(合并特征到1维度)
    # TODO 判断初始手机朝向, 数据转换(暂时先试试不用转换能不能做)

    # 滑动窗口平均噪声
    data.rolling(window=3).mean()

    data_agg = pd.DataFrame(columns =['xyz','label'])
    # 数据合并
    for index, row in data.iterrows():
        magnitude = np.sqrt(row['attr_x'] ** 2 + row['attr_y'] ** 2 + row['attr_z'] ** 2)
        # new_row = pd.Series({'xyz': magnitude, 'label': data['label']})
        # data_agg = pd.concat([data_agg, new_row])
        data['xyz'] = magnitude
    #######################################################################

    # 分割后的数据 100个 X组
    data_sliced = slide_window2(data, slide_window_length,0.5)

    # 对于每一个dataframe , 滑动窗口分割数据

    final_data.extend(data_sliced)

# [df,df,df,]
# df: attr1,attrx,attry,attrz,
# print(final_data)

# 取出来的df 数据,

# df to tensor

# tensor_data = torch.tensor(final_data, dtype=torch.float32)

# simple shuffle
random.shuffle(final_data)

# 提取输入和标签
input_features = np.array([df['xyz'].values for df in final_data])
labels = np.array([df['label'].values for df in final_data])[:,0]

# 将NumPy数组转换为Tensor
inputs_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels_tensor = torch.tensor(labels, dtype=torch.long)


print(inputs_tensor.shape)
print(labels_tensor.shape)
print()

# model instance
model = Simple1DCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# 例如简单训练循环
for epoch in range(epochs):
    permutation = torch.randperm(inputs_tensor.size()[0])
    for i in range(0, inputs_tensor.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        input, label = inputs_tensor[indices], labels_tensor[indices]

        # 前向
        outputs = model(input)
        loss = loss_function(outputs, label)
        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


torch.save(model.state_dict(), '../model/1D-CNN.pth')
