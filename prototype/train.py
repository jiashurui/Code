import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Simple1DCNN

from utils.slidewindow import slide_window2
import random

# 示例数据
channel = 1  # 输入通道数
slide_window_length = 100  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 50
batch_size = 4  # 或其他合适的批次大小

# 创建示例输入数据 TODO
file_list = glob.glob('../data/realworld/*/acc_walking_*.csv')
final_data = []

# make label by fileName (walking)
# chest 1 forearm 2 head 3 shin 4 thigh 5 upper arm 6 waist 7
label_map = {
    'waist': 0,
    'chest': 1,
    'forearm': 2,
    'head': 3,
    'shin': 4,
    'thigh': 5,
    'upperarm': 6
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

    data_agg = pd.DataFrame(columns=['xyz', 'label'])
    # 数据合并
    for index, row in data.iterrows():
        magnitude = np.sqrt(row['attr_x'] ** 2 + row['attr_y'] ** 2 + row['attr_z'] ** 2)
        # new_row = pd.Series({'xyz': magnitude, 'label': data['label']})
        # data_agg = pd.concat([data_agg, new_row])
        data['xyz'] = magnitude
    #######################################################################

    # 分割后的数据 100个 X组
    data_sliced = slide_window2(data, slide_window_length, 0.5)

    # 对于每一个dataframe , 滑动窗口分割数据
    final_data.extend(data_sliced)

# shuffle data
random.shuffle(final_data)
# 提取输入和标签
input_features = np.array([df['xyz'].values for df in final_data])
labels = np.array([df['label'].values for df in final_data])[:, 0]

# 将NumPy数组转换为Tensor
inputs_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels_tensor = torch.tensor(labels, dtype=torch.long)

# 计算分割点 7:3
split_point = int(0.7 * len(inputs_tensor))

# train data/label   test data/label
train_data = inputs_tensor[:split_point]
test_data = inputs_tensor[split_point:]
train_labels = labels_tensor[:split_point]
test_labels = labels_tensor[split_point:]

# model instance
model = Simple1DCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_function = nn.CrossEntropyLoss()

# train
model.train()
for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])
    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        input_data, label = train_data[indices], train_labels[indices]

        # forward
        outputs = model(input_data)
        loss = loss_function(outputs, label)
        # BP
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss.item()))
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# save my model
torch.save(model.state_dict(), '../model/1D-CNN.pth')