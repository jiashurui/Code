import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Simple1DCNN
from sklearn.preprocessing import StandardScaler

from utils.show import show_me_data1, show_me_data2, show_me_data0
from utils.slidewindow import slide_window2
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例数据
channel = 1  # 输入通道数
slide_window_length = 200  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 100
batch_size = 128  # 或其他合适的批次大小
stop_simple = 500  # 数据静止的个数
learning_rate = 0.0001
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

    # 去除头部
    data = data[stop_simple: len(data)]

    # 滑动窗口平均噪声
    data.rolling(window=3).mean()

    # 特征合并
    data['xyz'] = data.apply(lambda row:
                             np.sqrt(row['attr_x'] ** 2 + row['attr_y'] ** 2 + row['attr_z'] ** 2)
                             , axis=1)

    # show_me_data1(data[1000:1100], ['attr_x','attr_y','attr_z','xyz'])

    # 分割后的数据 100个 X组
    data_sliced = slide_window2(data, slide_window_length, 0.5)

    # show_me_data2(data_sliced,['attr_x','attr_y','attr_z','xyz'])
    # 对于每一个dataframe , 滑动窗口分割数据
    final_data.extend(data_sliced)

# shuffle data
random.shuffle(final_data)
# 提取输入和标签
input_features = np.array([df['xyz'].values for df in final_data])
labels = np.array([df['label'].values for df in final_data])[:, 0]

# show_me_data0(input_features[0])

# 将NumPy数组转换为Tensor
inputs_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels_tensor = torch.tensor(labels, dtype=torch.long)

# 计算分割点 7:3
split_point = int(0.7 * len(inputs_tensor))

# train data/label   test data/label
train_data = inputs_tensor[:split_point].to(device)
test_data = inputs_tensor[split_point:].to(device)
train_labels = labels_tensor[:split_point].to(device)
test_labels = labels_tensor[split_point:].to(device)

# model instance
model = Simple1DCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# train
model.train()
lost_arr = []

for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])
    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        input_data, label = train_data[indices], train_labels[indices]

        # forward
        outputs = model(input_data)
        loss = loss_function(outputs, label)
        lost_arr.append(loss.item())
        # BP
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss.item()))
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

show_me_data0(lost_arr)
# save my model
torch.save(model.state_dict(), '../model/1D-CNN.pth')
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# 实例化模型(加载模型参数)
model_load = Simple1DCNN().to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN.pth'))

model_load.eval()
num_sum = 0
correct = 0
test_loss = 0
with torch.no_grad():

    for i in range(0, test_data.size()[0], batch_size):
        input_data, label = train_data[i: i + batch_size], test_labels[i: i + batch_size]
        if label.size(0) != batch_size:
            continue

        input_data, label = test_data[i: i + batch_size], test_labels[i: i + batch_size]
        outputs = model_load(input_data)

        # test_loss += loss_function(outputs, label).item()
        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引

        correct += torch.eq(pred, label.reshape(batch_size,1)).sum().item()
        num_sum += batch_size

print(
    f'\nTest set: Average loss: {test_loss/num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')
