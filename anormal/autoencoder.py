import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from anormal.AEModel import LSTMFCAutoencoder
from datareader.child_datareader import get_child_all_features, get_child_part_action
from datareader.show_child_2024 import show_tensor_data
from utils import show
from utils.show import GradientUtils

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 256  # Hidden state size
latent_dim = 128  # Latent space size
num_layers = 3  # Number of LSTM layers
learning_rate = 0.01  # Learning rate
epochs = 20  # Number of training epochs
slide_window_length = 40  # 序列长度
batch_size = 16

# https://arxiv.org/abs/2109.08203
torch.manual_seed(3407)

# (simple_size, window_length, features_num)
# train_data = get_child_all_features(slide_window_length)
train_data, test_data = get_child_part_action(slide_window_length)
input_dim = train_data.size(2)  # Dimensionality of input sequence

# LSTM Autoencoder Model
# Forward Input (batch_size, seq_length, dim)
model = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
model_load = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
loss_function = nn.MSELoss()  # We use MSE loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
gradient_utils = GradientUtils(model)

# Train
model.train()
lost_arr = []
lost_avg_arr = []
for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])

    loss_per_epoch = 0.0
    loss_sum = 0.0

    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        input_data = train_data[indices]

        # 输入长度不符合模型要求,则跳过这个输入
        if input_data.size(0) != batch_size:
            continue

        # 模型输出
        output = model(input_data)

        # 自己和重构后的自己比较
        loss = loss_function(output, input_data)

        # 样本Loss
        loss_per_epoch = loss_per_epoch + loss.item() / batch_size  # 每一轮epoch的样本总loss
        loss_sum = (loss_sum + loss.item())  # 该轮每一个样本的平均loss

        # BP
        loss.backward()
        # 记录梯度数据
        gradient_utils.record_gradient_norm()

        optimizer.step()

    lost_avg_arr.append(loss_sum/i)      # 平均单样本 loss
    lost_arr.append(loss_per_epoch)      # 平均每轮累计 loss
    print('epoch: {}, sum_loss: {}, avg_loss_per_simple:{}'.format(epoch, loss_per_epoch, loss_sum/i))

print("Training complete.")

# save my model
torch.save(model.state_dict(), '../model/autoencoder.pth')
loss_plot = show.show_me_data0(lost_arr)
loss_avg_plot = show.show_me_data0(lost_avg_arr)

# 展示梯度数据
gradient_utils.show()

# Test
##############################################################################################################
# 实例化模型(加载模型参数)
model_load.load_state_dict(torch.load('../model/autoencoder.pth'))
model_load.eval()

# 测试Ground_truth 的反例
with torch.no_grad():

    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, test_data.size()[0], batch_size):
        input_data = test_data[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs = model_load(input_data)
        loss = loss_function(outputs, input_data)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss)
            show_count += 1

        every_simple_loss.append(loss.item())

    print(f'平均单样本(反例) loss: {loss_sum_test / i}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)

# 测试Ground_truth 的正例
with torch.no_grad():
    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, train_data.size()[0], batch_size):
        input_data = train_data[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs = model_load(input_data)
        loss = loss_function(outputs, input_data)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss)
            show_count += 1

    print(f'平均单样本(正例) loss:{loss_sum_test / i}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)
