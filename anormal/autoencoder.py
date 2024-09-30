import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from anormal.AEModel import LSTMFCAutoencoder, ConvAutoencoder, VAE
from datareader.child_datareader import get_child_all_features, get_child_part_action, get_child_2024_all_features
from datareader.show_child_2024 import show_tensor_data
from utils import show
from utils.show import GradientUtils
from utils.uci_datareader import get_data_1d_uci_all_features, get_data_1d_uci_all_data

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 1024  # Hidden state size
latent_dim = 512  # Latent space size
num_layers = 3  # Number of LSTM layers
learning_rate = 0.0001  # Learning rate
epochs = 40  # Number of training epochs
slide_window_length = 128  # 序列长度
batch_size = 8
dataset_name = 'uci'
# https://arxiv.org/abs/2109.08203
torch.manual_seed(3407)

# (simple_size, window_length, features_num)
# train_data = get_child_all_features(slide_window_length)
# train_data, test_data = get_child_part_action(slide_window_length)
# train_data, test_data = get_child_2024_all_features(slide_window_length)

train_normal, train_abnormal, test_normal, test_abnormal = get_data_1d_uci_all_data()
input_dim = train_normal.size(2)  # Dimensionality of input sequence

# LSTM Autoencoder Model
# Forward Input (batch_size, seq_length, dim)
# model = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
# model_load = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)

# Conv Autoencoder Model
# Forward Input (batch_size, dim(channel), data_dim(length/height & width))
# train_data = train_data.transpose(1, 2)
# test_data = test_data.transpose(1, 2)
# input_dim = train_data.size(1)  # dim for CNN is changed
# model = ConvAutoencoder(input_dim).to(device)
# model_load = ConvAutoencoder(input_dim).to(device)

# VAE


model = VAE(input_dim,10).to(device)
model_load = VAE(input_dim, 10).to(device)

# loss_function = nn.MSELoss()  # MSE loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
gradient_utils = GradientUtils(model)

# Train
model.train()
lost_arr = []
lost_avg_arr = []
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


for epoch in range(epochs):
    permutation = torch.randperm(train_normal.size()[0])

    loss_per_epoch = 0.0
    loss_sum = 0.0

    for i in range(0, train_normal.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        input_data = train_normal[indices]

        # 输入长度不符合模型要求,则跳过这个输入
        if input_data.size(0) != batch_size:
            continue

        # 模型输出
        # output = model(input_data)
        output, _, u, sigma = model(input_data)

        # 自己和重构后的自己比较
        # loss = loss_function(output, input_data)
        loss = model.loss_function(output, input_data, u, sigma)

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

# 测试异常
with torch.no_grad():

    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, train_abnormal.size()[0], batch_size):
        input_data = train_abnormal[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs, _, u, sigma= model_load(input_data)
        # loss = loss_function(outputs, input_data)
        loss = model_load.loss_function(outputs, input_data, u, sigma)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss, dataset_name, title='train-abnormal-showcase')
            show_count += 1

        every_simple_loss.append(loss.item())

    print(f'训练集(没参加训练)平均单样本(反例) loss: {loss_sum_test / (i+1)}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)

# 测试正例
with torch.no_grad():
    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, test_normal.size()[0], batch_size):
        input_data = test_normal[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs, _, u, sigma = model_load(input_data)
        loss = model_load.loss_function(outputs, input_data,u, sigma)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())
        every_simple_loss.append(loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss,dataset_name, title='test-normal-showcase')
            show_count += 1

    print(f'测试集平均单样本(正例) loss:{loss_sum_test / i}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)

# 测试反例
with torch.no_grad():
    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, test_abnormal.size()[0], batch_size):
        input_data = test_abnormal[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs, _, u, sigma = model_load(input_data)
        loss = model_load.loss_function(outputs, input_data,u, sigma)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())
        every_simple_loss.append(loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss,dataset_name, title='test-abnormal-showcase')
            show_count += 1

    print(f'测试集平均单样本(反例) loss:{loss_sum_test / i}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)
