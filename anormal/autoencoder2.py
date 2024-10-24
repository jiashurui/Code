import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from anormal.AEModel import LSTMFCAutoencoder, ConvAutoencoder, VAE, LSTMAutoencoder, LSTM_VAE, ConvLSTMAutoencoder, \
    ConvLSTM_VAE
from anormal.t_SNE import plot_tsne, plot_pca
from datareader.child_datareader import get_child_all_features, get_child_part_action, get_child_2024_all_features
from datareader.realworld_datareader import get_realworld_raw_for_abnormal
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
epochs = 50  # Number of training epochs
slide_window_length = 128  # 序列长度
batch_size = 64
dataset_name = 'realworld'
model_name = 'conv_lstm_vae'
trans_flag = False
# https://arxiv.org/abs/2109.08203
torch.manual_seed(3407)

# (simple_size, window_length, features_num)
# train_data = get_child_all_features(slide_window_length)
# train_data, test_data = get_child_part_action(slide_window_length)
# train_data, test_data = get_child_2024_all_features(slide_window_length)
train_normal,  test_abnormal = get_realworld_raw_for_abnormal(slide_window_length, 6)

input_dim = train_normal.size(2)  # Dimensionality of input sequence

# LSTM Autoencoder Model
# Forward Input (batch_size, seq_length, dim)

if model_name == 'lstm':
    hidden_dim = 128  # Hidden state size
    latent_dim = 64  # Latent space size
    num_layers = 3  # Number of LSTM layers
    model = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
    model_load = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
    loss_function = nn.MSELoss(reduction='sum')  # MSE loss for reconstruction

elif model_name == 'vae':
    model = VAE(input_dim,2).to(device)
    model_load = VAE(input_dim, 32).to(device)

elif model_name == 'lstm_vae':
    hidden_dim = 128 * 2  # Hidden state size
    num_layers = 3  # Number of LSTM layers
    model = LSTM_VAE(input_dim, hidden_dim, num_layers).to(device)
    model_load = LSTM_VAE(input_dim, hidden_dim, num_layers).to(device)

elif model_name == 'conv_lstm_vae':
    train_normal = train_normal.transpose(1,2)
    test_abnormal = test_abnormal.transpose(1, 2)
    trans_flag = True
    model = ConvLSTM_VAE(input_dim).to(device)
    model_load = ConvLSTM_VAE(input_dim).to(device)

elif model_name == 'conv_lstm':
    train_normal = train_normal.transpose(1,2)

    test_abnormal = test_abnormal.transpose(1, 2)
    trans_flag = True
    model = ConvLSTMAutoencoder(input_dim).to(device)
    model_load = ConvLSTMAutoencoder(input_dim).to(device)

# Conv Autoencoder Model
# Forward Input (batch_size, dim(channel), data_dim(length/height & width))
# train_data = train_normal.transpose(1, 2)
# test_data = train_abnormal.transpose(1, 2)
# input_dim = train_data.size(2)  # dim for CNN is changed
# model = ConvAutoencoder(input_dim).to(device)
# model_load = ConvAutoencoder(input_dim).to(device)

# VAE
# model = VAE(input_dim,50).to(device)
# model_load = VAE(input_dim, 50).to(device)

loss_function = nn.MSELoss()  # MSE loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
gradient_utils = GradientUtils(model)

# Train
model.train()
lost_arr = []
lost_avg_arr = []
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
        # 自己和重构后的自己比较
        # VAE
        if model_name == 'vae' or model_name == 'lstm_vae' or model_name == 'conv_lstm_vae':
            output, latent_vector, u, sigma = model(input_data)
            loss = model.loss_function(output, input_data, u, sigma)
        else:
            output, latent_vector = model(input_data)
            loss = loss_function(output, input_data)

        # 样本Loss
        loss_per_epoch = loss_per_epoch + loss.item() / batch_size  # 每一轮epoch的样本总loss
        loss_sum = (loss_sum + loss.item())  # 该轮每一个样本的平均loss

        # BP
        loss.backward()

        # 梯度剪裁
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 记录梯度数据
        gradient_utils.record_gradient_norm()

        optimizer.step()

    lost_avg_arr.append(loss_sum/i)      # 平均单样本 loss
    lost_arr.append(loss_per_epoch)      # 平均每轮累计 loss
    print('epoch: {}, sum_loss: {}, avg_loss_per_simple:{}'.format(epoch, loss_per_epoch, loss_sum/i))

print("Training complete.")

# save my model
torch.save(model.state_dict(), '../model/autoencoder.pth')
loss_plot = show.show_me_data0(lost_arr[1:len(lost_arr)-1])
loss_avg_plot = show.show_me_data0(lost_avg_arr[1:len(lost_avg_arr)-1])

# 展示梯度数据
gradient_utils.show()

# Test
##############################################################################################################
# 实例化模型(加载模型参数)
model_load.load_state_dict(torch.load('../model/autoencoder.pth'))
model_load.eval()

# 测试反例
with torch.no_grad():
    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    latent_abnormal = []
    show_count = 0

    for i in range(0, test_abnormal.size()[0], batch_size):
        input_data = test_abnormal[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        # VAE
        if model_name == 'vae' or model_name == 'lstm_vae' or model_name == 'conv_lstm_vae':
            outputs, latent_vector, u, sigma = model(input_data)
            latent_abnormal.append(latent_vector)
            loss = model.loss_function(outputs, input_data, u, sigma)
        else:
            outputs,latent_vector = model(input_data)
            latent_abnormal.append(latent_vector)
            loss = loss_function(outputs, input_data)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())
        every_simple_loss.append(loss.item())

        # 输出
        if show_count < 2:
            show_tensor_data(input_data, outputs, loss, trans_flag, title='test-abnormal-showcase')
            show_count += 1

    print(f'测试集平均单样本(反例) loss:{loss_sum_test / i}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)

latent_abnormal_tensor = torch.cat(latent_abnormal, dim=0)

if model_name == 'conv_lstm' or model_name == 'conv_lstm_vae':
    test_abnormal = test_abnormal.transpose(1,2)
