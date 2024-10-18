import torch
from torch import nn

from anormal.AEModel import VAE, LSTMFCAutoencoder, ConvLSTMAutoencoder, LSTM_VAE, ConvLSTM_VAE
from datareader.datareader_stu import get_stu_part_features
from datareader.show_child_2024 import show_tensor_data
from utils import show
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slide_window_length = 128  # 序列长度
batch_size = 1

model_name = 'conv_lstm_vae'

normal_data, abnormal_data = get_stu_part_features(slide_window_length,6,1)
input_dim = normal_data.size(2)  # Dimensionality of input sequence
transflag = False
dataset_name = 'student'

if model_name == 'lstm':
    hidden_dim = 1024  # Hidden state size
    latent_dim = 512  # Latent space size
    num_layers = 3  # Number of LSTM layers
    model = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
    model_load = LSTMFCAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers).to(device)
    loss_function = nn.MSELoss()  # MSE loss for reconstruction

elif model_name == 'vae':
    model_load = VAE(input_dim, 50).to(device)

elif model_name == 'lstm_vae':
    hidden_dim = 128 * 2  # Hidden state size
    num_layers = 3  # Number of LSTM layers
    model = LSTM_VAE(input_dim, hidden_dim, num_layers).to(device)
    model_load = LSTM_VAE(input_dim, hidden_dim, num_layers).to(device)

elif model_name == 'conv_lstm_vae':
    normal_data = normal_data.transpose(1,2)
    abnormal_data = abnormal_data.transpose(1, 2)
    transflag = True
    model = ConvLSTM_VAE(input_dim).to(device)
    model_load = ConvLSTM_VAE(input_dim).to(device)

elif model_name == 'conv_lstm':
    normal_data = normal_data.transpose(1,2)
    abnormal_data = abnormal_data.transpose(1, 2)
    transflag = True
    model = ConvLSTMAutoencoder(input_dim).to(device)
    model_load = ConvLSTMAutoencoder(input_dim).to(device)
    loss_function = nn.MSELoss()  # MSE loss for reconstruction



model_load.load_state_dict(torch.load('../model/autoencoder.pth'))
model_load.eval()

# 测试正常
with torch.no_grad():

    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, normal_data.size()[0], batch_size):
        input_data = normal_data[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue

        # VAE
        if model_name == 'vae' or model_name == 'lstm_vae' or model_name == 'conv_lstm_vae':
            outputs, _, u, sigma = model_load(input_data)
            loss = model_load.loss_function(outputs, input_data, u, sigma)
        else:
            outputs,latent = model_load(input_data)
            loss = loss_function(outputs, input_data)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())
        every_simple_loss.append(loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss, transflag, title=f'{dataset_name}-normal-showcase')
            show_count += 1

    print(f'测试集({dataset_name})平均单样本(正例) loss: {loss_sum_test / (i+1)}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)

# 测试异常
with torch.no_grad():

    loss_sum_test = 0.0
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, abnormal_data.size()[0], batch_size):
        input_data = abnormal_data[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue

        # VAE
        if model_name == 'vae' or model_name == 'lstm_vae' or model_name == 'conv_lstm_vae':
            outputs, _, u, sigma = model_load(input_data)
            loss = model_load.loss_function(outputs, input_data, u, sigma)
        else:
            outputs,latent = model_load(input_data)
            loss = loss_function(outputs, input_data)

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss, transflag, title=f'{dataset_name}-abnormal-showcase')
            show_count += 1

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())
        every_simple_loss.append(loss.item())

    print(f'测试集({dataset_name})平均单样本(反例) loss: {loss_sum_test / (i+1)}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)
