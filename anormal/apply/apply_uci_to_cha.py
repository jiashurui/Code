import torch

from anormal.AEModel import VAE
from datareader.cha_datareader import get_cha_data_for_abnormal
from datareader.show_child_2024 import show_tensor_data
from utils import show
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slide_window_length = 128  # 序列长度
normal_data, abnormal_data = get_cha_data_for_abnormal(slide_window_length)
input_dim = normal_data.size(2)  # Dimensionality of input sequence

dataset_name = 'cha'
model_load = VAE(input_dim, 50).to(device)
batch_size = 64

model_load.load_state_dict(torch.load('../../model/autoencoder.pth'))
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
        outputs, _, u, sigma = model_load(input_data)
        # loss = loss_function(outputs, input_data)
        loss = model_load.loss_function(outputs, input_data, u, sigma)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss, dataset_name, title=f'{dataset_name}-normal-showcase')
            show_count += 1

        every_simple_loss.append(loss.item())

    print(f'测试集(mHealth)平均单样本(正例) loss: {loss_sum_test / (i+1)}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)

# 测试异常
with torch.no_grad():

    loss_sum_test = 0.0  #
    every_simple_loss = []  # 每个样本的loss(batch)
    show_count = 0

    for i in range(0, abnormal_data.size()[0], batch_size):
        input_data = abnormal_data[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs, _, u, sigma = model_load(input_data)
        # loss = loss_function(outputs, input_data)
        loss = model_load.loss_function(outputs, input_data, u, sigma)

        # 单样本Loss
        loss_sum_test = (loss_sum_test + loss.item())

        # 输出
        if show_count < 5:
            show_tensor_data(input_data, outputs, loss, dataset_name, title=f'{dataset_name}-abnormal-showcase')
            show_count += 1

        every_simple_loss.append(loss.item())

    print(f'测试集(mHealth)平均单样本(反例) loss: {loss_sum_test / (i+1)}')  # 平均单样本 loss

    show.show_me_data0(every_simple_loss)