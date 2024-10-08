from datetime import datetime

import numpy as np
import torch
from torch import nn

from cnn.cnn import DeepOneDimCNN
from datareader.mh_datareader import get_mh_data_1d_9ch
from datareader.realworld_datareader import get_realworld_for_recon
from prototype import constant
from prototype.constant import Constant
from utils import show, report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slide_window_length = 128  # 序列长度
learning_rate: float = 0.0001
batch_size = 64
epochs = 100
dataset = 'mh'
label_map = constant.Constant.mHealth.action_map
in_channel = 6
out_channel = len(label_map)

model = DeepOneDimCNN(in_channels=in_channel, out_channel=out_channel).to(device)
model_load = DeepOneDimCNN(in_channels=in_channel, out_channel=out_channel).to(device)


def train_model():
    # mHealth
    train_data, train_labels, test_data, test_labels = get_mh_data_1d_9ch(slide_window_length, in_channel)
    train_labels -= 1
    test_labels -= 1



    # CNN need transformed
    train_data = train_data.transpose(1, 2)
    test_data = test_data.transpose(1, 2)

    # model instance
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()



    # train
    model.train()

    lost_arr = []
    for epoch in range(epochs):
        permutation = torch.randperm(train_data.size()[0])
        correct_train = 0
        num_sum_train = 0

        loss_per_epoch = 0.0
        for i in range(0, train_data.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            input_data, label = train_data[indices], train_labels[indices]
            if input_data.size(0) != batch_size:
                continue

            # forward (batch, 3 features(xyz) , 200 size_dim(时间维度) , 1 (空间维度), 1(空间维度))
            outputs = model(input_data)
            loss = loss_function(outputs, label)
            loss_per_epoch = loss_per_epoch + loss.item()/batch_size

            # 训练过程中 预测分类结果
            # TODO mh_Health 索引需要加一
            pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引

            correct_train += pred.eq(label.view_as(pred)).sum().item()
            num_sum_train += batch_size

            # BP
            loss.backward()
            optimizer.step()

        lost_arr.append(loss_per_epoch)
        print('epoch: {}, loss: {}'.format(epoch, loss_per_epoch))
        print(f'Accuracy: {correct_train}/{num_sum_train} ({100. * correct_train / num_sum_train:.0f}%)\n')


    loss_plot = show.show_me_data0(lost_arr)
    report.save_plot(loss_plot, 'learn-loss')

    # save my model
    torch.save(model.state_dict(), f'../model/1D-CNN-{dataset}-{out_channel}CH.pth')



    model_load.load_state_dict(torch.load(f'../model/1D-CNN-{dataset}-{out_channel}CH.pth'))

    model_load.eval()
    num_sum = 0
    correct = 0
    test_loss = 0
    confusion_matrix = np.zeros((out_channel, out_channel))

    with torch.no_grad():
        for i in range(0, test_data.size()[0], batch_size):
            input_data, label = test_data[i: i + batch_size], test_labels[i: i + batch_size]
            if label.size(0) != batch_size:
                continue

            outputs = model_load(input_data)
            pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引

            for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
                confusion_matrix[actual, expected] += 1
                if actual == expected:
                    correct += 1

            num_sum += batch_size

    print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

    heatmap_plot = show.show_me_mh_hotmap(confusion_matrix)
    fig = heatmap_plot.gcf()
    report.save_plot(heatmap_plot, 'heat-map')


def apply_1d_cnn(test_data):
    start_time = datetime.now()
    # 归一化(128, 9)
    test_data = standlize(test_data)
    tensor_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    data = tensor_data.unsqueeze(0).transpose(1, 2)[:, :in_channel, :]
    outputs = model_load(data)
    pred = outputs.argmax(dim=1, keepdim=True)
    print(f"Model predict finished, start: {start_time} , end: {datetime.now()}")
    return pred

def standlize(arr):
    # 对每一列进行标准化到 [-1, 1]
    arr_min = np.min(arr, axis=0)  # 计算每一列的最小值
    arr_max = np.max(arr, axis=0)  # 计算每一列的最大值

    # 如果最大值和最小值相等，标准化会出现问题，需要检查
    standardized_arr = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        if arr_max[i] - arr_min[i] != 0:
            # 标准化公式：2 * (x - min) / (max - min) - 1
            standardized_arr[:, i] = 2 * (arr[:, i] - arr_min[i]) / (arr_max[i] - arr_min[i]) - 1
        else:
            standardized_arr[:, i] = 0  # 如果列中最大值等于最小值，全部置0

    return standardized_arr