from datetime import datetime

import numpy as np
import torch
from torch import nn

from datareader.realworld_datareader import get_realworld_for_recon
from prototype.constant import Constant
from prototype.model import LSTM
from train.train_mh_1d_cnn import standlize
from utils import show, report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slide_window_length = 128  # 序列长度
learning_rate: float = 0.0001
batch_size = 64
epochs = 30
in_channel = 6
out_channel = 8
model_path = '../model/LSTM.pth'
# realworld
model = LSTM(input_size=in_channel,output_size=out_channel).to(device)
model_load = LSTM(input_size=in_channel,output_size=out_channel).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_function = nn.CrossEntropyLoss()
label_map = Constant.RealWorld.action_map


def train_model():
    train_data, train_labels, test_data, test_labels = get_realworld_for_recon(slide_window_length,in_channel)

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
            pred = outputs.argmax(dim=1, keepdim=True) # 获取概率最大的索引
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
    torch.save(model.state_dict(), model_path)



    model_load.load_state_dict(torch.load(model_path))

    model_load.eval()
    num_sum = 0
    correct = 0
    test_loss = 0
    confusion_matrix = np.zeros((len(label_map), len(label_map)))

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

    heatmap_plot = show.show_me_hotmap(confusion_matrix)
    fig = heatmap_plot.gcf()
    report.save_plot(heatmap_plot, 'heat-map')

model_load_flag = False

def apply_lstm(test_data):
    global model_load_flag
    model_apply = LSTM(input_size=in_channel,output_size=out_channel).to(device)

    if not model_load_flag:
        model_apply.load_state_dict(torch.load(model_path, map_location=device))
        model_apply.eval()

    start_time = datetime.now()
    # 归一化(128, 9)
    # test_data = standlize(test_data)
    tensor_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    data = tensor_data.unsqueeze(0)[:, :, :in_channel]
    outputs = model_apply(data)
    pred = outputs.argmax(dim=1, keepdim=True)
    print(f"Model predict finished, start: {start_time} , end: {datetime.now()}")
    return pred

if __name__ == '__main__':
    train_model()