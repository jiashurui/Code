from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from cnn.conv_lstm import ConvLSTM
from datareader.datareader_stu_1111 import simple_get_stu_1111_all_features
from datareader.realworld_datareader import get_realworld_for_recon
from prototype.constant import Constant
from utils import show, report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slide_window_length = 40  # 序列长度
learning_rate: float = 0.0001
batch_size = 64
epochs = 1
model_path = '../model/Conv_LSTM_STU_1111.pth'
label_map = Constant.uStudent_1111.action_map_en_reverse
label_map_str = Constant.uStudent_1111.action_map

in_channel = 9
out_channel = len(label_map)
model = ConvLSTM(input_dim=in_channel, output_dim=out_channel).to(device)
model_load = ConvLSTM(input_dim=in_channel, output_dim=out_channel).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_function = nn.CrossEntropyLoss()


def train_model():
    #
    origin_data_np = simple_get_stu_1111_all_features(slide_window_length, type='np')

    # Label processing (rounding to avoid float errors)
    label = np.round(origin_data_np[:, 0, 9]).astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(origin_data_np[:,:,:9], label, test_size=0.3, random_state=0)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)


    X_train = X_train.transpose(1, 2)
    X_test = X_test.transpose(1, 2)

    lost_arr = []
    for epoch in range(epochs):
        # train
        model.train()

        permutation = torch.randperm(X_train.size()[0])
        correct_train = 0
        num_sum_train = 0

        loss_per_epoch = 0.0
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            input_data, label = X_train[indices], y_train[indices]
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

        test_model(model, X_train, y_test)

    # 展示训练Loss
    show.show_me_data0(lost_arr)

    # save my model
    torch.save(model.state_dict(), model_path)

    # Load Model
    model_load.load_state_dict(torch.load(model_path, map_location=device))

    # 测试模型
    confusion_matrix = test_model(model_load, X_test, y_test)

    show.show_me_hotmap(confusion_matrix, label_map=label_map_str)

def test_model(model ,test_data ,test_label):
    model.eval()
    num_sum = 0
    correct = 0
    test_loss = 0
    confusion_matrix = np.zeros((len(label_map), len(label_map)))

    with torch.no_grad():
        for i in range(0, test_data.size()[0], batch_size):
            input_data, label = test_data[i: i + batch_size], test_label[i: i + batch_size]
            if label.size(0) != batch_size:
                continue

            outputs = model(input_data)
            pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引

            for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
                confusion_matrix[actual, expected] += 1
                if actual == expected:
                    correct += 1

            num_sum += batch_size

    print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

    return confusion_matrix

model_load_flag = False
def apply_conv_lstm(test_data):
    global model_load_flag
    model_apply = ConvLSTM(input_dim=in_channel, output_dim=out_channel).to(device)

    if not model_load_flag:
        model_apply.load_state_dict(torch.load(model_path, map_location=device))
        model_apply.eval()

    start_time = datetime.now()
    # 归一化(128, 9)
    # test_data = standlize(test_data)
    tensor_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    data = tensor_data.unsqueeze(0).transpose(1, 2)[:, :in_channel, :]
    outputs = model_apply(data)
    pred = outputs.argmax(dim=1, keepdim=True)
    print(f"Model predict finished, start: {start_time} , end: {datetime.now()}")
    return pred

if __name__ == '__main__':
    train_model()