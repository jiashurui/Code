import numpy as np
import torch
from torch import nn

from datareader.mh_datareader import get_mh_data
from prototype.constant import Constant
from prototype.model import SimpleRNN, LSTM
from utils import show, report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
slide_window_length = 100  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 100
batch_size = 12  # 或其他合适的批次大小
learning_rate = 0.00001
label_map = Constant.mHealth.action_map

# read data
train_data, train_labels, test_data, test_labels = get_mh_data(slide_window_length)
train_labels -= 1
test_labels -= 1

train_data = torch.transpose(train_data, 1, 2)
test_data = torch.transpose(test_data, 1, 2)

# model instance
model = LSTM(input_size=3, output_size=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# train
model.train()

lost_arr = []
acc_arr = []

for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])
    num_sum_train = 0
    correct_train = 0

    loss_per_epoch = 0.0
    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        indices = permutation[i:i + batch_size]
        input_data, label = train_data[indices], train_labels[indices]
        if input_data.size(0) != batch_size:
            continue
        # forward
        outputs = model(input_data)
        loss = loss_function(outputs, label)
        loss_per_epoch = loss_per_epoch + loss.item()/batch_size

        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引
        correct_train += pred.eq(label.view_as(pred)).sum().item()
        num_sum_train += batch_size

        # BP
        loss.backward()
        optimizer.step()

    lost_arr.append(loss_per_epoch)
    print('epoch: {}, loss: {}'.format(epoch, loss_per_epoch))

    acc_train = correct_train / num_sum_train
    acc_arr.append(acc_train * 100.)
    print(f'Accuracy: {acc_train} ({100. * correct_train / num_sum_train:.0f}%)\n')


loss_plot = show.show_me_data0(lost_arr)
acc_plot = show.show_me_acc(acc_arr)

report.save_plot(loss_plot, 'learn-loss')

# save my model
torch.save(model.state_dict(), '../model/lstm.pth')
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# 实例化模型(加载模型参数)
model_load = LSTM(input_size=3,output_size=12).to(device)
model_load.load_state_dict(torch.load('../model/lstm.pth'))

model_load.eval()
num_sum = 0
correct = 0
test_loss = 0
confusion_matrix = np.zeros((len(label_map), len(label_map)))

with torch.no_grad():
    for i in range(0, test_data.size()[0], batch_size):
        input_data, label = test_data[i: i + batch_size], test_labels[i: i + batch_size]

        if input_data.size(0) != batch_size:
            continue
        outputs = model_load(input_data)  # tensor(64,1,7)  概率

        # test_loss += loss_function(outputs, label).item()
        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引
        # correct += torch.eq(pred, label.reshape(batch_size, 1)).sum().item()

        for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
            confusion_matrix[actual, expected] += 1
            if actual == expected:
                correct += 1

        num_sum += batch_size

print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

heatmap_plot = show.show_me_mh_hotmap(confusion_matrix)
report.save_plot(heatmap_plot, 'heat-map')


