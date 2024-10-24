import numpy as np
import torch
import torch.nn as nn

from model import SimpleRNN
from prototype.constant import Constant
from prototype.dataReader import get_data_1d_3ch_child
import utils.show as show
import utils.report as report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
slide_window_length = 20  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 20
batch_size = 100  # 或其他合适的批次大小
stop_simple = 500  # 数据静止的个数
learning_rate = 0.0001
label_map = Constant.ChildWalk.action_map

# read data
train_data, train_labels, test_data, test_labels = get_data_1d_3ch_child(slide_window_length)

train_labels -= 1
test_labels -= 1

train_data = torch.transpose(train_data, 1, 2)
test_data = torch.transpose(test_data, 1, 2)

# model instance
model = SimpleRNN(hidden_layer_size=slide_window_length,input_size=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# train
model.train()

lost_arr = []
for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])

    loss_per_epoch = 0.0
    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        hidden_state = model.init_hidden(batch_size).to(device)

        indices = permutation[i:i + batch_size]
        input_data, label = train_data[indices], train_labels[indices]
        if input_data.size(0) != batch_size:
            continue
        # forward
        outputs, hidden_state = model(input_data, hidden_state)
        loss = loss_function(outputs, label)
        loss_per_epoch = loss_per_epoch + loss.item()/batch_size

        # BP
        loss.backward()
        optimizer.step()

    lost_arr.append(loss_per_epoch)
    print('epoch: {}, loss: {}'.format(epoch, loss_per_epoch))

loss_plot = show.show_me_data0(lost_arr)
report.save_plot(loss_plot, 'learn-loss')

# save my model
torch.save(model.state_dict(), '../model/rnn.pth')
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# 实例化模型(加载模型参数)
model_load = SimpleRNN(hidden_layer_size=slide_window_length,input_size=3).to(device)
model_load.load_state_dict(torch.load('../model/rnn.pth'))

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
        hidden = model.init_hidden(batch_size).to(device)
        outputs, _ = model_load(input_data, hidden)  # tensor(64,1,7)  概率

        # test_loss += loss_function(outputs, label).item()
        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引
        # correct += torch.eq(pred, label.reshape(batch_size, 1)).sum().item()

        for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
            confusion_matrix[actual, expected] += 1
            if actual == expected:
                correct += 1

        num_sum += batch_size

print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

heatmap_plot = show.show_me_child_hotmap(confusion_matrix)
report.save_plot(heatmap_plot, 'heat-map')