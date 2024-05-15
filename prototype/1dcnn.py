import numpy as np
import torch
import torch.nn as nn

from model import Simple1DCNN
from prototype.constant import Constant
from prototype.dataReader import get_data
import utils.show as show

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
channel = 1  # 1D data
slide_window_length = 200  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 20
batch_size = 128  # 或其他合适的批次大小
stop_simple = 500  # 数据静止的个数
learning_rate = 0.0001
label_map = Constant.RealWorld.label_map

# read data
train_data, train_labels, test_data, test_labels = get_data(slide_window_length)

# model instance
model = Simple1DCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# train
model.train()

lost_arr = []
for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])
    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        input_data, label = train_data[indices], train_labels[indices]

        # forward
        outputs = model(input_data)
        loss = loss_function(outputs, label)
        lost_arr.append(loss.item())
        # BP
        loss.backward()
        optimizer.step()
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))

show.show_me_data0(lost_arr)
# save my model
torch.save(model.state_dict(), '../model/1D-CNN.pth')
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# 实例化模型(加载模型参数)
model_load = Simple1DCNN().to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN.pth'))

model_load.eval()
num_sum = 0
correct = 0
test_loss = 0
confusion_matrix = np.zeros((label_map.size(), label_map.size()))

with torch.no_grad():
    for i in range(0, test_data.size()[0], batch_size):
        input_data, label = test_data[i: i + batch_size], test_labels[i: i + batch_size]
        if label.size(0) != batch_size:
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

show.show_me_hotmap(confusion_matrix)