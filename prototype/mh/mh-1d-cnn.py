import numpy as np
import torch
from torch import nn

from datareader.mh_datareader import get_mh_data, get_mh_data_1d_3ch
from prototype.constant import Constant
from prototype.model import SimpleRNN, Simple1DCNN
from utils import show, report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
slide_window_length = 200  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 200
batch_size = 128  # 或其他合适的批次大小
learning_rate = 0.0001
label_map = Constant.mHealth.action_map

# read data
train_data, train_labels, test_data, test_labels = get_mh_data_1d_3ch(slide_window_length)
train_labels -= 1
test_labels -= 1

# model instance
model = Simple1DCNN(in_channels=3,out_label=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
loss_function = nn.CrossEntropyLoss()

# train
model.train()

lost_arr = []
for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])

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

        # BP
        loss.backward()
        optimizer.step()

    lost_arr.append(loss_per_epoch)
    print('epoch: {}, loss: {}'.format(epoch, loss_per_epoch))

loss_plot = show.show_me_data0(lost_arr)
report.save_plot(loss_plot, 'learn-loss')

# save my model
torch.save(model.state_dict(), '../model/1D-CNN-3CH.pth')


################################################################################
################################################################################
################################################################################

# 实例化模型(加载模型参数)
model_load = Simple1DCNN(in_channels=3,out_label=12).to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN-3CH.pth'))

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

heatmap_plot = show.show_me_mh_hotmap(confusion_matrix)
fig = heatmap_plot.gcf()
report.save_plot(heatmap_plot, 'heat-map')