import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch

# 共通の緯度・経度グリッドを定義します
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)


file_path = '../data/weather/Data.csv'
df = pd.read_csv(file_path, index_col=0)
df.head()

# 沖縄周辺
selected_lat = common_lat[(common_lat >= 23.5) & (common_lat <= 29.5)]
selected_lon = common_lon[(common_lon >= 124.5) & (common_lon <= 130.5)]

# 日本列島
# selected_lat = common_lat[(common_lat >= 30.0) & (common_lat < 46.0)]
# selected_lon = common_lon[(common_lon >= 128.0) & (common_lon < 144.0)]


# build select index
selected_index = ['沖縄の天気','沖縄の降水量']
for lat in selected_lat:
    for lon in selected_lon:
        selected_index.append(f"{lat}_{lon}")

df = df[selected_index]

Y1 = (df.iloc[:, 0] / 100).astype(int)
Y2 = df.iloc[:, 1]
X = df.iloc[:, 2:]

df[df.iloc[:, 0] >= 500].iloc[:, :2]
Y1[Y1 == 5] = 3
Y1[Y1 == 7] = 3

y1_train = Y1.iloc[:365]
y1_test = Y1.iloc[365:]
y2_train = Y2.iloc[:365]
y2_test = Y2.iloc[365:]
x_train = X.iloc[:365]
x_test = X.iloc[365:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTM_Weather(nn.Module):
    def __init__(self):
        super(ConvLSTM_Weather, self).__init__()
        # Class Parameter
        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # CONV
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        # # LSTM
        # self.lstm = nn.LSTM(128 * 2 * 2, 256, num_layers=3, batch_first=True, dropout=0.1)
        #
        # # Layer Norm
        # self.layer_norm1 = nn.LayerNorm(normalized_shape=256)

        # FC
        self.fc = nn.Linear(256 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

        # dropout
        self.dropout = nn.Dropout(0.1)

        # Batch Norm
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        # Init Weight
        self.init_weights()

    def init_weights(self):
        for m in [self.fc, self.fc2, self.fc3]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)


        # Flatten the output of CNN
        # x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, features)
        # x = x.unsqueeze(0)  # Add a sequence dimension (batch_size, sequence_length, input_size)
        #
        # # LSTM
        # x, (hidden, cell) = self.lstm(x)
        # x = self.dropout(x)
        # x = self.layer_norm1(x)

        # FC
        result = self.fc(x.view(x.size(0), -1))
        result = self.dropout(result)
        result = self.fc2(result)
        result = self.dropout(result)
        result = self.fc3(result)
        return result


# apply Conv-LSTM model
model = ConvLSTM_Weather().to(device)

torch.manual_seed(3407)
random.seed(3407)
learning_rate: float = 0.01
batch_size = 16
epochs = 100
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

x_train_tensor = torch.tensor(x_train.values.reshape(365, 1, 13, 13), dtype=torch.float32)
y1_train_tensor = torch.tensor(y1_train.values, dtype=torch.long)
x_test_tensor = torch.tensor(x_test.values.reshape(90, 1, 13, 13), dtype=torch.float32)
y1_test_tensor = torch.tensor(y1_test.values, dtype=torch.long)


train_loader = DataLoader(
    TensorDataset(x_train_tensor, y1_train_tensor),
    batch_size=batch_size,
    shuffle=False
)
test_loader = DataLoader(
    TensorDataset(x_test_tensor, y1_test_tensor),
    batch_size=batch_size,
    shuffle=False
)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # 获取预测类别
    correct = (predicted == labels).sum().item()  # 计算正确的预测数量
    accuracy = correct / labels.size(0)  # 计算准确率
    return accuracy

# 训练循环
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_train_accuracy = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if inputs.size(0) != batch_size:
            continue

        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = loss_function(outputs, labels - 1)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_accuracy = calculate_accuracy(outputs, labels - 1)
        running_train_accuracy += train_accuracy

        running_loss += loss.item()

    avg_train_accuracy = running_train_accuracy / len(train_loader)

    # 计算测试精度
    model.eval()  # 设置模型为评估模式
    running_test_accuracy = 0.0

    # Test model
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs.size(0) != batch_size:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算测试精度
            test_accuracy = calculate_accuracy(outputs, labels - 1)
            running_test_accuracy += test_accuracy
        avg_test_accuracy = running_test_accuracy / len(test_loader)

    # 打印每个 epoch 的损失
    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Loss: {running_loss / len(train_loader):.4f} "
          f"Train Accuracy: {avg_train_accuracy * 100:.2f}%, "
          f"Test Accuracy: {avg_test_accuracy * 100:.2f}%")
