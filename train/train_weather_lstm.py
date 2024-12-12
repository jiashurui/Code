import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch

from train.conv_lstm import ConvLSTM


import numpy as np

def sliding_window_split_2d(data, window_size, step):
    seq_len, features = data.shape
    windows = []

    for start in range(0, seq_len - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end, :])

    return np.stack(windows, axis=0)

# 共通の緯度・経度グリッドを定義します
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)


file_path = '../data/weather/Data.csv'
df = pd.read_csv(file_path, index_col=0)
df.head()

# 沖縄周辺
selected_lat = common_lat[(common_lat >= 23.0) & (common_lat <= 30.0)]
selected_lon = common_lon[(common_lon >= 124.0) & (common_lon <= 131.0)]

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

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = sliding_window_split_2d(x_train, 8, 1)
x_test = sliding_window_split_2d(x_test, 8, 1)
y1_train = sliding_window_split_2d(y1_train.values.reshape(-1, 1), 8, 1).squeeze(-1)[:, -1]
y1_test = sliding_window_split_2d(y1_test.values.reshape(-1, 1), 8, 1).squeeze(-1)[:, -1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        # Class Parameter
        kernel_size = 3
        stride = 1
        padding = 1

        self.lstm = nn.LSTM(15 * 15, 32, 3, batch_first=True)

        # CNN
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        self.layer_norm = nn.LayerNorm(32)
        self.fc_cnn = nn.Linear(15 * 15 * 32, 512)

        self.fc1 = nn.Linear(32 + 512, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):

        cnn_features = []

        # (batch, seq, feature_num)
        for data in x:
            data = data.reshape(data.shape[0], 1, 15, 15)
            data = self.relu(self.conv1(data))
            data = self.relu(self.conv2(data))
            data = self.relu(self.conv3(data))
            data = data.view(data.size(0), -1)  # 展平 (batch, 64*7*7)

            data = self.fc_cnn(data)

            cnn_features.append(data)

        cnn_features = torch.stack(cnn_features, dim=0)  # (batch, seq_len, output_dim)


        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        concat_features = torch.cat((cnn_features, lstm_out), dim=2)

        out = self.fc1(concat_features[:, -1, :])  # 只取最后一个时间步 [batch_size, hidden_dim] -> [batch_size, output_dim]
        out = self.fc2(out)  # 只取最后一个时间步 [batch_size, hidden_dim] -> [batch_size, output_dim]
        out = self.fc3(out)  # 只取最后一个时间步 [batch_size, hidden_dim] -> [batch_size, output_dim]

        return out

# apply Conv-LSTM model
# model = ConvLSTM().to(device)
# 初始化 ConvLSTM 模型
model = LSTMClassifier()

torch.manual_seed(3407)
random.seed(3407)
learning_rate: float = 0.00001
batch_size = 16
epochs = 200
class_weights = torch.tensor([1, 1.2, 4.0]).to(device)  # 根据标签分布设置权重

loss_function = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y1_train_tensor = torch.tensor(y1_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y1_test_tensor = torch.tensor(y1_test, dtype=torch.long)


train_loader = DataLoader(
    TensorDataset(x_train_tensor, y1_train_tensor),
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(x_test_tensor, y1_test_tensor),
    batch_size=batch_size,
    shuffle=True
)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # 获取预测类别
    correct = (predicted == labels).sum().item()  # 计算正确的预测数量
    accuracy = correct / labels.size(0)  # 计算准确率
    return accuracy

# 训练循环
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

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

        train_accuracy = calculate_accuracy(outputs.view(-1, outputs.size(-1)), labels.view(-1) - 1)
        running_train_accuracy += train_accuracy

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    avg_train_accuracy = running_train_accuracy / len(train_loader)
    train_accuracies.append(avg_train_accuracy)

    # 计算测试精度
    model.eval()  # 设置模型为评估模式
    running_test_accuracy = 0.0
    test_loss = 0.0

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
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        avg_test_accuracy = running_test_accuracy / len(test_loader)
        test_accuracies.append(avg_test_accuracy)

    # 打印每个 epoch 的损失
    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Loss: {running_loss / len(train_loader):.4f} "
          f"Train Accuracy: {avg_train_accuracy * 100:.2f}%, "
          f"Test Accuracy: {avg_test_accuracy * 100:.2f}%"
          )

plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()