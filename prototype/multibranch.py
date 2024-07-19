import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



# 定义多分支的CNN-BiLSTM模型
class CNNBiLSTM(nn.Module):
    def __init__(self, num_channels, seq_length, num_classes):
        super(CNNBiLSTM, self).__init__()

        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 双向LSTM层
        self.bilstm = nn.LSTM(128, 64, bidirectional=True, batch_first=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * seq_length // 4, 256),  # 这里需要根据池化层后的序列长度调整
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        # LSTM层
        x, _ = self.bilstm(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


# 超参数
num_channels = 6  # 传感器通道数量
seq_length = 128  # 每个样本的时间序列长度
num_classes = 10  # 类别数量
batch_size = 32
num_epochs = 20
learning_rate = 0.001

model = CNNBiLSTM(num_channels, seq_length, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

dataloader = []

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")
