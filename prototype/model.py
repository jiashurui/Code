import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(Simple1DCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv1d2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=stride,
                                 padding=padding)

        self.fc = nn.Linear(16 * 25, 7)  # 输出大小调整为与标签相匹配
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1d2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.softmax(x)
        return x