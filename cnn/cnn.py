import torch
import torch.nn as nn


class DeepOneDimCNN(nn.Module):
    def __init__(self, in_channels=9, out_channel=8):
        super(DeepOneDimCNN, self).__init__()

        # Class param
        self.kernel_size = 7
        self.stride = 1
        self.padding = 3
        self.in_channels = in_channels
        self.out_label = out_channel

        self.conv1d = nn.Conv1d(in_channels=self.in_channels, out_channels=32, kernel_size=self.kernel_size, stride=self.stride,
                                padding=self.padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv1d2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding)
        self.conv1d3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding)
        self.fc = nn.Linear(128 * 16, self.out_label)  # 输出大小调整为与标签相匹配
        self.dropout = nn.Dropout(0.02)  # 添加Dropout层，dropout率为0.3

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1d2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1d3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x
