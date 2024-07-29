import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, in_channels=1, out_label=7):
        super(Simple1DCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv1d2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride,
                                 padding=padding)
        self.conv1d3 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=stride,
                                 padding=padding)
        self.fc = nn.Linear(1024 * 25, out_label)  # 输出大小调整为与标签相匹配
        self.dropout = nn.Dropout(0.3)  # 添加Dropout层，dropout率为0.3

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


class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=256, output_size=3):
        super(SimpleRNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_seq, hidden_state):
        rnn_out, hidden_state = self.rnn(input_seq, hidden_state)
        rnn_out = self.dropout(rnn_out)
        predictions = self.linear(rnn_out[:, -1, :])
        return predictions, hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(3, batch_size, self.hidden_layer_size)

# 3D CNN模型
class Simple3DCNN(nn.Module):
    def __init__(self, kernel_size=(3, 1, 1), stride=1, padding=1):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(128 * 1225, 512)
        self.fc2 = nn.Linear(512, 7)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=256, output_size=3):
        super(LSTM, self).__init__()
        self.layer_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.layer_size, batch_size, self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.layer_size, batch_size, self.hidden_layer_size).to(input_seq.device)

        lstm_out, (h0, c0) = self.lstm(input_seq, (h0, c0))

        predictions = self.linear(lstm_out[:, -1, :])

        return predictions

    def init_hidden(self, batch_size):
        return (torch.zeros(self.layer_size, batch_size, self.hidden_layer_size),
                torch.zeros(self.layer_size, batch_size, self.hidden_layer_size))