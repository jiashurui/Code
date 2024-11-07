import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(z_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z 的维度: (batch_size, seq_len, z_dim)
        lstm_out, _ = self.lstm(z)
        generated_seq = self.fc(lstm_out)
        return generated_seq  # 输出维度: (batch_size, seq_len, output_dim)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x 的维度: (batch_size, seq_len, input_dim)

        # 增加噪声
        x = x + torch.normal(0, 0.001, size=x.size()).to(x.device)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出来判断真假
        out = self.fc(lstm_out[:, -1, :])  # 输出维度: (batch_size, 1)
        return torch.sigmoid(out)
