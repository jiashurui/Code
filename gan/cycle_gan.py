import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        generated_seq = self.fc(lstm_out)
        return generated_seq  # 输出维度: (batch_size, seq_len, output_dim)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(out)

def adversarial_loss(pred, target):
    return nn.BCELoss()(pred, target)

def cycle_consistency_loss(real, reconstructed):
    return nn.L1Loss()(real, reconstructed)
