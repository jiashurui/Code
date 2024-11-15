from torch import nn


class ConvLSTM_SIMPLE(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3):
        super(ConvLSTM_SIMPLE, self).__init__()
        # Class Parameter
        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # CONV
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        # LSTM
        self.lstm = nn.LSTM(64, 128, num_layers, batch_first=True, dropout=0.1,
                                    bidirectional=True)
        # FC
        self.fc = nn.Linear(256, output_dim)

        # dropout
        self.dropout = nn.Dropout(0.1)

        # Layer Norm
        self.layer_norm1 = nn.LayerNorm(256)

        # Batch Norm
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(64)

        # Init Weight
        self.init_weights()

    def init_weights(self):
        # Xavier 初始化线性层
        nn.init.xavier_normal_(self.fc.weight)

        # 对LSTM层的权重进行Xavier初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

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

        # CNN 的第二个维度和第三个维度(batch_size, features, seq_len)
        # LSTM(batch_size, seq_len, features)
        x = x.transpose(1, 2)
        x, (hidden, cell) = self.lstm(x)
        x = self.dropout(x)
        x = self.layer_norm1(x)
        result = self.fc(x[:, -1, :])
        return result
