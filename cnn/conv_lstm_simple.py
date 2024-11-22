from torch import nn


class ConvLSTM_SIMPLE(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3):
        super(ConvLSTM_SIMPLE, self).__init__()
        # Class Parameter
        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # CONV
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=8, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv8 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        # LSTM
        self.lstm = nn.LSTM(1024, 2048, num_layers, batch_first=True, dropout=0.1,
                                    bidirectional=True)
        # FC
        self.fc = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, output_dim)

        # dropout
        self.dropout = nn.Dropout(0.1)

        # Layer Norm
        self.layer_norm1 = nn.LayerNorm(4096)

        # Batch Norm
        self.batch_norm1 = nn.BatchNorm1d(8)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.batch_norm5 = nn.BatchNorm1d(128)
        self.batch_norm6 = nn.BatchNorm1d(256)
        self.batch_norm7 = nn.BatchNorm1d(512)
        self.batch_norm8 = nn.BatchNorm1d(1024)
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
        # 卷积部分
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(self.relu(self.batch_norm4(self.conv4(x))))
        x = self.pool(self.relu(self.batch_norm5(self.conv5(x))))
        x = self.pool(self.relu(self.batch_norm6(self.conv6(x))))
        x = self.pool(self.relu(self.batch_norm7(self.conv7(x))))
        x = self.pool(self.relu(self.batch_norm8(self.conv8(x))))

        # CNN 的第二个维度和第三个维度(batch_size, features, seq_len)
        # LSTM(batch_size, seq_len, features)
        x = x.transpose(1, 2)
        x, (hidden, cell) = self.lstm(x)
        x = self.dropout(x)

        x = self.layer_norm1(x)
        result = self.fc(x[:, -1, :])
        result = self.fc2(result)
        result = self.fc3(result)
        result = self.fc4(result)
        result = self.fc5(result)
        result = self.fc6(result)

        return result
