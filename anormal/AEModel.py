from torch import nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=3):
        super(LSTMAutoencoder, self).__init__()

        # Class Parameter
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)

    def forward(self, x):
        # Encoder
        encoder_output, (encoder_h, encoder_c) = self.encoder_lstm(x)

        # Decoder
        decoder_output, (decoder_h, decoder_c) = self.decoder_lstm(encoder_output)

        # Result
        return decoder_output

class LSTMFCAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=3):
        super(LSTMFCAutoencoder, self).__init__()

        # Class Parameter
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.encoder_fc3 = nn.Linear(256, 128)
        self.encoder_fc4 = nn.Linear(128, 64)

        self.relu = nn.ReLU()
        # Decoder LSTM
        self.decoder_fc = nn.Linear(64, 128)
        self.decoder_fc2 = nn.Linear(128, 256)
        self.decoder_fc3 = nn.Linear(256, 512)
        self.decoder_fc4 = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)

    def forward(self, x):
        # Encoder
        encoder_output, (encoder_h, encoder_c) = self.encoder_lstm(x)
        latent_vector = self.encoder_fc(encoder_output)
        latent_vector = self.relu(latent_vector)
        latent_vector = self.encoder_fc2(latent_vector)
        latent_vector = self.relu(latent_vector)
        latent_vector = self.encoder_fc3(latent_vector)
        latent_vector = self.relu(latent_vector)
        latent_vector = self.encoder_fc4(latent_vector)
        latent_vector = self.relu(latent_vector)



        # Decoder
        latent_vector_out = self.decoder_fc(latent_vector)
        latent_vector_out = self.relu(latent_vector_out)
        latent_vector_out = self.decoder_fc2(latent_vector_out)
        latent_vector_out = self.relu(latent_vector_out)
        latent_vector_out = self.decoder_fc3(latent_vector_out)
        latent_vector_out = self.relu(latent_vector_out)
        latent_vector_out = self.decoder_fc4(latent_vector_out)
        latent_vector_out = self.relu(latent_vector_out)



        decoder_output, (decoder_h, decoder_c) = self.decoder_lstm(latent_vector_out)

        # Result
        return decoder_output

# CAE
class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvAutoencoder, self).__init__()

        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.upool = nn.MaxPool1d(2)

        # Encoder 516 dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=stride, padding=padding)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=stride, padding=padding)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=input_dim, kernel_size=3, stride=stride, padding=padding)
        # self.decoder_fc = nn.Linear(hidden_dim, latent_dim)

    def upsample(self, x):
        # 图像的上采样是双线性插值(mode='bilinear')
        return F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Decoder
        # 使用线性插值上采样
        x = self.conv4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.upsample(x)
        # Result
        return x