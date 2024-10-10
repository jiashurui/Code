import torch
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

        self.relu = nn.LeakyReLU()
        # self.relu = nn.ReLU()
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
        encoder_output = self.relu(latent_vector)

        # Decoder
        latent_vector_out = self.decoder_fc(encoder_output)
        latent_vector_out = self.relu(latent_vector_out)
        latent_vector_out = self.decoder_fc2(latent_vector_out)
        latent_vector_out = self.relu(latent_vector_out)
        latent_vector_out = self.decoder_fc3(latent_vector_out)
        latent_vector_out = self.relu(latent_vector_out)
        latent_vector_out = self.decoder_fc4(latent_vector_out)
        latent_vector_out = self.relu(latent_vector_out)

        decoder_output, (decoder_h, decoder_c) = self.decoder_lstm(latent_vector_out)

        # Result
        return decoder_output,encoder_output


# CAE
class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvAutoencoder, self).__init__()

        # Class Parameter
        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.upool = nn.MaxPool1d(2)

        # Encoder 516 dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride,
                               padding=padding)
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


class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=3):
        super(ConvLSTMAutoencoder, self).__init__()


# VAE based on MLP
# https://qiita.com/gensal/items/613d04b5ff50b6413aa0
class VAE(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        # Class Param
        self.relu = nn.LeakyReLU()
        # self.tanh = nn.Tanh()

        # Encoder
        self.lr = nn.Linear(input_dim, 1000)
        self.lr2 = nn.Linear(1000, 500)
        self.lr_ave = nn.Linear(500, z_dim)  # average
        self.lr_dev = nn.Linear(500, z_dim)  # log(sigma^2)

        # Decoder
        self.lr3 = nn.Linear(z_dim, 500)
        self.lr4 = nn.Linear(500, 1000)
        self.lr5 = nn.Linear(1000, input_dim)

    def forward(self, x):
        # Encoder
        x = self.lr(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        u = self.lr_ave(x)  # average μ
        log_sigma2 = self.lr_dev(x)  # log(sigma^2)

        ep = torch.randn_like(u)  # 平均0分散1の正規分布に従い生成されるz_dim次元の乱数
        z = u + torch.exp(log_sigma2 / 2) * ep  # 再パラメータ化トリック

        # Decoder
        x = self.lr3(z)
        x = self.relu(x)
        x = self.lr4(x)
        x = self.relu(x)
        x = self.lr5(x)
        # for reconstruction
        # x: model_output(reconstructed data)
        # z: encoder_output
        # u: latent_space(mean of distribution)
        # sigma2: standard deviation
        return x, z, u, log_sigma2

    def loss_function(self, recon, origin, ave, log_sigma2):
        # 重建Loss (reconstruction Loss)
        # TODO 记住,这里要用差值总和,不能用平均值,人类数据偏向非线性,用平均值会导致VAE重建糟糕,重建图像的时候会很模糊
        recon_loss = nn.MSELoss(reduction='sum')(recon, origin)
        # recon_loss = nn.BCELoss(reduction='sum')(recon, origin)

        # KL散度 (KL)
        kl_loss = -0.5 * torch.sum(1 + log_sigma2 - ave ** 2 - log_sigma2.exp())

        #  如果KL散度和重构损失不在一个尺度,则会有问题,要确保统一量纲
        # print(f'recon_loss: {recon_loss:.4f}, kl_loss:{kl_loss:.4f}')
        loss = recon_loss + 0.000001 * kl_loss
        return loss

class DeepVAE(VAE):
    def __init__(self, input_dim, z_dim):
        super().__init__()


    def forward(self, x):
        print(x.shape)
        return self.forward(x)

class LSTM_VAE(VAE):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=3):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)
