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
        return decoder_output,encoder_output


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
        self.encoder_fc2 = nn.Linear(64, 32)
        self.encoder_fc3 = nn.Linear(32, 16)
        self.encoder_fc4 = nn.Linear(16, 8)

        self.relu = nn.LeakyReLU()
        # self.relu = nn.ReLU()
        # Decoder LSTM
        self.decoder_fc = nn.Linear(8, 16)
        self.decoder_fc2 = nn.Linear(16, 32)
        self.decoder_fc3 = nn.Linear(32, 64)
        self.decoder_fc4 = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)
        # 初始化模型权重
        self.init_weights()

    def init_weights(self):
        # Xavier 初始化线性层
        nn.init.xavier_normal_(self.encoder_fc.weight)
        nn.init.xavier_normal_(self.encoder_fc2.weight)
        nn.init.xavier_normal_(self.encoder_fc3.weight)
        nn.init.xavier_normal_(self.encoder_fc4.weight)
        nn.init.xavier_normal_(self.decoder_fc.weight)
        nn.init.xavier_normal_(self.decoder_fc2.weight)
        nn.init.xavier_normal_(self.decoder_fc3.weight)
        nn.init.xavier_normal_(self.decoder_fc4.weight)

        # 对LSTM层的权重进行Xavier初始化
        for name, param in self.encoder_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for name, param in self.decoder_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

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


# 1D-CNN --> LSTM --> FC --> {Latent}--> FC -->LSTM --> 1D-CNN --> FC
class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, num_layers=6):
        super(ConvLSTMAutoencoder, self).__init__()
        # Class Parameter
        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.LeakyReLU()
        self.pool = nn.AvgPool1d(2)

        # Encoder
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        # LSTM
        self.encoder_lstm = nn.LSTM(64, 32, num_layers, batch_first=True, dropout=0,
                                    bidirectional=False)

        # FC
        self.encoder_fc = nn.Linear(32, 16)

        # Decoder
        # FC
        self.decoder_fc = nn.Linear(16, 32)

        # LSTM
        self.decoder_lstm = nn.LSTM(32, 64, num_layers, batch_first=True, dropout=0,
                                    bidirectional=False)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.up_conv1 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.up_conv2 = nn.ConvTranspose1d(256, 256, kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(in_channels=256, out_channels=input_dim, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.up_conv3 = nn.ConvTranspose1d(input_dim, input_dim, kernel_size=2, stride=2)


        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(32)
        self.layer_norm2 = nn.LayerNorm(64)

        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)

        self.decoder_batch_norm1 = nn.BatchNorm1d(128)
        self.decoder_batch_norm2 = nn.BatchNorm1d(256)

        self.fc_out = nn.Linear(input_dim, input_dim)
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.pool(x)

        # CNN 的第二个维度和第三个维度(batch_size, features, seq_len)
        # LSTM(batch_size, seq_len, features)
        x = x.transpose(1, 2)
        x, (encoder_h, encoder_c) = self.encoder_lstm(x)
        x = self.dropout(x)
        x = self.layer_norm1(x)
        latent = self.encoder_fc(x)

        # Decoder
        x = self.decoder_fc(latent)
        x, (decoder_h, decoder_c) = self.decoder_lstm(x)
        x = self.dropout(x)
        x = self.layer_norm2(x)

        x = x.transpose(1, 2)
        x = self.conv4(x)
        x = self.decoder_batch_norm1(x)
        x = self.up_conv1(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.decoder_batch_norm2(x)
        x = self.up_conv2(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.up_conv3(x)
        return x, latent

    def init_weights(self):
        # Xavier 初始化线性层
        nn.init.xavier_normal_(self.encoder_fc.weight)
        nn.init.xavier_normal_(self.decoder_fc.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

        # 对LSTM层的权重进行Xavier初始化
        for name, param in self.encoder_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for name, param in self.decoder_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
# VAE based on MLP
# https://qiita.com/gensal/items/613d04b5ff50b6413aa0
class VAE(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        # Class Param
        self.relu = nn.LeakyReLU()
        # self.tanh = nn.Tanh()

        # Encoder
        self.lr = nn.Linear(input_dim, 128)
        self.lr2 = nn.Linear(128, 64)
        self.lr_ave = nn.Linear(64, z_dim)  # average
        self.lr_dev = nn.Linear(64, z_dim)  # log(sigma^2)

        # Decoder
        self.lr3 = nn.Linear(z_dim, 64)
        self.lr4 = nn.Linear(64, 128)
        self.lr5 = nn.Linear(128, input_dim)

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
        loss = recon_loss + 0.1 * kl_loss
        return loss

# 混合高斯分布(TODO 确认数据符合哪种分布)
#  Variational Deep Embedding:
#  An Unsupervised and Generative Approach to Clustering（ICLR 2016）
class GMM_VAE(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()

class DeepVAE(VAE):
    def __init__(self, input_dim, z_dim):
        super().__init__()


    def forward(self, x):
        print(x.shape)
        return self.forward(x)

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()

        # Class Param
        self.relu = nn.LeakyReLU()
        # self.tanh = nn.Tanh()

        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0,
                                    bidirectional=True)

        self.lr = nn.Linear(hidden_dim * 2, 128)
        self.lr2 = nn.Linear(128, 64)
        self.lr_ave = nn.Linear(64, 32)  # average
        self.lr_dev = nn.Linear(64, 32)  # log(sigma^2)

        # Decoder
        self.lr3 = nn.Linear(32, 64)
        self.lr4 = nn.Linear(64, 128)
        self.lr5 = nn.Linear(128, hidden_dim)

        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=0,
                                    bidirectional=True)
        self.output_player = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x):
        # Encoder
        x, (encoder_h, encoder_c) = self.encoder_lstm(x)

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
        x = self.relu(x)

        x, (decoder_h, decoder_c) = self.decoder_lstm(x)
        x = self.output_player(x)
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
        loss = recon_loss + 0.02 * kl_loss
        return loss

class ConvLSTM_VAE(ConvLSTMAutoencoder):
    def __init__(self, input_dim, num_layers=6):
        super().__init__(input_dim, num_layers)
        self.lr_ave = nn.Linear(64, 32)  # average
        self.lr_dev = nn.Linear(64, 32)  # log(sigma^2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.pool(x)

        # CNN 的第二个维度和第三个维度(batch_size, features, seq_len)
        # LSTM(batch_size, seq_len, features)
        x = x.transpose(1, 2)
        x, (encoder_h, encoder_c) = self.encoder_lstm(x)
        x = self.dropout(x)
        x = self.layer_norm1(x)
        x = self.encoder_fc(x)

        # 变分
        u = self.lr_ave(x)  # average μ
        log_sigma2 = self.lr_dev(x)  # log(sigma^2)
        ep = torch.randn_like(u)  # 平均0分散1の正規分布に従い生成されるz_dim次元の乱数
        z = u + torch.exp(log_sigma2 / 2) * ep  # 再パラメータ化トリック


        # Decoder
        x = self.decoder_fc(z)
        x, (decoder_h, decoder_c) = self.decoder_lstm(x)
        x = self.dropout(x)
        x = self.layer_norm2(x)

        x = x.transpose(1, 2)
        x = self.conv4(x)
        x = self.decoder_batch_norm1(x)
        x = self.up_conv1(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.decoder_batch_norm2(x)
        x = self.up_conv2(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.up_conv3(x)
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
        loss = recon_loss + 0.02 * kl_loss
        return loss