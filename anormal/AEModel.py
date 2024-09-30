from torch import nn


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
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=3):
        super(ConvAutoencoder, self).__init__()

        kernel_size = 3
        stride = 1
        padding = 1
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        # Encoder
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, stride=stride,
                                padding=padding)
        self.conv1d2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride,
                                 padding=padding)
        self.conv1d3 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=stride,
                                 padding=padding)

        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.conv4d = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=stride,
                                padding=padding)
        self.conv5d = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=stride,
                                padding=padding)
        self.conv6d = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=stride,
                                padding=padding)

        self.decoder_fc = nn.Linear(hidden_dim, latent_dim)

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