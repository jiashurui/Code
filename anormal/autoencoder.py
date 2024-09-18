import torch
import torch.nn as nn
import torch.optim as optim

from prototype.dataReader import get_data_1d_3ch_child


# LSTM Autoencoder Model
# Forward Input (batch_size, seq_length, dim)
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=3):
        super(LSTMAutoencoder, self).__init__()

        # Class Parameter
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=0, bidirectional=False)

    def forward(self, x):
        # Encoder
        encoder_output, (encoder_h, encoder_c) = self.encoder_lstm(x)

        # Decoder
        decoder_output, (decoder_h, decoder_c) = self.decoder_lstm(encoder_output)

        # Result
        return decoder_output


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 3  # Dimensionality of input sequence
hidden_dim = 128  # Hidden state size
latent_dim = 64  # Latent space size
num_layers = 1  # Number of LSTM layers
learning_rate = 0.0001  # Learning rate
epochs = 100  # Number of training epochs
slide_window_length = 20  # 序列长度

# Instantiate the model, loss function and optimizer
model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, slide_window_length, num_layers)
loss_function = nn.MSELoss()  # We use MSE loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy training data (for demonstration purposes)
batch_size = 16
# dummy_input = torch.randn(batch_size, sequence_length, input_dim)
train_data, train_labels, test_data, test_labels = get_data_1d_3ch_child(slide_window_length)
train_data = torch.transpose(train_data, 1, 2)
test_data = torch.transpose(test_data, 1, 2)


# Train
model.train()
lost_arr = []

for epoch in range(epochs):
    permutation = torch.randperm(train_data.size()[0])

    loss_per_epoch = 0.0
    for i in range(0, train_data.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        input_data, label = train_data[indices], train_labels[indices]

        # 输入长度不符合模型要求,则跳过这个输入
        if input_data.size(0) != batch_size:
            continue

        # 模型输出
        output = model(input_data)
        # 自己和重构后的自己比较
        loss = loss_function(output, input_data)
        loss_per_epoch = loss_per_epoch + loss.item()/batch_size

        # BP
        loss.backward()
        optimizer.step()

    lost_arr.append(loss_per_epoch)
    print('epoch: {}, loss: {}'.format(epoch, loss_per_epoch))

print("Training complete.")


# save my model
torch.save(model.state_dict(), '../model/autoencoder.pth')


# Test
