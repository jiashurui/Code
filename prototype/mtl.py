import torch
import torch.nn as nn
import torch.optim as optim

from prototype.constant import Constant
from prototype.dataReader import get_data_1d_3ch_child


# MTL
class MultiTask1DCNN(nn.Module):
    def __init__(self, seq_length):
        super(MultiTask1DCNN, self).__init__()

        # share CNN layer
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.shared_conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # BiLSTM 1
        self.lstm1 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        # BiLSTM 2
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)

        # task 1 FC
        self.task1_fc = nn.Sequential(
            nn.Linear(32 * (seq_length // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # run stop walk
        )

        # task 2 FC
        self.task2_fc = nn.Sequential(
            nn.Linear(32 * (seq_length // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 上下左右
        )

    def forward(self, x):
        # CNN
        shared_features = self.shared_conv(x)
        shared_features = self.shared_conv2(shared_features)

        # Flatten
        shared_features = shared_features.view(shared_features.size(0), -1)

        # LSTM
        task1_features, _ = self.lstm1(shared_features)
        task2_features, _ = self.lstm2(shared_features)

        # Multi-Task
        task1_output = self.task1_fc(shared_features)
        task2_output = self.task2_fc(shared_features)

        return task1_output, task2_output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
slide_window_length = 40  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
epochs = 200
batch_size = 128  # 或其他合适的批次大小
learning_rate = 0.00001
label_map = Constant.ChildWalk.action_map

# read data
train_data, train_labels, test_data, test_labels = get_data_1d_3ch_child(slide_window_length)
train_labels -= 1
test_labels -= 1

model = MultiTask1DCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataloader = []

# train
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels1, labels2 in dataloader:
        optimizer.zero_grad()

        outputs1, outputs2 = model(inputs)

        loss1 = criterion(outputs1, labels1)
        loss2 = criterion(outputs2, labels2)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")
