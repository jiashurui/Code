import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn, optim
from torch.utils.data import TensorDataset, random_split, DataLoader

from cnn.conv_lstm import ConvLSTM
from datareader.mh_datareader import simple_get_mh_all_features
from datareader.realworld_datareader import simple_get_realworld_all_features
from gan.cycle_gan import Generator, Discriminator
from prototype import constant
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

slice_length = 256
z_dim = 9       # 噪声向量的维度
output_dim = 9   # 时间序列的特征维度，即加速度和角速度
slice_length = 256
predict_dim = 8
hidden_dim = 30
# 定义 LSTM 分类模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM 输出：output 为每个时间步的隐藏状态，h_n 为最后时间步的隐藏状态
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # 取最后一层的隐藏状态作为全连接层输入
        return out


target_data = simple_get_realworld_all_features(slice_length, with_rpy=False, need_transform=False)

features = target_data[:, :, :9].to(device)
labels = target_data[:, :, 9][:, 0].to(torch.long).to(device)

# 数据划分比例
train_ratio = 0.8  # 80% 用于训练，20% 用于测试
total_samples = features.shape[0]
train_size = int(total_samples * train_ratio)
test_size = total_samples - train_size

# 创建 TensorDataset
dataset = TensorDataset(features, labels)

# 使用 random_split 划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
input_dim = 9         # 输入特征维度


# 初始化模型、损失函数和优化器
model = ConvLSTM(input_dim, predict_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例训练过程
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_labels in train_loader:
        # 将数据移动到 device 上
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        # 前向传播、计算损失和反向传播
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # 每个 epoch 后测试模型
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), "../model/eval_cycle_gan_lstm.pth")


G = Generator(z_dim, hidden_dim, output_dim).to(device)
F = Generator(z_dim, hidden_dim, output_dim).to(device)
D_X = Discriminator(output_dim, hidden_dim).to(device)
D_Y = Discriminator(output_dim, hidden_dim).to(device)

G.load_state_dict(torch.load('../model/cycle_g_generator.pth', map_location=device, weights_only=True))
F.load_state_dict(torch.load('../model/cycle_f_generator.pth', map_location=device, weights_only=True))

D_X.load_state_dict(torch.load('../model/cycle_dx_discriminator.pth', map_location=device, weights_only=True))
D_Y.load_state_dict(torch.load('../model/cycle_dy_discriminator.pth', map_location=device, weights_only=True))

# 生成 1000 个数据样本
num_samples = 1000
generated_data = []


# Origin Data
filtered_label = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
mapping = constant.Constant.simple_action_set.mapping_mh
###############################################################
origin_data = simple_get_mh_all_features(slice_length, type='np',
                                         filtered_label=filtered_label,
                                         mapping_label=mapping, with_rpy=False, need_transform=False)

# 去除标签
origin_data = origin_data[:,:,:9].astype(np.float32)


for _ in range(num_samples):
    # 随机选择一个样本作为输入
    random_sample = origin_data[np.random.choice(origin_data.shape[0])][np.newaxis, ...]
    random_sample_tensor = torch.tensor(random_sample, dtype=torch.float32).to(device)

    # 生成假数据
    with torch.no_grad():  # 关闭梯度计算，节省内存
        generated_sequence = G(random_sample_tensor)

    # 将生成的数据添加到列表中
    generated_data.append(generated_sequence[0].cpu().numpy())

# 将所有生成的数据堆叠成一个形状为 (1000, 256, 9) 的数组
generated_data = np.stack(generated_data)

# 假设 generated_data 是生成的 1000 个样本，形状为 (1000, 256, 9)
# 为生成的数据设置标签，全部标记为7
generated_data = torch.tensor(generated_data, dtype=torch.float32)  # 将数据转换为 Tensor
generated_labels = torch.full((generated_data.shape[0],), 7, dtype=torch.long)  # 标签全部为7

# 创建 TensorDataset 和 DataLoader
dataset = TensorDataset(generated_data, generated_labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 假设模型已经训练完毕，这里直接进行预测
model.eval()
all_preds = []

with torch.no_grad():
    for batch_data, _ in data_loader:
        batch_data = batch_data.to(device)
        outputs = model(batch_data)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())  # 将预测结果添加到列表中

# 统计标签为1的样本预测为1的准确率
accuracy = accuracy_score(generated_labels.numpy(), all_preds)
# 计算混淆矩阵
conf_matrix = confusion_matrix(generated_labels.numpy(), all_preds)
print("Confusion Matrix:")
print(conf_matrix)

print("Prediction accuracy on generated data:", accuracy)
