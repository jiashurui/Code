import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义一个简单的 LSTM 模型
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型
model = SimpleLSTM()

# 获取 LSTM 层的参数字典
lstm_params = model.lstm.state_dict()

# 提取权重和偏置
W_ih = lstm_params['weight_ih_l0']  # 输入到隐藏层的权重
W_hh = lstm_params['weight_hh_l0']  # 隐藏层到隐藏层的权重
b_ih = lstm_params['bias_ih_l0']    # 输入到隐藏层的偏置
b_hh = lstm_params['bias_hh_l0']    # 隐藏层到隐藏层的偏置

# 将张量转换为 NumPy 数组
W_ih = W_ih.detach().numpy()
W_hh = W_hh.detach().numpy()
b_ih = b_ih.detach().numpy()
b_hh = b_hh.detach().numpy()

# 获取隐藏单元数量
hidden_size = model.lstm.hidden_size

# 拆分权重矩阵（按行拆分为4个部分，对应4个门）
W_ii, W_if, W_ig, W_io = np.split(W_ih, 4, axis=0)
W_hi, W_hf, W_hg, W_ho = np.split(W_hh, 4, axis=0)

# 可视化输入门的输入权重
plt.figure(figsize=(6, 4))
plt.imshow(W_ii, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Input Gate Input Weights (W_ii)')
plt.xlabel('Input Features')
plt.ylabel('Hidden Units')
plt.show()

# 可视化遗忘门的输入权重
plt.figure(figsize=(6, 4))
plt.imshow(W_if, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Forget Gate Input Weights (W_if)')
plt.xlabel('Input Features')
plt.ylabel('Hidden Units')
plt.show()

# 同理可视化其他门的权重
