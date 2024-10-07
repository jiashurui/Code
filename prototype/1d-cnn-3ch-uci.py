import numpy as np
import torch

import utils.report as report
import utils.show as show

from prototype.constant import Constant
from prototype.model import Simple1DCNN
from utils.uci_datareader import get_data_1d_uci

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
slide_window_length = 128  # 序列长度
stripe = int(slide_window_length * 0.5)  # overlap 50%
batch_size = 128  # 或其他合适的批次大小
learning_rate: float = 0.0001
label_map = Constant.RealWorld.label_map

test_data, test_labels = get_data_1d_uci()
# to GPU
test_data = test_data.to(device)
test_labels = test_labels.to(device)

# 实例化模型(加载模型参数)
model_load = Simple1DCNN(in_channels=3).to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN-3CH.pth'))

model_load.eval()
num_sum = 0
correct = 0
test_loss = 0
confusion_matrix = np.zeros((len(label_map), len(label_map)))

with torch.no_grad():
    for i in range(0, test_data.size()[0], batch_size):
        input_data, label = test_data[i: i + batch_size], test_labels[i: i + batch_size]
        if label.size(0) != batch_size:
            continue

        outputs = model_load(input_data)
        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引

        for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
            confusion_matrix[actual, expected] += 1
            if actual == expected:
                correct += 1

        num_sum += batch_size

print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

heatmap_plot = show.show_me_hotmap(confusion_matrix)
fig = heatmap_plot.gcf()
report.save_plot(heatmap_plot, 'heat-map')